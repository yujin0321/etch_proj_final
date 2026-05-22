"""
GraphRAG Agent v2 — Two-stage retrieval + SHAP fingerprint matching

Token optimization techniques (estimated 40-60 percent saving vs v1):
  1. SHAP context: JSON dump -> compact natural-language line (~25 vs ~120 tokens)
  2. SOP: abstract first, full steps fetched only on follow-up (~70 vs ~400 tokens)
  3. Cypher returns trimmed property set, not whole node dicts
  4. System prompt: 4-bullet guideline -> 2-line role + format directive

Retrieval design (paper-grounded):
  Stage 1 (Coarse):
     SHAP top-k sensors -> match against Fault.sensors_low/high fingerprints
     -> rank candidate Faults by overlap score
     -> aggregate up to FaultCategory if no single fault dominates
  Stage 2 (Fine, only if user asks for detail):
     For top-1 fault, traverse Fault -> Mechanism -> Component -> SOP (abstract)
     Full SOPStep fetch only on explicit user request

References:
  - Wise 1999 fault catalog -> fingerprint matching
  - Sofge 1997 g_model -> wafer-state risk signals
  - Microsoft GraphRAG (Edge et al. 2024) -> coarse-to-fine pattern
"""

import os
from typing import Optional, Dict, List, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from dotenv import load_dotenv
from etch_fault_prompts import FAULT_KNOWLEDGE, build_prompt

load_dotenv()


# ------------------------------------------------------------------
# SHAP -> compact natural language (token saver #1)
# ------------------------------------------------------------------
def _iter_shap_items(shap_analysis: Optional[Any]):
    if not shap_analysis:
        return []
    if isinstance(shap_analysis, dict):
        return list(shap_analysis.items())
    if isinstance(shap_analysis, list):
        return [
            (
                item.get("sensor") or item.get("base_sensor") or f"sensor_{idx}",
                item,
            )
            for idx, item in enumerate(shap_analysis)
            if isinstance(item, dict)
        ]
    return []


def compact_shap(shap_analysis: Optional[Any]) -> str:
    """
    Convert SHAP analysis into a single short line.
    """
    items = _iter_shap_items(shap_analysis)
    if not items:
        return "No SHAP analysis provided."
    ranked = sorted(items, key=lambda kv: kv[1].get("rank", 99))
    parts = []
    for sensor, info in ranked[:5]:  # cap at top-5; Sofge showed >5 vars degrade models
        direction = info.get("direction", "?")
        value = info.get("value", info.get("shap_value", 0.0))
        rank = info.get("rank", "?")
        sign = "+" if value >= 0 else ""
        parts.append(f"{sensor} ({direction}, {sign}{value:.2f}, rank {rank})")
    return "Top SHAP deviations: " + "; ".join(parts)


def parse_shap_to_lists(shap_analysis: Optional[Any]) -> Tuple[List[str], List[str]]:
    """Split SHAP into (low_sensors, high_sensors) for Cypher matching.
    SHAP dict often uses underscores; KG uses spaces. Normalize.
    """
    low, high = [], []
    for _, info in _iter_shap_items(shap_analysis):
        sensor_name = (info.get("sensor") or info.get("base_sensor") or "").replace("_", " ")
        d = info.get("direction", "").lower()
        if d == "low":
            low.append(sensor_name)
        elif d == "high":
            high.append(sensor_name)
    return low, high


def _sensor_group(sensor: str) -> str:
    if isinstance(sensor, str) and sensor.replace(".", "", 1).isdigit():
        return "OES"
    if isinstance(sensor, str) and sensor.startswith("S") and any(ch in sensor for ch in ("V", "I")):
        return "RFM"
    return "MACHINE"


def _escape_prompt_template(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


class GraphRAGAgentV2:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Default to mini for cost.
        # Switch to gpt-4o only when the user asks for deep root-cause analysis.
        self.llm = ChatOpenAI(model=model_name, temperature=0)

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Stage 1 — fingerprint matching
    # ------------------------------------------------------------------
    def stage1_match_fault(self, shap_analysis: Dict[str, Any],
                           top_n: int = 3) -> List[Dict]:
        low, high = parse_shap_to_lists(shap_analysis)
        if not low and not high:
            return []

        query = """
        MATCH (f:Fault)
        WITH f,
             size([s IN f.sensors_low  WHERE s IN $low ]) AS low_hits,
             size([s IN f.sensors_high WHERE s IN $high]) AS high_hits,
             size(f.sensors_low) + size(f.sensors_high)   AS fp_size
        WITH f, low_hits + high_hits AS hits, fp_size
        WHERE hits > 0
        WITH f, hits, fp_size,
             toFloat(hits) /
               CASE fp_size WHEN 0 THEN 1 ELSE fp_size END AS score
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(cat:FaultCategory)
        RETURN f.fault_id    AS fault_id,
               f.name        AS name,
               f.induced_via AS induced_via,
               f.region      AS region,
               cat.name      AS category,
               score, hits
        ORDER BY score DESC, hits DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            result = session.run(query, low=low, high=high, top_n=top_n)
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Stage 2 — causal chain + SOP abstract (no steps yet)
    # ------------------------------------------------------------------
    def stage2_fetch_chain(self, fault_id: str) -> Dict:
        query = """
        MATCH (f:Fault {fault_id:$fid})
        OPTIONAL MATCH (f)-[:CAUSED_BY]->(m:Mechanism)
        OPTIONAL MATCH (m)-[:DEGRADES]->(c:Component)
        OPTIONAL MATCH (m)-[:REMEDIATED_BY]->(sop:SOP)
        RETURN f.name      AS fault_name,
               f.region    AS region,
               f.reference AS fault_ref,
               collect(DISTINCT {
                 mechanism: m.name,
                 timescale: m.timescale,
                 symptom: m.symptom,
                 reference: m.reference
               }) AS mechanisms,
               collect(DISTINCT c.name) AS components,
               collect(DISTINCT {
                 sop_id: sop.sop_id,
                 title: sop.title,
                 abstract: sop.abstract,
                 reference: sop.reference
               }) AS sops
        """
        with self.driver.session() as session:
            result = session.run(query, fid=fault_id).single()
            if not result:
                return {}
            return dict(result)

    def fetch_sop_steps(self, sop_id: str) -> List[Dict]:
        """Expensive: only call when user asks for full procedure."""
        query = """
        MATCH (s:SOP {sop_id:$sop_id})-[:HAS_STEP]->(st:SOPStep)
        RETURN st.step_no AS step_no, st.text AS text
        ORDER BY st.step_no
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, sop_id=sop_id)]

    def _stats_from_shap_analysis(self, shap_analysis: Optional[Any]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        stats: Dict[str, Dict[str, Tuple[float, float]]] = {"MACHINE": {}, "OES": {}, "RFM": {}}
        if not shap_analysis:
            return stats

        for _, info in _iter_shap_items(shap_analysis):
            sensor = (info.get("sensor") or info.get("base_sensor") or "").replace("_", " ")
            if not sensor:
                continue
            group = _sensor_group(sensor)
            mean_value = info.get("mean_value")
            current_value = info.get("current_value")
            std_value = info.get("std", 0.0)
            if mean_value is None and current_value is not None:
                mean_value = current_value
            if mean_value is None:
                continue
            try:
                stats[group][sensor] = (float(mean_value), float(std_value))
            except (TypeError, ValueError):
                continue
        return stats

    def _normalize_fault_label(self, fault_name: str) -> str:
        normalized = fault_name.strip()
        if normalized in FAULT_KNOWLEDGE:
            return normalized
        fault_lower = normalized.lower()
        for candidate in FAULT_KNOWLEDGE:
            if candidate.lower() == fault_lower:
                return candidate
        for candidate in FAULT_KNOWLEDGE:
            if candidate.lower().startswith(fault_lower) or fault_lower in candidate.lower():
                return candidate
        return fault_name

    def _build_prompt_messages(self,
                               fault_name: str,
                               kg_context: str,
                               shap_line: str,
                               question: str,
                               shap_analysis: Optional[Dict[str, Any]] = None,
                               fallback: bool = False) -> List[Tuple[str, str]]:
        system_prompt = "You are a senior semiconductor maintenance expert for the Lam 9600 metal etcher. "
        system_prompt += "Use the retrieved Knowledge Graph context and SHAP signals to answer in Korean. "
        system_prompt += "Structure the answer as a short fault summary, root cause, and recommended corrective actions."
        if fallback:
            system_prompt += " If Knowledge Graph data is missing or insufficient, rely on domain-specific plasma etch heuristics and SHAP evidence to generate safe first-line corrective actions."

        try:
            target_label = self._normalize_fault_label(fault_name)
            stats = self._stats_from_shap_analysis(shap_analysis)
            system_prompt, user_prompt = build_prompt(target_label, stats, fallback=fallback)

            # Preserve the original fault name in the top section when labels differ.
            if target_label != fault_name:
                user_prompt = user_prompt.replace(
                    f"[Classified Fault]\n{target_label}",
                    f"[Classified Fault]\n{fault_name}"
                )
        except Exception:
            user_prompt = (
                f"Detected fault candidate: {fault_name}\n\n"
                f"Knowledge Graph context:\n{kg_context}\n\n"
                f"SHAP summary:\n{shap_line}\n\n"
                f"Question: {question}\n\n"
                "Use the Knowledge Graph context and SHAP evidence together to recommend the most likely cause and corrective actions in Korean."
            )
        else:
            user_prompt = (
                f"{user_prompt}\n\n"
                "[Knowledge Graph Context]\n"
                f"{kg_context}\n\n"
                "[SHAP Evidence]\n"
                f"{shap_line}\n\n"
                "[Instruction]\n"
                "Knowledge Card 기반 대응 절차와 GraphRAG에서 가져온 추가 컨텍스트를 함께 사용하여, "
                "현재 센서 이상과 가장 직접적으로 연결되는 조치를 우선 제시하고, 실제 적용 전 자사 SOP 우선 확인을 명시하세요."
            )
            if fallback:
                user_prompt += (
                    "\n\n[Knowledge Gap Fallback]\n"
                    "Knowledge Graph가 충분한 세부 SOP를 제공하지 못하는 경우, "
                    "etch_fault_prompts.py에 담긴 예방 및 점검 원칙을 활용하여 안전한 1차 대응을 우선 제시하십시오. "
                    "이 때 불확실성을 명시하되, 절대 '아무 것도 모른다'처럼 답변하지 마십시오."
                )

        return [
            ("system", _escape_prompt_template(system_prompt)),
            ("user", _escape_prompt_template(user_prompt)),
        ]

    # ------------------------------------------------------------------
    # Format chain to compact context (token saver #2)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_chain(chain: Dict) -> str:
        if not chain:
            return "No causal chain found."
        lines = [
            f"Fault: {chain.get('fault_name','?')} "
            f"(region: {chain.get('region','?')}, ref: {chain.get('fault_ref','?')})"
        ]
        mechs = [m for m in chain.get("mechanisms", []) if m.get("mechanism")]
        if mechs:
            lines.append("Mechanisms:")
            for m in mechs:
                lines.append(
                    f"  - {m['mechanism']} "
                    f"(timescale: {m.get('timescale','?')}; "
                    f"symptom: {m.get('symptom','?')})"
                )
        comps = [c for c in chain.get("components", []) if c]
        if comps:
            lines.append(f"Affected components: {', '.join(comps)}")
        sops = [s for s in chain.get("sops", []) if s.get("sop_id")]
        if sops:
            lines.append("Recommended SOPs:")
            for s in sops:
                lines.append(f"  - [{s['sop_id']}] {s['title']}: {s['abstract']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def recommend(self,
                  shap_analysis: Optional[Dict[str, Any]] = None,
                  fault_name_hint: Optional[str] = None,
                  question: str = "What is the recommended countermeasure?",
                  include_full_steps: bool = False) -> Dict:
        """
        Returns:
            'candidates'      : stage-1 ranking
            'chain'           : stage-2 causal chain for top-1
            'answer'          : LLM final text
            'token_estimate'  : rough context token count
        """
        # Stage 1
        candidates = self.stage1_match_fault(shap_analysis or {}, top_n=3)
        if not candidates and fault_name_hint:
            with self.driver.session() as session:
                rec = session.run(
                    "MATCH (f:Fault) WHERE f.name CONTAINS $name "
                    "RETURN f.fault_id AS fault_id LIMIT 1",
                    name=fault_name_hint
                ).single()
                if rec:
                    candidates = [{"fault_id": rec["fault_id"],
                                   "name": fault_name_hint,
                                   "score": 0.0}]

        if not candidates:
            # Knowledge gap fallback: no fingerprint match in KG, but we may still have a known label.
            if fault_name_hint:
                kg_context = "No Knowledge Graph match found for this fault. Use fallback domain heuristics."
                shap_line = compact_shap(shap_analysis)
                prompt = ChatPromptTemplate.from_messages(
                    self._build_prompt_messages(
                        fault_name_hint,
                        kg_context,
                        shap_line,
                        question,
                        shap_analysis,
                        fallback=True,
                    )
                )
                answer = (prompt | self.llm | StrOutputParser()).invoke({})
                return {
                    "candidates": [],
                    "chain": {},
                    "answer": answer,
                    "token_estimate": len(kg_context + shap_line) // 4,
                }
            return {
                "candidates": [],
                "chain": {},
                "answer": "SHAP에 매칭되는 fault가 KG에 없습니다. 센서 이름 정규화를 확인하세요.",
                "token_estimate": 0,
            }

        # Stage 2 for top-1
        top = candidates[0]
        chain = self.stage2_fetch_chain(top["fault_id"])

        full_steps_text = ""
        if include_full_steps and chain.get("sops"):
            sop_id = chain["sops"][0]["sop_id"]
            steps = self.fetch_sop_steps(sop_id)
            full_steps_text = "\n\nFull SOP steps:\n" + "\n".join(
                f"  {s['step_no']}. {s['text']}" for s in steps
            )

        kg_context = self._format_chain(chain) + full_steps_text
        shap_line = compact_shap(shap_analysis)
        token_estimate = (len(kg_context) + len(shap_line)) // 4

        try:
            prompt = ChatPromptTemplate.from_messages(
                self._build_prompt_messages(
                    top["name"],
                    kg_context,
                    shap_line,
                    question,
                    shap_analysis,
                )
            )
            answer = (prompt | self.llm | StrOutputParser()).invoke({})
        except Exception as e:
            answer = f"LLM error: {e}"

        return {
            "candidates": candidates,
            "chain": chain,
            "answer": answer,
            "token_estimate": token_estimate,
        }


class GraphRAGAgent:
    """Compatibility wrapper for older call sites that expect get_recommendation."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._inner = GraphRAGAgentV2(model_name=model_name)

    def get_recommendation(self, fault_name: str, shap_analysis: Optional[Dict[str, Any]] = None) -> str:
        result = self._inner.recommend(
            shap_analysis=shap_analysis,
            fault_name_hint=fault_name,
            question=f"Describe the most likely cause and recommended corrective actions for {fault_name}.",
            include_full_steps=False,
        )
        return result.get("answer", "No recommendation generated")

    def close(self):
        self._inner.close()


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = GraphRAGAgentV2(model_name="gpt-4o-mini")
    try:
        # Wise 1999 Fault #9 (He Chuck) pattern
        shap = {
            "Helium_Pressure":  {"value": -2.8, "direction": "Low",  "rank": 1},
            "Chamber_Pressure": {"value":  0.4, "direction": "High", "rank": 2},
        }
        out = agent.recommend(
            shap_analysis=shap,
            question="이 이상 징후의 원인과 점검 절차를 알려주세요."
        )
        print("=" * 60)
        print("Stage-1 candidates:")
        for c in out["candidates"]:
            print(f"  {c['fault_id']} {c['name']:15s} "
                  f"score={c['score']:.2f} cat={c.get('category')}")
        print(f"\nContext tokens (est): {out['token_estimate']}")
        print("\nLLM answer:\n")
        print(out["answer"])
    finally:
        agent.close()
