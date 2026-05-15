from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
from dotenv import load_dotenv
from etch_fault_prompts import build_prompt

# Load environment variables
load_dotenv()

if "OPEN_AI_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]

class SHAPAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(self._build_prompt_messages("{fault_name}", "{analysis_json}"))
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_prompt_messages(self, fault_name, analysis_results):
        try:
            stats = self._stats_from_shap_analysis(analysis_results)
            
            # Map common generic names or handle missing keys
            target_label = fault_name
            if fault_name == "감지된 결함" or fault_name == "UNKNOWN FAULT":
                # Use a representative key for generic analysis if needed, 
                # or just use the first available one as a structure template.
                # Here we try to find the closest match or use a generic behavior.
                target_label = "TCP +10" # Default template for unknown faults
            
            system_prompt, user_prompt = build_prompt(target_label, stats)
            
            # If it was originally unknown, override the label in the user prompt
            if fault_name in ["감지된 결함", "UNKNOWN FAULT"]:
                user_prompt = user_prompt.replace(f"[Classified Fault]\n{target_label}", f"[Classified Fault]\n{fault_name} (미분류 이상)")

            shap_json = json.dumps(analysis_results, indent=2, ensure_ascii=False)
            user_prompt = (
                f"{user_prompt}\n\n"
                "[SHAP Evidence — model-local explanation]\n"
                f"{shap_json}\n\n"
                "[Additional Instruction]\n"
                "SHAP Evidence를 우선 증거로 사용해 Knowledge Card의 일반 절차 중 "
                "현재 센서 패턴과 가장 직접적으로 연결되는 조치를 먼저 제시하시오."
            )
            return [
                ("system", self._escape_prompt_template(system_prompt)),
                ("user", self._escape_prompt_template(user_prompt)),
            ]
        except Exception as e:
            print(f"❌ SHAPAgent build_prompt failed for '{fault_name}': {e}")
            return [
                ("system", """You are an expert Semiconductor Process Engineer.
Your task is to analyze sensor data anomalies based on SHAP values and raw sensor statistics.

Context:
- Process: Plasma Etching
- You will receive a list of top sensors that contributed to a fault detection.
- Each sensor entry includes: Current Value, Mean (Normal), Normal Range, Status (High/Low), and SHAP Direction.

Guidelines:
1. Explain the physical impact of the sensor being 'High' or 'Low' for the specific fault.
2. Link the sensor status to potential equipment issues (e.g., "Pressure is High, suggesting a Throttle Valve or Vacuum Pump issue").
3. Use the SHAP direction to confirm if the sensor's current state is what primarily drove the AI's decision.
4. Output should be in Korean.
5. 리포트 서두에 '알려지지 않은 결함'이라는 표현을 쓰지 마십시오. 대신 '감지된 결함' 또는 단순히 '결함'이라는 표현을 사용하고, '결함에 대한 센서 분석 결과를 바탕으로 기술적 분석을 제공합니다'와 같은 정중하고 전문적인 문구로 시작하십시오."""),
                ("user", "Detected Fault: {fault_name}\nSensor Analysis Data:\n{analysis_json}\n\nPlease provide a deep technical analysis.")
            ]

    def _stats_from_shap_analysis(self, analysis_results):
        stats = {"MACHINE": {}, "OES": {}, "RFM": {}}
        if not isinstance(analysis_results, list):
            return stats

        for item in analysis_results:
            sensor = item.get("base_sensor") or item.get("sensor")
            if not sensor:
                continue
            value = item.get("current_value", item.get("mean_value", 0.0))
            std = item.get("std", 0.0)
            group = self._sensor_group(sensor)
            try:
                stats[group][sensor] = (float(value), float(std))
            except (TypeError, ValueError):
                continue
        return stats

    @staticmethod
    def _escape_prompt_template(text):
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _sensor_group(sensor):
        if sensor.replace(".", "", 1).isdigit():
            return "OES"
        if sensor.startswith("S") and any(ch in sensor for ch in ("V", "I")):
            return "RFM"
        return "MACHINE"

    def explain_fault(self, fault_name, analysis_results):
        """
        analysis_results: list of dicts from SHAPExplainer.explain
        """
        # Convert list to a readable string for the LLM
        analysis_str = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        
        try:
            prompt = ChatPromptTemplate.from_messages(
                self._build_prompt_messages(fault_name, analysis_results)
            )
            chain = prompt | self.llm | StrOutputParser()
            explanation = chain.invoke({
                "fault_name": fault_name,
                "analysis_json": analysis_str
            })
            return explanation
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

if __name__ == "__main__":
    agent = SHAPAgent()
    print("SHAPAgent updated.")
