import importlib
import sys
import unittest
import asyncio
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FinalPipelineWiringTest(unittest.TestCase):
    def test_server_points_to_final_runtime_assets(self):
        server = importlib.import_module("server")

        self.assertEqual(server.MODEL_DIR, "models_final")
        self.assertEqual(server.LOCAL_SIMULATION_CSV, "data/test_v6_20.csv")
        self.assertEqual(server.INFERENCE_ENGINE_MODULE, "modeling_final")

    def test_modeling_final_defaults_to_final_model_dir(self):
        modeling_final = importlib.import_module("modeling_final")

        self.assertEqual(modeling_final.InferenceEngine.__init__.__defaults__[0], "models_final")

    def test_shap_analysis_defaults_to_final_sensor_stats(self):
        shap_analysis = importlib.import_module("shap_analysis")

        self.assertEqual(
            shap_analysis.SHAPExplainer.__init__.__defaults__[0],
            "models_final/sensor_stats.json",
        )

    def test_retrain_exposes_sensor_stats_builder(self):
        retrain = importlib.import_module("re_train_models_final")

        self.assertTrue(hasattr(retrain, "build_sensor_stats"))

    def test_known_fault_prompt_uses_etch_fault_prompts(self):
        captured = {}

        class FakePrompt:
            @classmethod
            def from_messages(cls, messages):
                captured["messages"] = messages
                return cls()

            def __or__(self, other):
                return self

        class FakeLLM:
            def __or__(self, other):
                return self

        class FakeParser:
            pass

        with patch("agents.shap_agent.ChatPromptTemplate", FakePrompt), \
             patch("agents.shap_agent.ChatOpenAI", lambda *args, **kwargs: FakeLLM()), \
             patch("agents.shap_agent.StrOutputParser", lambda: FakeParser()):
            from agents.shap_agent import SHAPAgent

            messages = SHAPAgent()._build_prompt_messages(
                "RF -12",
                [{"sensor": "RF Btm Rfl Pwr", "current_value": 35.2, "status": "High"}],
            )

        system_message = messages[0][1]
        user_message = messages[1][1]

        self.assertIn("FDC(Fault Detection & Classification)", system_message)
        self.assertIn("[SHAP Evidence", user_message)

    def test_known_fault_prompt_treats_shap_json_as_literal_text(self):
        from agents.shap_agent import SHAPAgent
        from langchain_core.prompts import ChatPromptTemplate

        agent = SHAPAgent.__new__(SHAPAgent)
        messages = agent._build_prompt_messages(
            "RF -12",
            [{"sensor": "RF Btm Rfl Pwr", "current_value": 35.2, "status": "High"}],
        )
        prompt = ChatPromptTemplate.from_messages(messages)

        rendered = prompt.invoke({})

        self.assertIn("RF Btm Rfl Pwr", rendered.messages[1].content)

    def test_llm_report_keeps_shap_explanation_when_rag_fails(self):
        server = importlib.import_module("server")

        async def run():
            async def shap_call():
                return "SHAP explanation generated"

            async def rag_call():
                raise RuntimeError("neo4j unavailable")

            return await server._collect_llm_outputs(shap_call(), rag_call())

        explanation, recommendation = asyncio.run(run())

        self.assertEqual(explanation, "SHAP explanation generated")
        self.assertIn("Neo4j/RAG", recommendation)


if __name__ == "__main__":
    unittest.main()
