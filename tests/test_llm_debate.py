import unittest

from fii_analytics.analysis.llm_debate import (
    OpenRouterClient,
    filter_free_openrouter_models,
    filter_paid_openrouter_models,
    parse_judge_response,
    robust_model_subset,
)


class LLMDebateTest(unittest.TestCase):
    def test_parse_judge_response_from_json_fence(self):
        parsed = parse_judge_response(
            """```json
{"vencedor": "Ativo A", "ativo_vencedor": "Oferta 1", "resumo": "Argumento mais claro."}
```"""
        )
        self.assertEqual(parsed["vencedor"], "Ativo A")
        self.assertEqual(parsed["ativo_vencedor"], "Oferta 1")

    def test_parse_judge_response_from_escaped_json(self):
        parsed = parse_judge_response(
            r'{\"vencedor\": \"Ativo B\", \"ativo_vencedor\": \"Oferta 2\", \"resumo\": \"Melhor risco-retorno.\"}'
        )
        self.assertEqual(parsed["vencedor"], "Ativo B")
        self.assertEqual(parsed["ativo_vencedor"], "Oferta 2")

    def test_parse_judge_response_from_truncated_escaped_json(self):
        parsed = parse_judge_response(
            r'{\"vencedor\": \"Ativo B\", \"ativo_vencedor\": \"HGBS11\", \"resumo\": \"O argumento B foi mais objetivo'
        )
        self.assertEqual(parsed["vencedor"], "Ativo B")
        self.assertEqual(parsed["ativo_vencedor"], "HGBS11")
        self.assertIn("recuperados parcialmente", parsed["alerta"])

    def test_openrouter_client_ignores_environment_proxy(self):
        client = OpenRouterClient(api_key="test")
        self.assertFalse(client.session.trust_env)
        retries = client.session.adapters["https://"].max_retries
        self.assertNotIn(429, retries.status_forcelist)

    def test_filter_free_openrouter_models(self):
        payload = {
            "data": [
                {"id": "paid/model", "pricing": {"prompt": "0.1", "completion": "0.2"}},
                {"id": "free/by_suffix:free", "pricing": {"prompt": "0.1", "completion": "0.2"}},
                {"id": "free/by_price", "pricing": {"prompt": "0", "completion": "0"}},
            ]
        }
        self.assertEqual(filter_free_openrouter_models(payload), ["free/by_suffix:free"])

    def test_filter_paid_openrouter_models(self):
        payload = {"data": [{"id": "paid/model"}, {"id": "free/model:free"}]}
        self.assertEqual(filter_paid_openrouter_models(payload), ["paid/model"])

    def test_robust_model_subset_prefers_high_parameter_models(self):
        models = [
            "mistral/7b:free",
            "openai/gpt-oss-120b:free",
            "qwen/qwen3-30b-a3b:free",
            "qwen/qwen3-235b-a22b:free",
        ]
        self.assertEqual(
            robust_model_subset(models, minimum_b=70),
            ["qwen/qwen3-235b-a22b:free", "openai/gpt-oss-120b:free"],
        )


if __name__ == "__main__":
    unittest.main()
