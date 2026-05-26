"""LLM debate helpers for comparing primary offerings."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fii_analytics.config import Settings, clean_secret, settings


DEFAULT_FREE_OPENROUTER_MODELS = [
    "qwen/qwen3-235b-a22b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "moonshotai/kimi-k2:free",
    "qwen/qwen3-30b-a3b:free",
    "poolside/laguna-xs.2:free",
]

DEFAULT_PAID_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-3.3-70b-instruct",
]


@dataclass(frozen=True)
class DebateArgument:
    model: str
    asset_name: str
    content: str


@dataclass(frozen=True)
class DebateResult:
    argument_1: DebateArgument
    argument_2: DebateArgument
    judge_model: str
    verdict: dict[str, Any]


class OpenRouterRateLimitError(RuntimeError):
    """Raised when OpenRouter refuses a request because of rate limits."""


class OpenRouterPaymentRequiredError(RuntimeError):
    """Raised when OpenRouter requires credits or billing for a request."""


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, config: Settings = settings):
        self.api_key = clean_secret(api_key) or clean_secret(getattr(config, "openrouter_api_key", None))
        self.api_url = getattr(config, "openrouter_api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.models_url = "https://openrouter.ai/api/v1/models"
        self.timeout = max(getattr(config, "request_timeout", 30), 60)
        self.session = build_openrouter_session()

    def query(self, model: str, messages: list[dict[str, str]], max_tokens: int = 1100) -> str:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY nao configurada.")

        from fii_analytics.analysis.chains import _openrouter_llm

        try:
            response = _openrouter_llm(model, api_key=self.api_key, max_tokens=max_tokens).invoke(
                [(message.get("role", "user"), message.get("content", "")) for message in messages]
            )
        except Exception as exc:
            _raise_openrouter_specific_error(exc)
            raise
        return str(getattr(response, "content", "")).strip()

    def list_free_models(self) -> list[str]:
        return filter_free_openrouter_models(self._models_payload())

    def list_paid_models(self) -> list[str]:
        return filter_paid_openrouter_models(self._models_payload())

    def _models_payload(self) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.get(self.models_url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def filter_free_openrouter_models(payload: dict[str, Any]) -> list[str]:
    free_models = []
    for model in payload.get("data", []) or []:
        model_id = str(model.get("id", "")).strip()
        pricing = model.get("pricing") or {}
        if model_id.endswith(":free"):
            free_models.append(model_id)
    return sort_models_by_capacity(free_models) or DEFAULT_FREE_OPENROUTER_MODELS


def filter_paid_openrouter_models(payload: dict[str, Any]) -> list[str]:
    paid_models = []
    for model in payload.get("data", []) or []:
        model_id = str(model.get("id", "")).strip()
        if model_id and not model_id.endswith(":free"):
            paid_models.append(model_id)
    return sort_models_by_capacity(paid_models) or DEFAULT_PAID_OPENROUTER_MODELS


def robust_model_subset(models: list[str], minimum_b: float = 70) -> list[str]:
    robust = [model for model in sort_models_by_capacity(models) if estimate_model_parameters_b(model) >= minimum_b]
    return robust or sort_models_by_capacity(models)


def sort_models_by_capacity(models: list[str]) -> list[str]:
    unique = list(dict.fromkeys(model for model in models if model))
    return sorted(unique, key=lambda model: (estimate_model_parameters_b(model), model), reverse=True)


def estimate_model_parameters_b(model_id: str) -> float:
    text = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b", text)
    if matches:
        return max(float(match) for match in matches)
    if "large" in text or "sonnet" in text or "opus" in text:
        return 70.0
    if "medium" in text:
        return 30.0
    if "small" in text or "mini" in text or "haiku" in text or "xs" in text:
        return 8.0
    return 0.0


def build_openrouter_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "fii-analytics/0.1"})
    return session


def _price_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_offer_for_llm(offer: dict[str, Any], manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest or {}
    cvm = manifest.get("cvm") or offer
    info = manifest.get("informacoes_gerais") or {}
    summaries = manifest.get("pdf_summaries") or []

    parts = [
        f"Numero do requerimento: {_clean(cvm.get('Numero_Requerimento') or offer.get('Numero_Requerimento'))}",
        f"Emissor/Fundo: {_clean(cvm.get('Nome_Emissor') or offer.get('Nome_Emissor'))}",
        f"Lider/coordenador: {_clean(cvm.get('Nome_Lider') or offer.get('Nome_Lider'))}",
        f"Valor mobiliario: {_clean(cvm.get('Valor_Mobiliario') or offer.get('Valor_Mobiliario'))}",
        f"Tipo de oferta: {_clean(cvm.get('Tipo_Oferta') or offer.get('Tipo_Oferta'))}",
        f"Status: {_clean(cvm.get('Status_Requerimento') or offer.get('Status_Requerimento') or info.get('status'))}",
        f"Publico alvo: {_clean(cvm.get('Publico_alvo') or offer.get('Publico_alvo'))}",
        f"Regime de distribuicao: {_clean(cvm.get('Regime_distribuicao') or offer.get('Regime_distribuicao'))}",
        f"Quantidade registrada: {_clean(cvm.get('Qtde_Total_Registrada') or offer.get('Qtde_Total_Registrada'))}",
        f"Valor total registrado: {_clean(cvm.get('Valor_Total_Registrado') or offer.get('Valor_Total_Registrado'))}",
        f"Bookbuilding: {_clean(cvm.get('Bookbuilding') or offer.get('Bookbuilding'))}",
        f"Oferta inicial: {_clean(cvm.get('Oferta_inicial') or offer.get('Oferta_inicial'))}",
        f"Emissao: {_clean(cvm.get('Emissao') or offer.get('Emissao'))}",
        f"Mercado de negociacao: {_clean(cvm.get('Mercado_negociacao') or offer.get('Mercado_negociacao'))}",
        f"Tipo de lastro: {_clean(cvm.get('Tipo_lastro') or offer.get('Tipo_lastro'))}",
        f"Descricao do lastro: {_clean(cvm.get('Descricao_lastro') or offer.get('Descricao_lastro'))}",
        f"Garantias: {_clean(cvm.get('Descricao_garantias') or offer.get('Descricao_garantias'))}",
        f"Agente fiduciario: {_clean(cvm.get('Agente_fiduciario') or offer.get('Agente_fiduciario'))}",
        f"Escriturador: {_clean(cvm.get('Escriturador') or offer.get('Escriturador'))}",
        f"Avaliador de risco: {_clean(cvm.get('Avaliador_Risco') or offer.get('Avaliador_Risco'))}",
        f"Destinacao dos recursos: {_clean(cvm.get('Destinacao_recursos') or offer.get('Destinacao_recursos'))}",
    ]

    participants = manifest.get("participants") or []
    if participants:
        readable = []
        for item in participants[:10]:
            readable.append(f"{_clean(item.get('razaoSocial'))} ({_clean(item.get('tipo'))})")
        parts.append("Participantes SRE: " + "; ".join(readable))

    snippets = []
    for summary in summaries[:3]:
        fields = summary.get("campos_extraidos") or {}
        for key in ["tipo_oferta", "valor_total", "preco_emissao", "taxa_distribuicao", "publico_alvo", "coordenador", "destinacao_recursos", "fatores_risco_trecho"]:
            value = fields.get(key)
            if value:
                snippets.append(f"{key}: {str(value)[:500]}")
    if snippets:
        parts.append("Trechos extraidos dos documentos: " + " | ".join(snippets[:8]))

    return "\n".join(parts)


def build_asset_argument_messages(asset_text: str, opponent_text: str) -> list[dict[str, str]]:
    system = (
        "Voce e uma analista de mercado de capitais em uma disputa de argumentos. "
        "Defenda o ativo designado como a alternativa mais interessante entre duas ofertas primarias do mesmo grupo. "
        "Use apenas os dados fornecidos, reconheca lacunas, avalie incentivos de quem vende/coordenada a oferta, "
        "custos, destino dos recursos, qualidade da informacao, risco de diluicao e contexto macro. "
        "Nao prometa retorno, nao faca recomendacao definitiva e nao invente dados."
    )
    user = f"""Ativo que voce deve defender:
{asset_text}

Ativo concorrente:
{opponent_text}

Escreva uma unica rodada de defesa em portugues. Estruture em 4 a 7 paragrafos curtos, com tom critico e profissional.
Explique por que esse ativo parece melhor ou mais investigavel que o concorrente, e quais riscos ainda precisam ser checados."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_messages(
    asset_1_name: str,
    asset_2_name: str,
    argument_1: str,
    argument_2: str,
) -> list[dict[str, str]]:
    system = (
        "Voce e um juiz imparcial de uma disputa entre duas LLMs que defenderam ofertas primarias do mesmo grupo. "
        "Julgue a qualidade dos argumentos, nao a popularidade do ativo. Seja critico, financeiro e cauteloso. "
        "Nao trate a decisao como recomendacao definitiva de investimento. "
        "Responda com um objeto JSON real, compacto, nao com uma string contendo JSON escapado."
    )
    user = f"""Compare os dois argumentos abaixo.

Ativo A: {asset_1_name}
Argumento da IA A:
{argument_1}

Ativo B: {asset_2_name}
Argumento da IA B:
{argument_2}

Responda somente JSON valido e curto no formato abaixo. Nao use markdown, nao use crases, nao escape aspas com barras invertidas. Limite cada lista a no maximo 2 itens e cada texto a uma frase:
{{
  "vencedor": "Ativo A" ou "Ativo B" ou "Empate",
  "ativo_vencedor": "nome do ativo ou Empate",
  "resumo": "uma frase objetiva com a decisao",
  "pontos_fortes_a": ["..."],
  "pontos_fortes_b": ["..."],
  "fragilidades_a": ["..."],
  "fragilidades_b": ["..."],
  "alerta": "limites da analise e dados que faltam"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_offer_debate_with_langchain(
    asset_a_text: str,
    asset_b_text: str,
    asset_a_name: str,
    asset_b_name: str,
    model_1: str,
    model_2: str,
    judge_model: str,
    api_key: str | None = None,
    model_pool: list[str] | None = None,
) -> DebateResult:
    from fii_analytics.analysis.chains import build_debate_chain, build_judge_chain, verdict_to_dict

    try:
        arguments = build_debate_chain(
            model_1=model_1,
            model_2=model_2,
            api_key=api_key,
            fallback_models=model_pool or [],
        ).invoke({"asset_a_text": asset_a_text, "asset_b_text": asset_b_text})
        verdict_obj = build_judge_chain(
            judge_model=judge_model,
            fallback_model=model_1,
            api_key=api_key,
        ).invoke(
            {
                "asset_1_name": asset_a_name,
                "asset_2_name": asset_b_name,
                "argumento_a": arguments["argumento_a"],
                "argumento_b": arguments["argumento_b"],
            }
        )
    except Exception as exc:
        _raise_openrouter_specific_error(exc)
        raise

    verdict = verdict_to_dict(verdict_obj)
    verdict.setdefault("vencedor", "Indefinido")
    verdict.setdefault("ativo_vencedor", verdict.get("vencedor") or "Indefinido")
    verdict.setdefault("resumo", "")
    verdict.setdefault("pontos_fortes_a", [])
    verdict.setdefault("pontos_fortes_b", [])
    verdict.setdefault("fragilidades_a", [])
    verdict.setdefault("fragilidades_b", [])
    return DebateResult(
        argument_1=DebateArgument(model=model_1, asset_name=asset_a_name, content=arguments["argumento_a"]),
        argument_2=DebateArgument(model=model_2, asset_name=asset_b_name, content=arguments["argumento_b"]),
        judge_model=judge_model,
        verdict=verdict,
    )


def _raise_openrouter_specific_error(exc: Exception) -> None:
    text = str(exc)
    lowered = text.lower()
    if "429" in text or "rate limit" in lowered:
        raise OpenRouterRateLimitError(
            "OpenRouter retornou 429: limite temporario de uso atingido para este modelo/chave. "
            "Aguarde alguns minutos ou escolha outro modelo gratuito."
        ) from exc
    if "402" in text or "payment required" in lowered or "credits" in lowered:
        raise OpenRouterPaymentRequiredError(
            "OpenRouter retornou 402: a chamada exigiu credito/billing na conta. "
            "Mesmo modelos gratuitos podem ser bloqueados por limite da conta, indisponibilidade do provedor gratuito ou politica do OpenRouter."
        ) from exc


def parse_judge_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if "```" in text:
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))

    candidates = [text]
    if '\\"' in text:
        candidates.append(text.replace('\\"', '"'))
    if text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                candidates.append(decoded)
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        candidate = _normalize_escaped_json_text(candidate)
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed

        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            parsed = _try_parse_json_object(match.group())
            if parsed is not None:
                return parsed

    partial = _parse_partial_judge_response(candidates)
    if partial is not None:
        return partial

    return {
        "vencedor": "Indefinido",
        "ativo_vencedor": "N/D",
        "resumo": text,
        "alerta": "O juiz nao retornou JSON estruturado.",
    }


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str):
        return _try_parse_json_object(parsed)
    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_escaped_json_text(text: str) -> str:
    normalized = text.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    return normalized.replace('\\"', '"').replace("\\n", "\n")


def _parse_partial_judge_response(candidates: list[str]) -> dict[str, Any] | None:
    text = _normalize_escaped_json_text("\n".join(candidates))
    if "vencedor" not in text and "ativo_vencedor" not in text:
        return None
    result: dict[str, Any] = {
        "vencedor": _extract_json_string_field(text, "vencedor") or "Indefinido",
        "ativo_vencedor": _extract_json_string_field(text, "ativo_vencedor") or "N/D",
        "resumo": _extract_json_string_field(text, "resumo") or text[:1000],
        "pontos_fortes_a": _extract_json_array_field(text, "pontos_fortes_a"),
        "pontos_fortes_b": _extract_json_array_field(text, "pontos_fortes_b"),
        "fragilidades_a": _extract_json_array_field(text, "fragilidades_a"),
        "fragilidades_b": _extract_json_array_field(text, "fragilidades_b"),
        "alerta": "O juiz retornou JSON incompleto/truncado; os campos acima foram recuperados parcialmente.",
    }
    return result


def _extract_json_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)', text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_json_array_field(text: str, field: str) -> list[str]:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    return [item.strip() for item in re.findall(r'"([^"]+)"', match.group(1))][:3]


def _clean(value: Any) -> str:
    if value is None:
        return "N/D"
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else "N/D"
