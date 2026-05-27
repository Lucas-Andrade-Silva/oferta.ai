"""LangChain chains used by the AI layer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from fii_analytics.analysis.parsers import OutputFixingParser, fixing_parser, verdict_parser
from fii_analytics.analysis.prompts import chat_prompt, debate_prompt, judge_prompt, report_prompt
from fii_analytics.config import clean_secret, settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def current_groq_api_key(api_key: str | None = None) -> str | None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return clean_secret(api_key) or clean_secret(os.getenv("GROQ_API_KEY")) or clean_secret(settings.groq_api_key)


def _groq_llm(api_key: str | None = None, model: str | None = None, max_tokens: int = 1400) -> ChatGroq:
    return ChatGroq(
        api_key=current_groq_api_key(api_key) or "missing-groq-api-key",
        model=model or settings.groq_report_model or "llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=max_tokens,
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


def _openrouter_llm(model: str, api_key: str | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=clean_secret(api_key) or clean_secret(settings.openrouter_api_key) or "missing-openrouter-api-key",
        model=model,
        default_headers={
            "HTTP-Referer": "http://localhost:8502",
            "X-Title": "Oferta.Ai",
        },
    )
    return llm.bind(max_tokens=max_tokens) if max_tokens else llm


def _unique_fallback_models(primary: str, fallback_models: list[str] | None) -> list[str]:
    return [model for model in dict.fromkeys(fallback_models or []) if model and model != primary]


def build_report_chain(api_key: str | None = None, model: str | None = None):
    return report_prompt | _groq_llm(api_key=api_key, model=model) | StrOutputParser()


def build_chat_chain(api_key: str | None = None, model: str | None = None):
    return chat_prompt | _groq_llm(api_key=api_key, model=model, max_tokens=1000) | StrOutputParser()


def build_debate_chain(
    model_1: str | None = None,
    model_2: str | None = None,
    api_key: str | None = None,
    fallback_models: list[str] | None = None,
):
    first_model = model_1 or settings.debate_model_1
    second_model = model_2 or settings.debate_model_2
    llm_a = _openrouter_llm(first_model, api_key=api_key, max_tokens=1400)
    llm_b = _openrouter_llm(second_model, api_key=api_key, max_tokens=1400)
    fallback_a = [_openrouter_llm(model, api_key=api_key, max_tokens=1400) for model in _unique_fallback_models(first_model, [second_model, *(fallback_models or [])])]
    fallback_b = [_openrouter_llm(model, api_key=api_key, max_tokens=1400) for model in _unique_fallback_models(second_model, [first_model, *(fallback_models or [])])]

    return RunnableParallel(
        argumento_a=(
            RunnableLambda(lambda values: {"asset_text": values["asset_a_text"], "opponent_text": values["asset_b_text"]})
            | debate_prompt
            | llm_a.with_fallbacks(fallback_a)
            | StrOutputParser()
        ),
        argumento_b=(
            RunnableLambda(lambda values: {"asset_text": values["asset_b_text"], "opponent_text": values["asset_a_text"]})
            | debate_prompt
            | llm_b.with_fallbacks(fallback_b)
            | StrOutputParser()
        ),
    )


def build_judge_chain(
    judge_model: str | None = None,
    fallback_model: str | None = None,
    api_key: str | None = None,
):
    selected_judge_model = judge_model or settings.debate_judge_model
    selected_fallback = fallback_model or settings.debate_model_1
    judge_llm = _openrouter_llm(selected_judge_model, api_key=api_key, max_tokens=1800)
    fallback_llm = _openrouter_llm(selected_fallback, api_key=api_key, max_tokens=1800)
    parser = fixing_parser
    if selected_judge_model != settings.debate_judge_model or api_key:
        parser = OutputFixingParser.from_llm(parser=verdict_parser, llm=judge_llm)
    return (
        judge_prompt.partial(format_instructions=verdict_parser.get_format_instructions())
        | judge_llm.with_fallbacks([fallback_llm])
        | parser
    )


def verdict_to_dict(verdict: Any) -> dict[str, Any]:
    if hasattr(verdict, "model_dump"):
        return verdict.model_dump()
    if hasattr(verdict, "dict"):
        return verdict.dict()
    return dict(verdict)


groq_llm = _groq_llm()
model_a = _openrouter_llm(settings.debate_model_1, max_tokens=1400)
model_b = _openrouter_llm(settings.debate_model_2, max_tokens=1400)
judge_llm = _openrouter_llm(settings.debate_judge_model, max_tokens=1800)

report_chain = report_prompt | groq_llm | StrOutputParser()
chat_chain = chat_prompt | _groq_llm(max_tokens=1000) | StrOutputParser()
debate_chain = build_debate_chain(settings.debate_model_1, settings.debate_model_2)
judge_chain = (
    judge_prompt.partial(format_instructions=verdict_parser.get_format_instructions())
    | judge_llm.with_fallbacks([model_a])
    | fixing_parser
)
