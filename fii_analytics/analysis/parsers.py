"""Structured parsers for LangChain LLM outputs."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from fii_analytics.config import clean_secret, settings


try:
    from langchain.output_parsers import OutputFixingParser
except ModuleNotFoundError:
    class OutputFixingParser(BaseOutputParser[Any]):
        parser: Any
        llm: Any

        @classmethod
        def from_llm(cls, parser: Any, llm: Any):
            return cls(parser=parser, llm=llm)

        def parse(self, text: str) -> Any:
            try:
                return self.parser.parse(text)
            except Exception:
                fixed = self.llm.invoke(
                    [
                        (
                            "system",
                            "Corrija a resposta para obedecer exatamente ao formato estruturado solicitado. Retorne apenas JSON valido.",
                        ),
                        (
                            "human",
                            f"Instrucoes de formato:\n{self.parser.get_format_instructions()}\n\nResposta original:\n{text}",
                        ),
                    ]
                )
                return self.parser.parse(str(getattr(fixed, "content", fixed)))

        @property
        def _type(self) -> str:
            return "output_fixing_parser_compat"


class VeredictJuiz(BaseModel):
    vencedor: str = Field(default="Indefinido")
    ativo_vencedor: str = Field(default="N/D")
    resumo: str = Field(default="")
    criterios: list[dict[str, Any]] = Field(default_factory=list)
    pontos_fortes_a: list[str] = Field(default_factory=list)
    pontos_fortes_b: list[str] = Field(default_factory=list)
    fragilidades_a: list[str] = Field(default_factory=list)
    fragilidades_b: list[str] = Field(default_factory=list)
    alerta: str = Field(default="")


def _openrouter_judge_llm() -> ChatOpenAI:
    api_key = clean_secret(settings.openrouter_api_key) or "missing-openrouter-api-key"
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=settings.debate_judge_model,
    )


verdict_parser = PydanticOutputParser(pydantic_object=VeredictJuiz)
fixing_parser = OutputFixingParser.from_llm(parser=verdict_parser, llm=_openrouter_judge_llm())
