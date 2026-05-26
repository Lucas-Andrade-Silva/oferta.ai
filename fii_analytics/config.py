from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    cvm_ofertas_url: str = "https://dados.cvm.gov.br/dados/OFERTA/DISTRIB/DADOS/oferta_distribuicao.zip"
    anbima_token_url: str = "https://api.anbima.com.br/oauth/access-token"
    anbima_base_url: str = "https://api.anbima.com.br"
    cvm_sre_base_url: str = os.getenv("CVM_SRE_BASE_URL", "https://web.cvm.gov.br/sre-publico-cvm")
    anbima_client_id: str | None = os.getenv("ANBIMA_CLIENT_ID")
    anbima_client_secret: str | None = os.getenv("ANBIMA_CLIENT_SECRET")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    debate_model_1: str = os.getenv("OPENROUTER_DEBATE_MODEL_1", "openai/gpt-4o-mini")
    debate_model_2: str = os.getenv("OPENROUTER_DEBATE_MODEL_2", "anthropic/claude-3.5-haiku")
    debate_judge_model: str = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_report_model: str = os.getenv("GROQ_REPORT_MODEL", "llama-3.3-70b-versatile")
    request_timeout: int = 30
    cache_dir: str = ".cache/fii_analytics"


settings = Settings()


def clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None
