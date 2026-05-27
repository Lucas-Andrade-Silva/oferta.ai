from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fii_analytics.config import settings
from fii_analytics.sources.http import build_session


logger = logging.getLogger(__name__)

FII_RESULT_URL = "https://www.fundamentus.com.br/fii_resultado.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}
COLUMNS = [
    "ticker",
    "segmento",
    "cotacao",
    "ffo_yield",
    "dividend_yield",
    "p_vp",
    "valor_mercado",
    "liquidez",
    "qtd_imoveis",
    "preco_m2",
    "aluguel_m2",
    "cap_rate",
    "vacancia_media",
    "endereco",
]

NUMERIC_COLUMNS = [
    "cotacao",
    "ffo_yield",
    "dividend_yield",
    "p_vp",
    "valor_mercado",
    "liquidez",
    "qtd_imoveis",
    "preco_m2",
    "aluguel_m2",
    "cap_rate",
    "vacancia_media",
]


@dataclass(frozen=True)
class MarketFii:
    ticker: str
    nome: str
    segmento: str | None
    cotacao: float | None
    dividend_yield: float | None
    p_vp: float | None
    valor_mercado: float | None
    liquidez: float | None


class FundamentusClient:
    def __init__(self):
        self.session = build_session()
        self.cache_path = Path(settings.cache_dir) / "fundamentus_fiis.csv"

    def load_fii_table(self) -> pd.DataFrame:
        try:
            response = self.session.get(FII_RESULT_URL, headers=HEADERS, timeout=settings.request_timeout)
            response.raise_for_status()
            df = parse_fii_table(response.text)
            if not df.empty:
                self._save_cache(df)
            return df
        except Exception:
            cached = self._load_cache()
            if not cached.empty:
                logger.warning("Fundamentus indisponivel; usando cache local em %s", self.cache_path, exc_info=True)
                return cached
            raise

    def _save_cache(self, df: pd.DataFrame) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.cache_path, index=False)
        except Exception:
            logger.warning("Nao foi possivel salvar cache do Fundamentus em %s", self.cache_path, exc_info=True)

    def _load_cache(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            return pd.DataFrame()
        try:
            return clean_fii_market_data(pd.read_csv(self.cache_path))
        except Exception:
            logger.warning("Nao foi possivel carregar cache do Fundamentus em %s", self.cache_path, exc_info=True)
            return pd.DataFrame()


def parse_fii_table(page_html: str) -> pd.DataFrame:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.I | re.S)
    records: list[dict[str, object]] = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)
        if len(cells) < len(COLUMNS):
            continue

        title_match = re.search(r'title="([^"]+)"', cells[0], flags=re.I)
        values = [_clean_cell(cell) for cell in cells[: len(COLUMNS)]]
        record = dict(zip(COLUMNS, values))
        record["nome"] = html.unescape(title_match.group(1)).strip() if title_match else values[0]
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].map(_parse_br_number)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    return clean_fii_market_data(df)


def clean_fii_market_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cleaned = df.copy()
    if "ticker" in cleaned.columns:
        cleaned["ticker"] = cleaned["ticker"].astype(str).str.upper().str.strip()

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_rules = {
        "cotacao": (cleaned.get("cotacao", pd.Series(dtype="float64")) <= 0)
        | (cleaned.get("cotacao", pd.Series(dtype="float64")) > 5000),
        "dividend_yield": (cleaned.get("dividend_yield", pd.Series(dtype="float64")) < 0)
        | (cleaned.get("dividend_yield", pd.Series(dtype="float64")) > 100),
        "p_vp": (cleaned.get("p_vp", pd.Series(dtype="float64")) <= 0)
        | (cleaned.get("p_vp", pd.Series(dtype="float64")) > 20),
        "valor_mercado": cleaned.get("valor_mercado", pd.Series(dtype="float64")) <= 0,
        "liquidez": cleaned.get("liquidez", pd.Series(dtype="float64")) < 0,
    }
    for column, mask in invalid_rules.items():
        if column in cleaned.columns:
            cleaned.loc[mask, column] = pd.NA

    return cleaned


def _clean_cell(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = html.unescape(text)
    return " ".join(text.split()).strip()


def _parse_br_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    text = text.replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None
