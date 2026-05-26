from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

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

    def load_fii_table(self) -> pd.DataFrame:
        response = self.session.get(FII_RESULT_URL, headers=HEADERS, timeout=settings.request_timeout)
        response.raise_for_status()
        return parse_fii_table(response.text)


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

    for col in [
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
    ]:
        df[col] = df[col].map(_parse_br_number)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    return df


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

