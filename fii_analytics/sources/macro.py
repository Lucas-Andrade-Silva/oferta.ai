from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from fii_analytics.config import settings
from fii_analytics.sources.http import build_session


logger = logging.getLogger(__name__)

BCB_SERIES = {
    "selic_meta": {"code": 432, "label": "Taxa Selic meta"},
    "cdi": {"code": 12, "label": "CDI diario"},
    "ipca": {"code": 433, "label": "IPCA mensal"},
    "igpm": {"code": 189, "label": "IGP-M mensal"},
}

MARKET_INDEXES = {
    "ibovespa": {"symbol": "^BVSP", "label": "Ibovespa"},
    "ifix": {"symbol": "IFIX.SA", "label": "IFIX"},
    "imob": {"symbol": "IMOB.SA", "label": "IMOB"},
}


@dataclass(frozen=True)
class MacroEvent:
    date: date
    title: str
    category: str
    description: str


class BCBClient:
    def __init__(self):
        self.session = build_session()

    def load_series(self, key: str, days: int = 730) -> pd.DataFrame:
        if key not in BCB_SERIES:
            raise KeyError(f"Unknown BCB series: {key}")
        code = BCB_SERIES[key]["code"]
        end = date.today()
        start = end - timedelta(days=days)
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        response = self.session.get(
            url,
            params={
                "formato": "json",
                "dataInicial": start.strftime("%d/%m/%Y"),
                "dataFinal": end.strftime("%d/%m/%Y"),
            },
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        if df.empty:
            return pd.DataFrame(columns=["data", "valor", "serie", "label"])
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["serie"] = key
        df["label"] = BCB_SERIES[key]["label"]
        return df

    def load_default_dashboard(self) -> pd.DataFrame:
        frames = []
        for key in BCB_SERIES:
            try:
                frames.append(self.load_series(key))
            except Exception:
                logger.exception("Failed to load BCB series: %s", key)
        try:
            frames.append(MarketIndexClient().load_default_indexes())
        except Exception:
            logger.exception("Failed to load market indexes")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class MarketIndexClient:
    def __init__(self):
        self.session = build_session()

    def load_index(self, key: str, days: int = 365) -> pd.DataFrame:
        if key not in MARKET_INDEXES:
            raise KeyError(f"Unknown market index: {key}")
        meta = MARKET_INDEXES[key]
        period2 = date.today()
        period1 = period2 - timedelta(days=days)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{meta['symbol']}"
        response = self.session.get(
            url,
            params={
                "period1": int(pd.Timestamp(period1).timestamp()),
                "period2": int(pd.Timestamp(period2 + timedelta(days=1)).timestamp()),
                "interval": "1d",
            },
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        result = (response.json().get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame(columns=["data", "valor", "serie", "label"])
        payload = result[0]
        timestamps = payload.get("timestamp") or []
        closes = ((payload.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        df = pd.DataFrame({"data": pd.to_datetime(timestamps, unit="s"), "valor": closes})
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["data", "valor"])
        df["serie"] = key
        df["label"] = meta["label"]
        return df

    def load_default_indexes(self) -> pd.DataFrame:
        frames = []
        for key in MARKET_INDEXES:
            try:
                frames.append(self.load_index(key))
            except Exception:
                logger.exception("Failed to load market index: %s", key)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def default_event_backlog() -> list[MacroEvent]:
    return [
        MacroEvent(date.today(), "Contexto macro pendente de enriquecimento", "sistema", "Estrutura pronta para comunicados do BCB, Copom, fiscal e eventos politicos."),
    ]
