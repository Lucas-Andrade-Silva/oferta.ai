import pandas as pd

from fii_analytics.sources.fundamentus import FundamentusClient, clean_fii_market_data


class FailingSession:
    def get(self, *args, **kwargs):
        raise RuntimeError("dns unavailable")


def test_fundamentus_uses_cache_when_network_fails(tmp_path):
    cache_path = tmp_path / "fundamentus_fiis.csv"
    pd.DataFrame(
        [
            {
                "ticker": "BTLG11",
                "nome": "BTG PACTUAL LOGISTICA FUNDO DE INVESTIMENTO IMOBILIARIO",
                "segmento": "Logistica",
                "liquidez": 11887900,
            }
        ]
    ).to_csv(cache_path, index=False)

    client = FundamentusClient()
    client.session = FailingSession()
    client.cache_path = cache_path

    df = client.load_fii_table()

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "BTLG11"
    assert df.iloc[0]["liquidez"] == 11887900


def test_fundamentus_cleaner_removes_invalid_market_values():
    df = pd.DataFrame(
        [
            {"ticker": "btlg11", "cotacao": 101.4, "p_vp": 0.97, "valor_mercado": 1000, "liquidez": 50},
            {"ticker": "tour11", "cotacao": 8.8, "p_vp": 0, "valor_mercado": 0, "liquidez": 0},
            {"ticker": "loft11b", "cotacao": 29.9, "p_vp": 59800, "valor_mercado": 1000, "liquidez": 0},
            {"ticker": "rbrm11", "cotacao": 160000, "p_vp": 8.16, "valor_mercado": 1000, "liquidez": 0},
        ]
    )

    cleaned = clean_fii_market_data(df)

    assert cleaned.iloc[0]["ticker"] == "BTLG11"
    assert cleaned.iloc[0]["p_vp"] == 0.97
    assert pd.isna(cleaned.iloc[1]["p_vp"])
    assert pd.isna(cleaned.iloc[1]["valor_mercado"])
    assert pd.isna(cleaned.iloc[2]["p_vp"])
    assert pd.isna(cleaned.iloc[3]["cotacao"])
