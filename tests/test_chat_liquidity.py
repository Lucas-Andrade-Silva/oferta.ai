import pandas as pd

from fii_analytics.analysis.chat_logic import deterministic_liquidity_answer, normalize_chat_text, resolve_focus_ticker


def sample_market_fiis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "BTLG11",
                "nome": "BTG PACTUAL LOGISTICA FUNDO DE INVESTIMENTO IMOBILIARIO",
                "segmento": "Logistica",
                "liquidez": 11887900,
            },
            {
                "ticker": "BTHF11",
                "nome": "BTG PACTUAL REAL ESTATE HEDGE FUND FII",
                "segmento": "Outros",
                "liquidez": 3097460,
            },
            {
                "ticker": "HPDP11",
                "nome": "HEDGE SHOPPING PARQUE DOM PEDRO FUNDO DE INVESTIMENTO IMOBILIARIO",
                "segmento": "Shoppings",
                "liquidez": 10544000,
            },
        ]
    )


def test_deterministic_liquidity_ranks_btg_fiis():
    answer = deterministic_liquidity_answer(
        "Quais sao os FIIs com maior liquidez do BTG?",
        [],
        sample_market_fiis(),
    )

    assert answer is not None
    assert answer.index("BTLG11") < answer.index("BTHF11")
    assert "11.887.900" in answer


def test_deterministic_liquidity_blocks_wrong_supera_inference():
    messages = [
        {
            "role": "assistant",
            "content": "Os FIIs do BTG sao BTLG11, com liquidez de 11.887.900, e BTHF11, com liquidez de 3.097.460. O HPDP11 tem liquidez de 10.544.000.",
        }
    ]
    answer = deterministic_liquidity_answer(
        "HPDP11 tem mais liquidez que estes?",
        messages,
        sample_market_fiis(),
    )

    assert answer is not None
    assert answer.startswith("Nao. HPDP11 tem 10.544.000, abaixo de BTLG11")
    ranking = answer.split("Pelos dados de mercado carregados", 1)[1]
    assert ranking.index("BTLG11") < ranking.index("HPDP11") < ranking.index("BTHF11")


def test_resolve_focus_ticker_uses_recent_history():
    messages = [{"role": "assistant", "content": "KNCR11 aparece com cotacao de 105,45."}]

    assert resolve_focus_ticker("e em qual banco esta sendo ofertado?", messages) == "KNCR11"


def test_normalize_chat_text_preserves_cedilla_letters():
    assert "PRECO" in normalize_chat_text("preço da cotação")
    assert "COTACAO" in normalize_chat_text("preço da cotação")
