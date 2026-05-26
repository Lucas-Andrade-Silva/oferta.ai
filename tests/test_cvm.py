import unittest

import pandas as pd

from fii_analytics.sources.cvm import filter_fii_related, normalize_cvm_offers


class CVMNormalizationTest(unittest.TestCase):
    def test_normalize_and_filter_fii_related(self):
        df = pd.DataFrame(
            {
                "Valor_Mobiliario": ["Cotas de FII", "Debêntures"],
                "Data_requerimento": ["2026-01-10", "2026-01-11"],
                "Valor_Total_Registrado": ["1000", "2000"],
            }
        )
        normalized = normalize_cvm_offers(df)
        filtered = filter_fii_related(normalized)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["Valor_Total_Registrado"], 1000)
