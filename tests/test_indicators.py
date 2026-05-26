import unittest

from fii_analytics.analysis.indicators import dividend_yield_annual, interpret_pvp, price_to_book


class IndicatorTest(unittest.TestCase):
    def test_price_to_book_and_interpretation(self):
        self.assertEqual(price_to_book(90, 100), 0.9)
        self.assertIn("desconto", interpret_pvp(0.95))

    def test_dividend_yield_annualized(self):
        self.assertEqual(dividend_yield_annual(1, 100), 12)
