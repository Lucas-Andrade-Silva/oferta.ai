import unittest

from fii_analytics.config import Settings
from fii_analytics.sources.anbima import AnbimaClient


class AnbimaClientTest(unittest.TestCase):
    def test_anbima_requires_credentials(self):
        client = AnbimaClient(Settings(anbima_client_id=None, anbima_client_secret=None))
        with self.assertRaises(ValueError):
            client.get_token()

    def test_anbima_records_from_payload_tags_product(self):
        payload = {"content": [{"codigo_ativo": "ABC123", "taxa_indicativa": 12.34}]}
        records = AnbimaClient._records_from_payload(payload, "CRI/CRA")
        self.assertEqual(records[0]["codigo_ativo"], "ABC123")
        self.assertEqual(records[0]["taxa_indicativa"], 12.34)
        self.assertEqual(records[0]["produto_anbima"], "CRI/CRA")
