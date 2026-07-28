import json
import unittest

from typer.testing import CliRunner

from cradle_codec.cli import app


class IntegrationCliTests(unittest.TestCase):
    def test_lmcache_mp_cli_emits_multi_server_vllm_config(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "integration",
                "vllm-lmcache",
                "--lmcache-server-url",
                "http://127.0.0.1:9000",
                "--lmcache-server-url",
                "http://127.0.0.1:9001",
                "--lmcache-mq-timeout-s",
                "2.5",
                "--lmcache-heartbeat-interval-s",
                "0.5",
                "--lmcache-mp-transfer-mode",
                "lmcache_driven",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        extra = data["kv_connector_extra_config"]
        self.assertEqual(data["kv_connector"], "LMCacheMPConnector")
        self.assertEqual(extra["lmcache.mp.server_urls"], ["http://127.0.0.1:9000", "http://127.0.0.1:9001"])
        self.assertEqual(extra["lmcache.mp.mq_timeout"], 2.5)
        self.assertEqual(extra["lmcache.mp.heartbeat_interval"], 0.5)
        self.assertEqual(extra["lmcache.mp.mp_transfer_mode"], "lmcache_driven")
        self.assertNotIn("lmcache.mp.host", extra)


if __name__ == "__main__":
    unittest.main()
