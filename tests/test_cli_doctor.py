import json
import unittest

from typer.testing import CliRunner

from cradle_codec.cli import app
from cradle_codec.cli.doctor import build_doctor_report


class DoctorCliTests(unittest.TestCase):
    def test_build_doctor_report_is_json_safe(self) -> None:
        report = build_doctor_report(timeout_s=0.1)
        encoded = json.dumps(report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["package"], "cradle-codec")
        self.assertTrue(decoded["modules"]["cradle_codec"]["available"])
        self.assertIn("nvidia-smi", decoded["commands"])
        self.assertIn("gst-launch-1.0", decoded["commands"])

    def test_doctor_cli_emits_probe_report(self) -> None:
        result = CliRunner().invoke(app, ["doctor", "--compact", "--timeout-s", "0.1"])
        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["package"], "cradle-codec")
        self.assertIn("serving_optional_dependencies_ready", payload)
        self.assertIn("gstreamer_packet_transport_ready", payload)
        self.assertTrue(payload["core_artifact_tools_ready"])


if __name__ == "__main__":
    unittest.main()
