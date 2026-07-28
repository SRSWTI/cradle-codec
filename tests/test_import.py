import unittest

import cradle_codec


class ImportTests(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertEqual(cradle_codec.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
