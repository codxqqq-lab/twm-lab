import copy
import json
import unittest
from pathlib import Path

from runtime.schema import observed_next, validate_scenario


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class PublicScenarioTest(unittest.TestCase):
    def test_examples_are_contiguous_public_observations(self):
        for path in EXAMPLES.glob("*.json"):
            with self.subTest(path=path):
                data = validate_scenario(json.loads(path.read_text(encoding="utf-8")))
                self.assertEqual(len(data["history"]), 8)
                self.assertEqual(len(data["recorded"]), 16)
                self.assertNotIn("node_factor", data)
                self.assertNotIn("edge_factor", data)

    def test_future_truth_only_exists_on_matching_action_and_state(self):
        data = validate_scenario(json.loads((EXAMPLES / "station_7.json").read_text(encoding="utf-8")))
        t = data["recorded"][0]
        self.assertIs(observed_next(data, 0, t["nodes_before"], t["edges_before"], t["action"]), t)
        self.assertIsNone(observed_next(data, 0, t["nodes_before"], t["edges_before"], (t["action"] + 1) % 7))
        self.assertIsNone(observed_next(data, 1, t["nodes_before"], t["edges_before"], t["action"]))

    def test_hidden_label_rejected(self):
        data = json.loads((EXAMPLES / "station_7.json").read_text(encoding="utf-8"))
        data["node_factor"] = 3
        with self.assertRaisesRegex(ValueError, "hidden"):
            validate_scenario(data)

    def test_broken_history_rejected(self):
        data = copy.deepcopy(json.loads((EXAMPLES / "station_7.json").read_text(encoding="utf-8")))
        data["history"][1]["nodes_before"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "does not follow"):
            validate_scenario(data)


if __name__ == "__main__":
    unittest.main()
