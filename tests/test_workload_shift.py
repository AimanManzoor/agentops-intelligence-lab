import unittest

from evals.evaluate_workload_shift import (
    PROTOCOL,
    WORKLOAD_INCREASE,
    scenario_rows,
)
from src.ingestion.generate_realistic_incidents import generate_records


class WorkloadShiftTests(unittest.TestCase):
    def test_scenarios_are_reproducible_and_preserve_input(self):
        rows = generate_records(seed=2001, num_runs=30)
        before = list(rows)

        first = scenario_rows(rows, seed=2001, increase=WORKLOAD_INCREASE)
        second = scenario_rows(rows, seed=2001, increase=WORKLOAD_INCREASE)

        self.assertEqual(first, second)
        self.assertEqual(rows, before)

    def test_only_workload_changes_among_model_features(self):
        rows = generate_records(seed=2001, num_runs=30)
        baseline = scenario_rows(rows, seed=2001, increase=0)
        shifted = scenario_rows(
            rows, seed=2001, increase=WORKLOAD_INCREASE
        )

        for original, normal, changed in zip(rows, baseline, shifted):
            self.assertEqual(normal[3:7], original[3:7])
            self.assertEqual(changed[3], min(10000, original[3] + 3000))
            self.assertEqual(changed[4:7], original[4:7])
            self.assertGreaterEqual(changed[7], normal[7])
            self.assertGreaterEqual(changed[8], normal[8])
            self.assertIsNone(changed[9])

    def test_protocol_uses_new_seeds_and_fixed_capacity(self):
        self.assertTrue(set(PROTOCOL["batch_seeds"]).isdisjoint(
            {42, *range(1001, 1011)}
        ))
        self.assertEqual(PROTOCOL["review_budget"], 50)
        self.assertEqual(PROTOCOL["batch_size"], 250)

    def test_unknown_shift_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_rows([], seed=2001, increase=-100)


if __name__ == "__main__":
    unittest.main()
