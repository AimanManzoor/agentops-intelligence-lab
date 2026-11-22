import math
import unittest
from evals.evaluate_alert_policy import measure, choose, compare

POLICY = {"false_alert_cost": 1, "missed_failure_cost": 5}

class AlertPolicyTests(unittest.TestCase):
    def test_confusion_counts_and_cost(self):
        row = measure([0, 0, 1, 1], [.1, .8, .4, .9], .5, 1, 5)
        self.assertEqual([row[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(row["cost_units"], 6)
        self.assertEqual(row["alerts_per_100"], 50)
        self.assertEqual(row["precision"], .5)
        self.assertEqual(row["recall"], .5)

    def test_threshold_is_inclusive(self):
        self.assertEqual(measure([1], [.5], .5, 1, 5)["tp"], 1)

    def test_zero_alerts_and_no_failures(self):
        row = measure([0, 0], [.1, .2], .9, 1, 5)
        self.assertEqual((row["precision"], row["recall"], row["cost_units"]), (0, 0, 0))

    def test_bad_inputs_are_rejected(self):
        cases = [([], []), ([1], []), ([2], [.5]), ([1], [math.nan]),
                 ([1], [math.inf]), ([1], [-.1]), ([1], [1.1])]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                measure(labels, scores, .5, 1, 5)
        for costs in [(0, 1), (-1, 1), (1, math.inf)]:
            with self.assertRaises(ValueError):
                measure([1], [.5], .5, *costs)

    def test_policy_minimizes_validation_cost(self):
        winner, curve = choose([0, 1], [.4, .3], POLICY)
        self.assertEqual(len(curve), 101)
        self.assertEqual(winner["cost_units"], min(r["cost_units"] for r in curve))
        self.assertEqual(winner["threshold"], .3)

    def test_capacity_limit_and_infeasibility(self):
        policy = dict(POLICY, max_alert_fraction=.25)
        winner, _ = choose([0, 1, 0, 1], [.1, .3, .5, .9], policy)
        self.assertLessEqual(winner["alerts"], 1)
        with self.assertRaises(ValueError):
            choose([1], [1.0], dict(POLICY, max_alert_fraction=0))
        with self.assertRaises(ValueError):
            choose([1], [.5], dict(POLICY, max_alert_fraction=-1))

    def test_test_labels_cannot_change_selected_threshold(self):
        first = compare([0, 1], [.2, .8], [0, 0], [.1, .9], POLICY)
        second = compare([0, 1], [.2, .8], [1, 1], [.1, .9], POLICY)
        self.assertEqual(first["validation"], second["validation"])
        self.assertNotEqual(first["retrospective_test"]["cost_units"],
                            second["retrospective_test"]["cost_units"])

    def test_constant_policy_comparators(self):
        result = compare([0, 1], [.2, .8], [0, 1], [.1, .9], POLICY)
        self.assertEqual(result["always_alert_test"]["cost_units"], 1)
        self.assertEqual(result["never_alert_test"]["cost_units"], 5)
        self.assertEqual(result["never_alert_test"]["alerts"], 0)
        self.assertIsNone(result["never_alert_test"]["threshold"])

    def test_validation_budget_is_not_a_test_guarantee(self):
        result = compare([0, 1], [.1, .9], [1, 1], [.95, .95],
                         dict(POLICY, max_alert_fraction=.5))
        self.assertFalse(result["test_alert_limit_met"])
        self.assertEqual(result["baseline_0_26_test"]["threshold"], .26)

if __name__ == "__main__":
    unittest.main()
