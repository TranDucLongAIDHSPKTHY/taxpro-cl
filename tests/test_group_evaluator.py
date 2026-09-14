import unittest

from utility.utility_train.group_evaluator import (
    CSV_COLUMNS,
    evaluate_rankings,
    flatten_metrics,
    support_rows,
    validate_result_schema,
)


class GroupEvaluatorTests(unittest.TestCase):
    def test_overall_and_groups_share_ranking_and_report_24_metrics(self):
        rankings = {0: list(range(20)), 1: list(range(20, 40))}
        overall = {0: [1, 15], 1: [21]}
        groups = {
            "near_cold": {0: [1]},
            "long_tail": {0: [1, 15]},
            "warm": {1: [21]},
        }
        result = evaluate_rankings(rankings, overall, groups)
        flattened = flatten_metrics(result)
        self.assertEqual(len(flattened), 24)
        self.assertEqual(list(flattened), CSV_COLUMNS)
        self.assertEqual(result["overall"]["recall"]["20"], 1.0)
        self.assertEqual(result["near_cold"]["eligible_users"], 1)
        self.assertEqual(result["long_tail"]["positive_interactions"], 2)
        for group in ("near_cold", "long_tail", "warm"):
            for field in (
                "eligible_users",
                "relevant_items",
                "positive_interactions",
            ):
                self.assertIn(field, result[group])
                self.assertGreaterEqual(result[group][field], 0)
        self.assertEqual(
            {row["group"] for row in support_rows(result)},
            {"near_cold", "long_tail", "warm"},
        )

    def test_precision_recall_ndcg_are_finite_for_every_slice(self):
        rankings = {0: list(range(20))}
        targets = {0: [0]}
        result = evaluate_rankings(
            rankings,
            targets,
            {"near_cold": targets, "long_tail": targets, "warm": targets},
        )
        for group in ("overall", "near_cold", "long_tail", "warm"):
            for metric in ("precision", "recall", "ndcg"):
                for value in result[group][metric].values():
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_incomplete_metric_or_support_schema_is_rejected(self):
        rankings = {0: list(range(20))}
        targets = {0: [0]}
        result = evaluate_rankings(
            rankings,
            targets,
            {"near_cold": targets, "long_tail": targets, "warm": targets},
        )
        del result["near_cold"]["precision"]["10"]
        with self.assertRaises(ValueError):
            validate_result_schema(result)

        result = evaluate_rankings(
            rankings,
            targets,
            {"near_cold": targets, "long_tail": targets, "warm": targets},
        )
        del result["warm"]["eligible_users"]
        with self.assertRaises(ValueError):
            validate_result_schema(result)
