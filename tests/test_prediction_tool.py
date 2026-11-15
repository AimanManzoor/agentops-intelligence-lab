import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.agents.graph import build_graph
from src.tools import prediction_tool as prediction


class PredictionToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'trusted.joblib'
        self.path.touch()
        patcher = patch.object(prediction, 'MODEL_PATH', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.features = {'expected_workload': 5000, 'cpu_usage_pct': 50.0,
                         'memory_usage_pct': 60.0, 'recent_failure_rate': 0.1}
        self.model = SimpleNamespace(classes_=np.array([0, 1]),
                                     predict_proba=Mock(return_value=np.array([[0.7, 0.3]])))
        self.artifact = {'model': self.model, 'features': list(prediction.REQUIRED_FEATURES),
                         'threshold': 0.26, 'version': 'test-only'}

    def predict(self, artifact=None, **changes):
        with patch.object(prediction.joblib, 'load', return_value=self.artifact if artifact is None else artifact):
            return prediction.predict_failure(**{**self.features, **changes})

    def test_invalid_inputs_rejected_before_artifact_load(self):
        bad = [('expected_workload', True), ('expected_workload', 1.5),
               ('expected_workload', -1), ('expected_workload', 10**1000),
               ('cpu_usage_pct', '50'), ('cpu_usage_pct', -0.1), ('cpu_usage_pct', 100.1),
               ('memory_usage_pct', None), ('memory_usage_pct', float('nan')),
               ('recent_failure_rate', float('inf')), ('recent_failure_rate', -0.01),
               ('recent_failure_rate', 1.01)]
        self.path.unlink()
        for name, value in bad:
            with self.subTest(name=name, value=str(value)[:30]):
                with patch.object(prediction.joblib, 'load') as load:
                    with self.assertRaises(ValueError):
                        prediction.predict_failure(**{**self.features, name: value})
                load.assert_not_called()

    def test_valid_feature_boundaries_are_accepted(self):
        for amount in [0, 1]:
            for pct in [0, 100]:
                for rate in [0, 1]:
                    result = self.predict(expected_workload=amount, cpu_usage_pct=pct,
                                          memory_usage_pct=pct, recent_failure_rate=rate)
                    self.assertEqual(result['input_features']['expected_workload'], amount)

    def test_missing_artifact_does_not_load_or_create_file(self):
        self.path.unlink()
        with patch.object(prediction.joblib, 'load') as load:
            with self.assertRaises(FileNotFoundError):
                prediction.predict_failure(**self.features)
        load.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_artifact_metadata_schema_is_checked_before_inference(self):
        bad = [[], {}, {**self.artifact, 'features': list(reversed(prediction.REQUIRED_FEATURES))},
               {**self.artifact, 'features': ['target_failed']},
               {**self.artifact, 'version': ''}, {**self.artifact, 'version': None}]
        for artifact in bad:
            with self.subTest(artifact=artifact):
                with self.assertRaises(ValueError):
                    self.predict(artifact)
        self.model.predict_proba.assert_not_called()

    def test_invalid_thresholds_rejected_before_inference(self):
        for threshold in [True, np.bool_(False), '0.26', None, -0.01, 1.01,
                          float('nan'), float('inf'), 10**1000]:
            with self.subTest(threshold=str(threshold)[:30]):
                with self.assertRaises(ValueError):
                    self.predict({**self.artifact, 'threshold': threshold})
        self.model.predict_proba.assert_not_called()

    def test_class_labels_and_prediction_interface_required(self):
        for classes in [[0], [0, 2], [1, 1], [False, True], ['0', '1'], [[0, 1]]]:
            self.model.classes_ = np.asarray(classes)
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    self.predict()
        self.model.predict_proba.assert_not_called()
        with self.assertRaises(ValueError):
            self.predict({**self.artifact, 'model': object()})

    def test_probability_shape_finiteness_range_and_normalization(self):
        for values in [[], [0.7, 0.3], [[0.7]], [[0.7, 0.3], [0.7, 0.3]],
                       [[float('nan'), float('nan')]], [[-0.1, 1.1]],
                       [[0.4, 0.4]], [[float('inf'), 0]], [['0.7', '0.3']],
                       [[True, False]], [[0.5+0j, 0.5+0j]]]:
            self.model.predict_proba.return_value = np.asarray(values)
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.predict()

    def test_failure_class_is_selected_by_label_not_column_position(self):
        self.model.classes_ = np.array([1, 0])
        self.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        result = self.predict()
        self.assertEqual(result['failure_probability'], 0.8)
        self.assertTrue(result['predicted_failure'])

    def test_threshold_boundary_and_unrounded_probability_control_decision(self):
        for probability, expected in [(0.25996, False), (0.26, True), (0.26004, True)]:
            self.model.predict_proba.return_value = np.array([[1-probability, probability]])
            result = self.predict()
            self.assertEqual(result['predicted_failure'], expected)
            self.assertEqual(result['failure_probability'], round(probability, 4))
        frame = self.model.predict_proba.call_args.args[0]
        self.assertEqual(list(frame.columns), prediction.REQUIRED_FEATURES)
        self.assertEqual(frame.iloc[0].to_dict(), self.features)

    def test_corrupt_trusted_file_error_propagates_without_rewriting(self):
        original = b'not-a-valid-joblib-artifact'
        self.path.write_bytes(original)
        with self.assertRaises(Exception):
            prediction.predict_failure(**self.features)
        self.assertEqual(self.path.read_bytes(), original)

    def test_real_locally_trained_artifact_roundtrip(self):
        frame = pd.DataFrame([
            [100, 10, 10, 0], [200, 20, 20, 0.02], [300, 30, 30, 0.03],
            [7000, 70, 70, 0.2], [8000, 80, 80, 0.3], [9000, 90, 90, 0.4],
        ], columns=prediction.REQUIRED_FEATURES)
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
        model.fit(frame, [0, 0, 0, 1, 1, 1])
        artifact = {**self.artifact, 'model': model}
        joblib.dump(artifact, self.path)
        result = prediction.predict_failure(**self.features)
        probability = float(model.predict_proba(pd.DataFrame([self.features]))[0, 1])
        self.assertEqual(result['failure_probability'], round(probability, 4))
        self.assertEqual(result['predicted_failure'], probability >= 0.26)
        self.assertEqual(result['model_version'], 'test-only')

    def test_invalid_probability_stops_graph_before_evidence_or_routing(self):
        self.model.predict_proba.return_value = np.array([[float('nan'), float('nan')]])
        with (patch.object(prediction.joblib, 'load', return_value=self.artifact),
              patch('src.agents.graph.get_pipeline_evidence') as sql,
              patch('src.agents.graph.retrieve_similar_incidents') as retrieval):
            with self.assertRaises(ValueError):
                build_graph().invoke({'pipeline_name': 'payment_etl',
                                      'input_features': self.features})
        sql.assert_not_called()
        retrieval.assert_not_called()


if __name__ == '__main__':
    unittest.main()
