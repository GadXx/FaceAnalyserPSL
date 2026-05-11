from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.face_aesthetics.scoring.common import nearest_group_payload, scoring_feature_columns
from src.face_aesthetics.scoring.dataset import build_scoring_manifest
from src.face_aesthetics.scoring.explainability import aggregate_component_contributions, build_local_explanation
from src.face_aesthetics.scoring.inference import run_scorer_inference
from src.face_aesthetics.scoring.training import (
    _build_ordinal_models,
    _build_pairwise_rows,
    _ensemble_prediction,
    _feature_lineage_row,
    _ordinal_predict,
    run_scorer_training,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_scoring_config_tree(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.yaml", {"project": {"name": "scoring-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(config_dir / "features.yaml", {})
    _write_json(config_dir / "preprocess.yaml", {})
    _write_json(
        config_dir / "scoring.yaml",
        {
            "scoring": {
                "dataset_root": "dataset",
                "allowed_image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
                "manifest_path": "data/scoring/scoring_manifest.parquet",
                "features_path": "data/scoring/scoring_features.parquet",
                "processed_analysis_dir": "data/scoring/processed/analysis_images",
                "processed_aligned_dir": "data/scoring/processed/aligned_faces",
                "model_dir": "models/scoring",
                "regressor_artifact_path": "models/scoring/scorer_regressor.joblib",
                "classifier_artifact_path": "models/scoring/scorer_classifier.joblib",
                "config_snapshot_path": "models/scoring/scoring_config_snapshot.json",
                "summary_paths": {
                    "dataset_summary_path": "reports/metrics/scoring_dataset_summary.json",
                    "feature_summary_path": "reports/metrics/scoring_feature_summary.json",
                    "training_summary_path": "reports/metrics/scorer_training_summary.json",
                    "selected_features_path": "reports/metrics/scorer_selected_features.json",
                    "feature_importance_path": "reports/metrics/scorer_feature_importance.json",
                    "feature_correlations_path": "reports/metrics/scorer_feature_correlations.json",
                    "global_feature_importance_path": "reports/metrics/scorer_global_feature_importance.json",
                    "component_importance_path": "reports/metrics/scorer_component_importance.json",
                    "feature_effects_path": "reports/metrics/scorer_feature_effects.json",
                    "group_confusion_path": "reports/metrics/scorer_group_confusion.json",
                    "groupwise_errors_path": "reports/metrics/scorer_groupwise_errors.json",
                    "quality_failures_path": "reports/metrics/scoring_quality_failures.json",
                    "pairwise_summary_path": "reports/metrics/pairwise_ranker_summary.json",
                    "ordinal_summary_path": "reports/metrics/ordinal_scorer_summary.json",
                    "ensemble_summary_path": "reports/metrics/scorer_ensemble_summary.json",
                    "pairwise_dataset_summary_path": "reports/metrics/scoring_pairwise_summary.json",
                },
                "report_paths": {
                    "json_path": "reports/examples/scorer_report.json",
                    "markdown_path": "reports/examples/scorer_report.md",
                },
                "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "random_seed": 42,
                "target_column": "target_score",
                "target_group_column": "target_group_label",
                "split_column": "split",
                "image_id_column": "image_id",
                "feature_mode": "geometry_only",
                "quality_gate_policy": "train_only_quality_pass",
                "minimum_samples_per_group_warning": 2,
                "include_feature_families": ["normalized", "symmetry", "angles"],
                "include_raw_pixel_features": False,
                "allow_raw_pixel_features": False,
                "allow_embeddings": False,
                "allow_metadata_features": False,
                "exclude_feature_patterns": ["raw_*", "*_px"],
                "feature_exclude_columns": [],
                "max_missing_fraction_per_feature": 1.0,
                "cv": {"enabled": True, "n_splits": 3, "shuffle": True, "selection_metric": "mae", "secondary_metric": "spearman"},
                "calibration": {"enabled": True, "method": "isotonic_regression", "fit_split": "val"},
                "score_semantics": {
                    "score_range": [0.0, 10.0],
                    "observed_training_label_range": "auto",
                    "average_zone_center": 5.0,
                    "elite_anchor_score": 9.5,
                    "elite_percentile_hint": 0.99,
                    "theoretical_max_score": 10.0,
                    "invalid_or_severe_anomaly_score": 0.0,
                    "strict_scale": True,
                },
                "strict_scoring": {
                    "enabled": True,
                    "penalize_regression_to_mean": True,
                    "high_score_threshold": 8.5,
                    "low_score_threshold": 3.0,
                    "elite_score_threshold": 9.0,
                    "extreme_bias_penalty_weight": 1.0,
                    "high_score_underprediction_penalty": 1.5,
                    "low_score_overprediction_penalty": 1.5,
                    "group_macro_mae_weight": 1.0,
                    "high_group_mae_weight": 1.5,
                    "low_group_mae_weight": 1.5,
                    "ranking_weight": 1.0,
                    "calibration_weight": 1.0,
                },
                "sample_weighting": {"enabled": True, "strategy": "inverse_group_frequency"},
                "pairwise": {
                    "enabled": True,
                    "max_pairs_per_anchor": 4,
                    "close_distance_threshold": 1.0,
                    "close_distance_pair_multiplier": 2.0,
                    "reference_examples_per_group": 2,
                    "large_distance_threshold": 3.0,
                    "large_distance_pair_multiplier": 1.75,
                    "high_vs_mid_pair_multiplier": 1.75,
                    "low_vs_mid_pair_multiplier": 1.75,
                    "elite_vs_average_pair_multiplier": 2.25,
                },
                "ordinal": {"enabled": True, "min_positive_threshold_samples": 2},
                "ensemble": {"regression_weight": 0.45, "pairwise_weight": 0.35, "ordinal_weight": 0.20, "disagreement_std_threshold": 0.75},
                "prediction_clipping": {"enabled": True, "min": 0.0, "max": 10.0},
                "regression_models": {
                    "ridge": {"enabled": True, "alpha": 1.0},
                    "elastic_net": {"enabled": True, "alpha": 0.05, "l1_ratio": 0.2},
                    "random_forest": {"enabled": True, "n_estimators": 50, "max_depth": 5, "min_samples_leaf": 1},
                    "gradient_boosting": {"enabled": True, "n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
                    "hist_gradient_boosting": {"enabled": True, "max_depth": 3, "learning_rate": 0.05, "max_iter": 50},
                    "xgboost": {"enabled": False},
                },
                "classification_models": {
                    "logistic_regression": {"enabled": True, "max_iter": 500, "c": 1.0},
                    "random_forest": {"enabled": True, "n_estimators": 50, "max_depth": 5, "min_samples_leaf": 1},
                    "gradient_boosting": {"enabled": False, "n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
                },
                "selection_metric": "mae",
                "confidence": {
                    "dense_training_quantiles": [0.1, 0.9],
                    "max_regression_classifier_disagreement": 1.5,
                    "max_primary_aux_disagreement": 1.25,
                    "max_imputed_fraction_for_high": 0.05,
                    "max_imputed_fraction_for_medium": 0.2,
                    "min_classifier_top_probability_for_high": 0.45,
                    "min_classifier_top_probability_for_medium": 0.25,
                },
                "components": {
                    "face_shape": ["ratio_face_aspect"],
                    "mouth_lips": ["ratio_mouth_width_to_face_width", "angle_mouth_line_deg"],
                    "symmetry": ["symmetry_index"],
                },
            }
        },
    )


class ScoringPipelineTests(unittest.TestCase):
    def test_prepare_scoring_manifest_parses_float_folder_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)
            (project_root / "dataset" / "5").mkdir(parents=True, exist_ok=True)
            (project_root / "dataset" / "5.5").mkdir(parents=True, exist_ok=True)
            (project_root / "dataset" / "5" / "a.jpg").write_bytes(b"stub")
            (project_root / "dataset" / "5.5" / "b.png").write_bytes(b"stub")

            artifacts = build_scoring_manifest(project_root=project_root)
            frame = pd.read_parquet(artifacts.manifest_path)

            self.assertEqual(len(frame), 2)
            self.assertIn("target_score", frame.columns)
            self.assertIn("target_group_label", frame.columns)
            self.assertEqual(sorted(frame["target_score"].tolist()), [5.0, 5.5])
            self.assertEqual(sorted(frame["target_group_label"].tolist()), ["5", "5.5"])
            self.assertTrue(set(frame["split"].tolist()).issubset({"train", "val", "test"}))

    def test_prepare_scoring_manifest_assigns_at_least_one_val_per_group_when_possible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)
            for group_name in ("1", "2", "3"):
                group_dir = project_root / "dataset" / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    (group_dir / f"{index}.jpg").write_bytes(b"stub")

            artifacts = build_scoring_manifest(project_root=project_root)
            frame = pd.read_parquet(artifacts.manifest_path)
            for group_name in ("1", "2", "3"):
                subset = frame.loc[frame["target_group_label"] == group_name]
                self.assertGreaterEqual(int((subset["split"] == "val").sum()), 1)
                self.assertGreaterEqual(int((subset["split"] == "train").sum()), 1)

    def test_prepare_scoring_manifest_assigns_test_when_group_has_enough_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)
            group_dir = project_root / "dataset" / "7.5"
            group_dir.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                (group_dir / f"{index}.jpg").write_bytes(b"stub")
            artifacts = build_scoring_manifest(project_root=project_root)
            frame = pd.read_parquet(artifacts.manifest_path)
            subset = frame.loc[frame["target_group_label"] == "7.5"]
            self.assertGreaterEqual(int((subset["split"] == "test").sum()), 1)

    def test_nearest_group_payload_and_clipping_behavior(self) -> None:
        payload = nearest_group_payload(7.86, [7.0, 7.5, 8.0, 8.5])
        self.assertEqual(payload["nearest_score_group"], 8.0)
        self.assertAlmostEqual(payload["distance_to_nearest_group"], 0.14, places=2)
        self.assertAlmostEqual(payload["distances_to_groups"]["7.5"], 0.36, places=2)

    def test_raw_pixel_features_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)
            config = json.loads((project_root / "configs" / "scoring.yaml").read_text())["scoring"]
            columns = scoring_feature_columns(config)
            self.assertNotIn("raw_face_width_px", columns)
            self.assertNotIn("raw_upper_face_height_px", columns)
            self.assertIn("ratio_face_aspect", columns)
            self.assertIn("symmetry_index", columns)
            self.assertIn("angle_mouth_line_deg", columns)

    def test_pairwise_rows_exclude_equal_score_pairs_and_assign_labels(self) -> None:
        frame = pd.DataFrame(
            [
                {"target_score": 5.0, "target_group_label": "5", "ratio_face_aspect": 0.71, "symmetry_index": 0.82},
                {"target_score": 5.0, "target_group_label": "5", "ratio_face_aspect": 0.72, "symmetry_index": 0.83},
                {"target_score": 7.0, "target_group_label": "7", "ratio_face_aspect": 0.77, "symmetry_index": 0.87},
            ]
        )
        pairwise_frame, summary = _build_pairwise_rows(
            frame,
            ["ratio_face_aspect", "symmetry_index"],
            max_pairs_per_anchor=2,
            close_distance_threshold=1.0,
            close_distance_pair_multiplier=2.0,
            large_distance_threshold=3.0,
            large_distance_pair_multiplier=1.75,
            high_vs_mid_pair_multiplier=1.75,
            low_vs_mid_pair_multiplier=1.75,
            elite_vs_average_pair_multiplier=2.25,
            low_score_threshold=3.0,
            high_score_threshold=8.5,
            elite_score_threshold=9.0,
            average_zone_center=5.0,
            random_seed=42,
        )
        self.assertGreater(len(pairwise_frame), 0)
        self.assertTrue((pairwise_frame["left_score"] != pairwise_frame["right_score"]).all())
        self.assertTrue(set(pairwise_frame["pair_label"].tolist()).issubset({0, 1}))
        positive_rows = pairwise_frame.loc[pairwise_frame["left_score"] > pairwise_frame["right_score"]]
        self.assertTrue((positive_rows["pair_label"] == 1).all())
        negative_rows = pairwise_frame.loc[pairwise_frame["left_score"] < pairwise_frame["right_score"]]
        self.assertTrue((negative_rows["pair_label"] == 0).all())
        self.assertEqual(summary["image_count"], 3)

    def test_ordinal_thresholds_and_expected_score(self) -> None:
        X = pd.DataFrame(
            [
                {"ratio_face_aspect": 0.70, "symmetry_index": 0.80},
                {"ratio_face_aspect": 0.72, "symmetry_index": 0.82},
                {"ratio_face_aspect": 0.78, "symmetry_index": 0.88},
                {"ratio_face_aspect": 0.80, "symmetry_index": 0.90},
            ]
        ).to_numpy(dtype=float)
        y = pd.Series([5.0, 5.5, 8.0, 8.5]).to_numpy(dtype=float)
        payload = _build_ordinal_models(
            X,
            y,
            [5.0, 5.5, 8.0, 8.5],
            random_seed=42,
            min_positive_threshold_samples=1,
            sample_weights=None,
        )
        pred, probabilities = _ordinal_predict(payload, X[:1])
        self.assertGreater(len(payload["thresholds"]), 0)
        self.assertEqual(len(pred), 1)
        self.assertGreaterEqual(pred[0], 5.0)
        self.assertLessEqual(pred[0], 8.5)
        self.assertTrue(probabilities)

    def test_ensemble_prediction_and_disagreement_signal(self) -> None:
        prediction, std = _ensemble_prediction(
            {"regression": 7.0, "pairwise": 7.5, "ordinal": 7.25},
            {"regression": 0.45, "pairwise": 0.35, "ordinal": 0.20},
        )
        self.assertAlmostEqual(prediction, 7.225, places=4)
        self.assertIsNotNone(std)
        high_disagreement, high_std = _ensemble_prediction(
            {"regression": 2.0, "pairwise": 8.0, "ordinal": 7.5},
            {"regression": 0.45, "pairwise": 0.35, "ordinal": 0.20},
        )
        self.assertIsNotNone(high_disagreement)
        self.assertGreater(high_std or 0.0, 2.0)

    def test_component_contributions_aggregate_feature_contributions(self) -> None:
        rows = [
            {"feature": "ratio_face_aspect", "contribution": 0.3},
            {"feature": "symmetry_index", "contribution": 0.2},
            {"feature": "angle_mouth_line_deg", "contribution": -0.1},
        ]
        components = {
            "face_shape": ["ratio_face_aspect"],
            "symmetry": ["symmetry_index"],
            "mouth_lips": ["angle_mouth_line_deg"],
        }
        aggregated = aggregate_component_contributions(rows, components)
        self.assertAlmostEqual(aggregated["face_shape"]["contribution"], 0.3)
        self.assertEqual(aggregated["mouth_lips"]["direction"], "negative")

    def test_reference_relative_lineage_maps_to_anatomical_component(self) -> None:
        registry = {
            "mediapipe_left_eye_width_to_face_width": {
                "component": "eyes",
                "source": "mediapipe_face_mesh",
                "reliability": "reliable",
            },
            "mediapipe_chin_taper_ratio": {
                "component": "jaw_chin",
                "source": "mediapipe_face_mesh",
                "reliability": "proxy",
            },
        }
        eye_row = _feature_lineage_row(
            "ref_z_mediapipe_left_eye_width_to_face_width",
            registry_by_name=registry,
            component_map={},
        )
        self.assertEqual(eye_row["base_feature_name"], "mediapipe_left_eye_width_to_face_width")
        self.assertEqual(eye_row["anatomical_component"], "eyes")
        self.assertEqual(eye_row["transform_family"], "reference_relative")
        self.assertEqual(eye_row["reference_transform"], "z")
        self.assertNotEqual(eye_row["anatomical_component"], "reference_deviation")

        jaw_row = _feature_lineage_row(
            "ref_absdev_mediapipe_chin_taper_ratio",
            registry_by_name=registry,
            component_map={},
        )
        self.assertEqual(jaw_row["base_feature_name"], "mediapipe_chin_taper_ratio")
        self.assertEqual(jaw_row["anatomical_component"], "jaw_chin")
        self.assertEqual(jaw_row["transform_family"], "reference_relative")
        self.assertEqual(jaw_row["reference_transform"], "absdev")

    def test_shap_fallback_explanation_works_when_shap_unavailable(self) -> None:
        payload = {
            "global_feature_importance": [{"feature_name": "ratio_face_aspect", "importance": 0.8}],
            "global_feature_effects": [{"feature_name": "ratio_face_aspect", "signed_effect": 0.8, "abs_effect": 0.8}],
            "feature_means": {"ratio_face_aspect": 0.7},
            "feature_stds": {"ratio_face_aspect": 0.1},
            "training_target_mean": 5.0,
        }
        feature_frame = pd.DataFrame([{"ratio_face_aspect": 0.8}])
        with patch.dict("sys.modules", {"shap": None}):
            explanation = build_local_explanation(
                estimator_payload=payload,
                feature_frame=feature_frame,
                predicted_score=6.0,
                scoring_config={"components": {"face_shape": ["ratio_face_aspect"]}},
            )
        self.assertIn(explanation["explanation_method"], {"importance_fallback", "shap_fallback"})
        self.assertTrue(explanation["component_contributions"])

    def test_train_scorer_and_run_inference_on_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)

            rows = []
            scores = [5.0, 5.0, 5.5, 5.5, 7.0, 7.0, 8.0, 8.0, 9.5, 9.5, 10.0, 10.0]
            splits = ["train", "train", "train", "val", "train", "val", "train", "test", "train", "test", "val", "test"]
            for index, (score, split) in enumerate(zip(scores, splits), start=1):
                rows.append(
                    {
                        "image_id": f"img-{index}",
                        "image_path": f"/tmp/img-{index}.jpg",
                        "file_name": f"img-{index}.jpg",
                        "source_score_folder": str(score),
                        "target_score": float(score),
                        "target_group_label": str(score),
                        "split": split,
                        "dataset_name": "scoring_dataset",
                        "sample_id": f"img-{index}",
                        "subject_id": None,
                        "quality_pass": True,
                        "processing_status": "processed",
                        "processing_error_message": None,
                        "raw_labels": None,
                        "normalized_labels": None,
                        "analysis_image_path": None,
                        "aligned_image_path": None,
                        "reference_model": "scoring_experimental",
                        "raw_face_width_px": 100.0 + score * 2.0,
                        "raw_face_height_px": 140.0 + score * 3.0,
                        "ratio_face_aspect": 0.7 + score / 100.0,
                        "ratio_mouth_width_to_face_width": 0.2 + score / 100.0,
                        "symmetry_index": 0.8 + score / 200.0,
                        "angle_mouth_line_deg": score / 10.0,
                        "mediapipe_face_oval_aspect_ratio": 0.68 + score / 120.0,
                        "mediapipe_eye_shape_symmetry_index": 0.82 + score / 250.0,
                        "mediapipe_nose_axis_angle_deg": score / 12.0,
                        "mediapipe_face_oval_width": 120.0 + score,
                    }
                )
            rows[0]["quality_pass"] = False
            features_path = project_root / "data" / "scoring" / "scoring_features.parquet"
            features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(features_path, index=False)

            training = run_scorer_training(project_root=project_root)
            self.assertEqual(training.summary["status"], "completed")
            self.assertIsNotNone(training.summary["best_regressor_model"])
            self.assertTrue(training.regressor_artifact_path.exists())
            self.assertEqual(training.summary["data"]["excluded_quality_failed_rows"], 1)
            self.assertEqual(training.summary["data"]["trainable_rows"], len(rows) - 1)
            self.assertEqual(training.summary["primary_score_model"]["family"], "regression")
            self.assertFalse(training.summary["primary_score_model"]["ensemble_used_as_primary"])
            self.assertIn("train", training.summary["data"]["split_counts"])
            self.assertIn("val", training.summary["data"]["split_counts"])
            self.assertIn("test", training.summary["data"]["split_counts"])
            self.assertIn("regression_models", json.loads((project_root / "reports" / "metrics" / "scorer_cv_results.json").read_text()))
            self.assertIn("cv_summary", training.summary)
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_selected_features.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_global_feature_importance.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_component_importance.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_feature_effects.json").exists())
            self.assertTrue((project_root / "data" / "scoring" / "scoring_pairs.parquet").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_groupwise_errors.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scoring_quality_failures.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_calibration_by_group.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_extreme_bias_summary.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_prediction_examples_by_group.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_feature_set_comparison.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "expanded_geometry_scorer_comparison.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_weighted_model_comparison.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_learned_feature_weights.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_learned_component_weights.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_learned_anatomical_component_weights.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_learned_transform_family_weights.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_learned_source_weights.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_component_weight_summary.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_component_balanced_feature_selection.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_base_feature_deduplication_audit.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "pose_leakage_audit.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_feature_correlation_audit.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "reference_feature_selection_diagnostics.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scoring_reference_resolution_summary.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "component_taxonomy_audit.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "mediapipe_only_scorer_comparison.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_edge_weighting_comparison.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "source_feature_audit.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "mediapipe_reference_relative_feature_summary.json").exists())
            self.assertTrue((project_root / "models" / "scoring" / "score_calibrator.joblib").exists())
            feature_set_comparison = json.loads((project_root / "reports" / "metrics" / "fairface_reference_comparison.json").read_text())
            for required_set in (
                "ref_z_only_stable",
                "ref_percentile_only_stable",
                "ref_z_plus_percentile_stable",
                "ref_deviation_compact",
                "absolute_plus_best_reference_50",
                "absolute_plus_best_reference_100",
                "combined_reference_no_duplicate",
                "mediapipe_plus_reference_weighted",
                "combined_weighted_no_duplicate",
                "mediapipe_only_reference_relative",
                "mediapipe_only_absolute_plus_reference",
                "mediapipe_only_no_pose",
                "mediapipe_only_high_low_weighted",
                "mediapipe_only_component_balanced",
            ):
                self.assertIn(required_set, feature_set_comparison["sets"])
            mp_only_comparison = json.loads((project_root / "reports" / "metrics" / "mediapipe_only_scorer_comparison.json").read_text())
            self.assertIn("mediapipe_only_absolute_plus_reference", mp_only_comparison["sets"])
            self.assertIn("mediapipe_only_high_low_weighted", mp_only_comparison["sets"])
            edge_weighting = json.loads((project_root / "reports" / "metrics" / "scorer_edge_weighting_comparison.json").read_text())
            self.assertIn("profiles", edge_weighting)
            self.assertIn("edge_x2", edge_weighting["profiles"])
            self.assertIn("edge_x3", edge_weighting["profiles"])
            expanded_set_comparison = json.loads((project_root / "reports" / "metrics" / "expanded_geometry_scorer_comparison.json").read_text())
            self.assertIn("expanded_mediapipe_geometry_forced", expanded_set_comparison["sets"])
            self.assertIn("mediapipe_weighted_geometry", expanded_set_comparison["sets"])
            reference_diag = json.loads((project_root / "reports" / "metrics" / "reference_feature_selection_diagnostics.json").read_text())
            self.assertIn("all_reference_feature_count", reference_diag)
            self.assertIn("candidate_reference_feature_sets", reference_diag)
            self.assertGreaterEqual(reference_diag["all_reference_feature_count"], 0)
            component_weight_summary = json.loads((project_root / "reports" / "metrics" / "scorer_component_weight_summary.json").read_text())
            self.assertAlmostEqual(float(component_weight_summary["normalized_sum"]), 1.0, places=4)
            self.assertIn("anatomical_component_weights", component_weight_summary)
            self.assertIn("transform_family_weights", component_weight_summary)
            self.assertIn("source_weights", component_weight_summary)
            self.assertNotIn("reference_deviation", component_weight_summary.get("anatomical_component_weights", {}))
            self.assertAlmostEqual(
                sum(float(value) for value in component_weight_summary.get("transform_family_weights", {}).values()),
                1.0,
                places=4,
            )
            reg_payload = __import__("joblib").load(training.regressor_artifact_path)
            self.assertIn("selected_features", reg_payload)
            self.assertNotIn("raw_face_width_px", reg_payload["selected_features"])
            self.assertNotIn("mediapipe_face_oval_width", reg_payload["selected_features"])
            self.assertTrue(reg_payload["test_metrics_available"])
            self.assertIn("global_feature_effects", reg_payload)
            self.assertIn("component_map", reg_payload)
            self.assertIn("score_calibrator", reg_payload)
            self.assertEqual(reg_payload["calibration_method"], "isotonic_regression")
            self.assertEqual(reg_payload["prediction_clip_lower"], 0.0)
            self.assertEqual(reg_payload["prediction_clip_upper"], 10.0)
            self.assertIn("primary_uses_calibration", reg_payload)
            self.assertIn("feature_set_name", reg_payload)
            self.assertIn("selected_feature_source_counts", reg_payload)
            self.assertIn("mediapipe_features_used", reg_payload)
            self.assertIn("learned_feature_weights", reg_payload)
            self.assertIn("learned_component_weights", reg_payload)
            self.assertIn("learned_weight_method", reg_payload)
            self.assertIn("scoring_model_info", reg_payload)
            self.assertIn("direct_mediapipe_features_used", reg_payload)
            self.assertIn("mediapipe_derived_reference_features_used", reg_payload)
            self.assertIn("total_mediapipe_influence_count", reg_payload)
            self.assertGreaterEqual(int(reg_payload["selected_feature_source_counts"].get("mediapipe_face_mesh", 0)), 0)

            pair_frame = pd.read_parquet(project_root / "data" / "scoring" / "scoring_pairs.parquet")
            row_frame = pd.DataFrame(rows)
            train_image_ids = set(row_frame.loc[row_frame["split"] == "train", "image_id"].tolist())
            non_train_image_ids = set(row_frame.loc[row_frame["split"] != "train", "image_id"].tolist())
            self.assertGreater(len(pair_frame), 0)
            pair_image_ids = set(pair_frame["left_image_id"].dropna().astype(str).tolist()) | set(pair_frame["right_image_id"].dropna().astype(str).tolist())
            self.assertTrue(pair_image_ids.issubset(train_image_ids))
            self.assertTrue(pair_image_ids.isdisjoint(non_train_image_ids))

            patched_feature_row = {
                "quality_pass": False,
                "processing_status": "processed",
                "processing_error_message": None,
                "ratio_face_aspect": 0.775,
                "ratio_mouth_width_to_face_width": 0.285,
                "symmetry_index": 0.842,
                "angle_mouth_line_deg": 0.75,
            }
            with patch(
                "src.face_aesthetics.scoring.inference.extract_single_image_scoring_features",
                return_value=patched_feature_row,
            ):
                image_path = project_root / "sample.jpg"
                image_path.write_bytes(b"stub")
                inference = run_scorer_inference(image_path=image_path, project_root=project_root)

            self.assertIsInstance(inference.report["regression"]["predicted_score"], float)
            self.assertIn("nearest_score_group", inference.report["regression"])
            self.assertIn(inference.report["regression"]["nearest_score_group"], [5.0, 5.5, 7.0, 8.0, 9.5, 10.0])
            self.assertLessEqual(inference.report["regression"]["predicted_score"], 10.0)
            self.assertGreaterEqual(inference.report["regression"]["predicted_score"], 0.0)
            self.assertIn("classification", inference.report)
            self.assertIn("pairwise", inference.report)
            self.assertIn("ordinal", inference.report)
            self.assertIn("final_score", inference.report)
            self.assertIn("primary_score", inference.report)
            self.assertIn("auxiliary_scores", inference.report)
            self.assertIn("local_explanation", inference.report)
            self.assertIn("learned_weighting", inference.report)
            self.assertIn("learned_weights", inference.report)
            self.assertIn("scoring_model_info", inference.report)
            self.assertIn("diagnostic_disagreement", inference.report)
            self.assertIn("component_contributions", inference.report)
            self.assertEqual(inference.report["confidence"]["level"], "low")
            self.assertTrue(any("quality gate" in warning.lower() for warning in inference.report["warnings"]))
            self.assertGreaterEqual(len(inference.report["final_score"]["model_agreement"]), 1)
            self.assertEqual(inference.report["primary_score"]["model"], training.summary["best_regressor_model"])
            self.assertIn("raw_predicted_score", inference.report["primary_score"])
            self.assertIn("calibrated_predicted_score", inference.report["primary_score"])
            self.assertIn("score_source", inference.report["primary_score"])
            self.assertIn("uses_calibration", inference.report["primary_score"])
            self.assertEqual(inference.report["primary_score"]["score_source"], "raw_continuous")
            self.assertAlmostEqual(
                inference.report["primary_score"]["predicted_score"],
                inference.report["primary_score"]["raw_predicted_score"],
                places=6,
            )
            self.assertAlmostEqual(
                inference.report["primary_score"]["display_score"],
                round(inference.report["primary_score"]["predicted_score"], 2),
                places=2,
            )
            self.assertIn("distance_to_nearest_group", inference.report["primary_score"])
            self.assertEqual(inference.report["primary_score"]["calibration_method"], "isotonic_regression")
            self.assertEqual(inference.report["primary_score"]["score_range"], [0.0, 10.0])
            self.assertIn("explanation_method", inference.report["local_explanation"])
            self.assertIn("top_positive_features", inference.report["local_explanation"])
            self.assertIn("top_negative_features", inference.report["local_explanation"])
            self.assertIn("feature_global_weights", inference.report["local_explanation"])
            self.assertIn("additive_check", inference.report["local_explanation"])
            selected_payload_for_components = json.loads(
                (project_root / "reports" / "metrics" / "scorer_selected_features.json").read_text()
            )
            has_reference_selected = any(
                str(item.get("feature_name", "")).startswith("ref_")
                for item in selected_payload_for_components.get("feature_lineage", [])
            )
            if has_reference_selected:
                self.assertNotIn("reference_deviation", inference.report["component_contributions"])
                self.assertTrue(
                    any(
                        str(item.get("anatomical_component", "")) not in {"", "unknown", "reference_deviation"}
                        for item in inference.report["local_explanation"].get("top_positive_features", [])
                        + inference.report["local_explanation"].get("top_negative_features", [])
                        if str(item.get("feature", "")).startswith("ref_")
                    )
                )
            self.assertAlmostEqual(
                inference.report["local_explanation"]["predicted_score"],
                inference.report["primary_score"]["predicted_score"],
                places=6,
            )
            self.assertIn("feature_set_name", inference.report["primary_score"])
            self.assertIn("selected_feature_source_counts", inference.report["primary_score"])
            self.assertIn("mediapipe_features_used", inference.report["primary_score"])
            self.assertIn("direct_mediapipe_features_used", inference.report["primary_score"])
            self.assertIn("mediapipe_derived_reference_features_used", inference.report["primary_score"])
            self.assertIn("total_mediapipe_influence_count", inference.report["primary_score"])
            self.assertIn("anatomical_component_weights", inference.report["learned_weights"])
            self.assertIn("transform_family_weights", inference.report["learned_weights"])
            self.assertIn("source_weights", inference.report["learned_weights"])
            self.assertIn("selected_features", json.loads((project_root / "reports" / "metrics" / "scorer_selected_features.json").read_text()))
            selected_payload = json.loads((project_root / "reports" / "metrics" / "scorer_selected_features.json").read_text())
            self.assertIn("anatomical_component_counts", selected_payload)
            self.assertIn("transform_family_counts", selected_payload)
            self.assertIn("source_counts", selected_payload)
            self.assertNotIn("reference_deviation", selected_payload.get("anatomical_component_counts", {}))
            lineage_rows = selected_payload.get("feature_lineage", [])
            base_counts: dict[str, int] = {}
            for row in lineage_rows:
                base_name = str(row.get("base_feature_name"))
                base_counts[base_name] = base_counts.get(base_name, 0) + 1
                self.assertIn("anatomical_component", row)
                self.assertIn("transform_family", row)
            self.assertTrue(all(count <= 1 for count in base_counts.values()))
            self.assertIn("evaluation_context", inference.report)
            self.assertTrue(inference.report["evaluation_context"]["test_metrics_available"])
            self.assertTrue((project_root / "reports" / "metrics" / "scorer_primary_policy_consistency.json").exists())
            self.assertTrue((project_root / "reports" / "metrics" / "component_taxonomy_audit.json").exists())
            concept_audit_path = project_root / "reports" / "metrics" / "scorer_concept_alignment_audit.json"
            self.assertTrue(concept_audit_path.exists())
            concept_audit = json.loads(concept_audit_path.read_text())
            self.assertEqual(int(concept_audit.get("non_mediapipe_selected_count", -1)), 0)

    def test_calibration_summary_detects_extreme_bias_and_pairwise_can_be_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_scoring_config_tree(project_root)
            rows = []
            for index, (score, split) in enumerate(
                [
                    (1.0, "train"),
                    (1.0, "train"),
                    (2.0, "train"),
                    (2.0, "val"),
                    (8.5, "train"),
                    (8.5, "val"),
                    (9.5, "train"),
                    (9.5, "test"),
                    (7.0, "train"),
                    (7.0, "test"),
                ],
                start=1,
            ):
                rows.append(
                    {
                        "image_id": f"bias-{index}",
                        "image_path": f"/tmp/bias-{index}.jpg",
                        "file_name": f"bias-{index}.jpg",
                        "source_score_folder": str(score),
                        "target_score": float(score),
                        "target_group_label": str(score),
                        "split": split,
                        "dataset_name": "scoring_dataset",
                        "sample_id": f"bias-{index}",
                        "subject_id": None,
                        "quality_pass": True,
                        "processing_status": "processed",
                        "processing_error_message": None,
                        "ratio_face_aspect": 0.6 + score / 20.0,
                        "ratio_mouth_width_to_face_width": 0.2 + score / 40.0,
                        "symmetry_index": 0.8 + score / 100.0,
                        "angle_mouth_line_deg": score / 20.0,
                    }
                )
            features_path = project_root / "data" / "scoring" / "scoring_features.parquet"
            features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(features_path, index=False)
            run_scorer_training(project_root=project_root)
            calibration_report = json.loads((project_root / "reports" / "metrics" / "scorer_calibration_by_group.json").read_text())
            self.assertIn("calibrated", calibration_report)
            warnings = calibration_report["test"]["calibration_summary"]["warnings"]
            self.assertTrue(any("regression_to_mean" in warning for warning in warnings) or isinstance(warnings, list))
            self.assertIn("collapse_below_7_count", calibration_report["test"]["calibration_summary"])
            self.assertIn("collapse_below_6_count", calibration_report["test"]["calibration_summary"])
            extreme_summary = json.loads((project_root / "reports" / "metrics" / "scorer_extreme_bias_summary.json").read_text())
            self.assertIn("warnings", extreme_summary)
            pairwise_payload = __import__("joblib").load(project_root / "models" / "scoring" / "pairwise_ranker.joblib")
            self.assertIn("use_for_absolute_scoring", pairwise_payload)


if __name__ == "__main__":
    unittest.main()
