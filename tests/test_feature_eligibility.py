from __future__ import annotations

import unittest

from src.face_aesthetics.pipelines.feature_extraction import _eligible_rows


class FeatureEligibilityTests(unittest.TestCase):
    def test_accepts_processed_and_success_statuses(self) -> None:
        rows, counters = _eligible_rows(
            [
                {
                    "image_id": "a",
                    "quality_pass": True,
                    "processing_status": "processed",
                    "bbox": "[1,2,3,4]",
                    "landmarks": "[[1,2],[3,4],[5,6],[7,8],[9,10]]",
                    "aligned_image_path": "data/processed/aligned_faces/a.jpg",
                },
                {
                    "image_id": "b",
                    "quality_pass": True,
                    "processing_status": "success",
                    "bbox": "[1,2,3,4]",
                    "landmarks": "[[1,2],[3,4],[5,6],[7,8],[9,10]]",
                    "aligned_image_path": "data/processed/aligned_faces/b.jpg",
                },
            ],
            quality_pass_column="quality_pass",
            processing_status_column="processing_status",
            required_processing_status="processed",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(counters["processing_status_filtered_rows"], 0)

    def test_accepts_legacy_rows_without_processing_status_only_if_geometry_present(self) -> None:
        rows, counters = _eligible_rows(
            [
                {
                    "image_id": "legacy-ok",
                    "quality_pass": True,
                    "bbox": "[1,2,3,4]",
                    "landmarks": "[[1,2],[3,4],[5,6],[7,8],[9,10]]",
                    "aligned_image_path": "data/processed/aligned_faces/ok.jpg",
                },
                {
                    "image_id": "legacy-bad",
                    "quality_pass": True,
                    "bbox": "[1,2,3,4]",
                    "landmarks": None,
                    "aligned_image_path": None,
                },
            ],
            quality_pass_column="quality_pass",
            processing_status_column="processing_status",
            required_processing_status="processed",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], "legacy-ok")
        self.assertEqual(counters["processing_status_filtered_rows"], 1)


if __name__ == "__main__":
    unittest.main()
