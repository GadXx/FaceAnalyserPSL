from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.face_aesthetics.data.io import read_parquet


class DataIoTests(unittest.TestCase):
    def test_read_parquet_directory_handles_mixed_null_and_string_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset.parquet"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "image_id": "img-1",
                        "normalized_labels": '{"age_group":"20-29"}',
                        "processing_error_message": None,
                    }
                ]
            ).to_parquet(dataset_dir / "part-000000.parquet", index=False)
            pd.DataFrame(
                [
                    {
                        "image_id": "img-2",
                        "normalized_labels": '{"age_group":"30-39"}',
                        "processing_error_message": "detector_failed",
                    }
                ]
            ).to_parquet(dataset_dir / "part-000001.parquet", index=False)

            rows = read_parquet(dataset_dir, columns=["image_id", "normalized_labels", "processing_error_message"])

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["image_id"] for row in rows}, {"img-1", "img-2"})
            self.assertIn("processing_error_message", rows[0])


if __name__ == "__main__":
    unittest.main()
