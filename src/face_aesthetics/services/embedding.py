from __future__ import annotations

from pathlib import Path

import numpy as np


class MediaPipeEmbedder:
    def __init__(
        self,
        encoder_name: str,
        encoder_model_path: Path,
        providers: list[str],
        ctx_id: int,
        expected_embedding_dim: int,
    ) -> None:
        raise RuntimeError(
            "Auxiliary embedding branch is disabled in MediaPipe-only mode."
        )

    @property
    def embedding_dim(self) -> int:
        return 0

    def encode(self, aligned_image: np.ndarray) -> np.ndarray:
        return np.asarray([], dtype=np.float32)
