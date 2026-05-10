from __future__ import annotations

from pathlib import Path

import numpy as np


class InsightFaceEmbedder:
    def __init__(
        self,
        encoder_name: str,
        encoder_model_path: Path,
        providers: list[str],
        ctx_id: int,
        expected_embedding_dim: int,
    ) -> None:
        try:
            from insightface.model_zoo import get_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "InsightFace is not available in the active Python environment."
            ) from exc

        if not encoder_model_path.exists():
            raise FileNotFoundError(f"Encoder model file not found: {encoder_model_path}")

        self.encoder_name = encoder_name
        self.expected_embedding_dim = expected_embedding_dim
        self._model = get_model(str(encoder_model_path), providers=providers)
        if self._model is None:
            raise RuntimeError(f"Unable to initialize encoder from {encoder_model_path}")
        self._model.prepare(ctx_id=ctx_id)

    @property
    def embedding_dim(self) -> int:
        output_shape = getattr(self._model, "output_shape", None)
        if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 2:
            dimension = int(output_shape[-1])
            if dimension > 0:
                return dimension
        return self.expected_embedding_dim

    def encode(self, aligned_image: np.ndarray) -> np.ndarray:
        embedding = self._model.get_feat(aligned_image)
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if vector.size != self.embedding_dim:
            raise ValueError(
                f"Unexpected embedding size {vector.size}, expected {self.embedding_dim}"
            )

        return vector
