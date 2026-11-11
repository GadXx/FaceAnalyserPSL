from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LANDMARK_COUNT = 478


@dataclass
class LandmarkSourceResult:
    source: str
    enabled: bool
    available: bool
    required: bool = False
    optional: bool = True
    coordinate_space: str = "analysis_image"
    confidence: float | None = None
    landmark_count: int = 0
    landmarks: list[list[float]] = field(default_factory=list)
    available_regions: list[str] = field(default_factory=list)
    direct_regions: list[str] = field(default_factory=list)
    proxy_regions: list[str] = field(default_factory=list)
    unavailable_regions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dependency_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "enabled": self.enabled,
            "available": self.available,
            "required": self.required,
            "optional": self.optional,
            "coordinate_space": self.coordinate_space,
            "confidence": self.confidence,
            "landmark_count": self.landmark_count,
            "landmarks": [list(point) for point in self.landmarks],
            "available_regions": list(self.available_regions),
            "direct_regions": list(self.direct_regions),
            "proxy_regions": list(self.proxy_regions),
            "unavailable_regions": list(self.unavailable_regions),
            "warnings": list(self.warnings),
            "dependency_message": self.dependency_message,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "LANDMARK_COUNT",
    "LandmarkSourceResult",
]
