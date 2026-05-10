import test from "node:test";
import assert from "node:assert/strict";

import {
  getMeasurementOverlaySegments,
  getRawLandmarks,
  getSemanticPoints,
} from "../src/utils/visualizationPayload.js";

test("raw landmark accessor uses validated raw_landmarks payload only", () => {
  const visualization = {
    raw_landmarks: [{ index: 12, x: 10, y: 20 }],
    landmarks: [{ index: 0, x: 1, y: 2 }],
  };
  assert.deepEqual(getRawLandmarks(visualization), [{ index: 12, x: 10, y: 20 }]);
  assert.deepEqual(getRawLandmarks({ landmarks: [{ index: 0, x: 1, y: 2 }] }), []);
});

test("semantic point accessor uses semantic_points payload only", () => {
  const visualization = {
    semantic_points: { mouth_left: { x: 11, y: 21 } },
    named_points: { mouth_left: { x: 999, y: 999 } },
  };
  assert.deepEqual(getSemanticPoints(visualization), { mouth_left: { x: 11, y: 21 } });
  assert.deepEqual(getSemanticPoints({ named_points: { mouth_left: { x: 10, y: 10 } } }), {});
});

test("measurement overlay accessor uses measurement_overlays keys only", () => {
  const visualization = {
    measurement_overlays: {
      lines: [{ key: "mouth_width_line" }],
      axes: [{ key: "facial_midline" }],
      guides: [{ key: "thirds_mouth_guide" }],
    },
    lines: [{ key: "legacy_line" }],
  };
  assert.deepEqual(getMeasurementOverlaySegments(visualization, "lines"), [{ key: "mouth_width_line" }]);
  assert.deepEqual(getMeasurementOverlaySegments(visualization, "axes"), [{ key: "facial_midline" }]);
  assert.deepEqual(getMeasurementOverlaySegments(visualization, "guides"), [{ key: "thirds_mouth_guide" }]);
  assert.deepEqual(getMeasurementOverlaySegments({ lines: [{ key: "legacy" }] }, "lines"), []);
});
