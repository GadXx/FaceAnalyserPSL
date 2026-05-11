import test from "node:test";
import assert from "node:assert/strict";

import {
  getDisplayFaceBox,
  getMediapipeLandmarks,
  getMediapipeRegions,
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

test("mediapipe accessors use dedicated payload keys only", () => {
  const visualization = {
    mediapipe_landmarks: [{ index: 1, x: 11, y: 21 }],
    mediapipe_regions: { dense_face_oval: [1, 2, 3] },
    raw_landmarks: [{ index: 0, x: 1, y: 2 }],
  };
  assert.deepEqual(getMediapipeLandmarks(visualization), [{ index: 1, x: 11, y: 21 }]);
  assert.deepEqual(getMediapipeRegions(visualization), { dense_face_oval: [1, 2, 3] });
  assert.deepEqual(getMediapipeLandmarks({ raw_landmarks: [] }), []);
  assert.deepEqual(getMediapipeRegions({}), {});
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

test("display face box accessor prefers semantic/display face box over legacy bbox", () => {
  const visualization = {
    display_face_box: [11, 22, 77, 99],
    bbox: [0, 0, 100, 120],
  };
  assert.deepEqual(getDisplayFaceBox(visualization, [5, 5, 50, 50]), [11, 22, 77, 99]);
  assert.deepEqual(getDisplayFaceBox({ bbox: [1, 2, 3, 4] }, [5, 5, 50, 50]), [1, 2, 3, 4]);
  assert.deepEqual(getDisplayFaceBox({}, [5, 5, 50, 50]), [5, 5, 50, 50]);
});
