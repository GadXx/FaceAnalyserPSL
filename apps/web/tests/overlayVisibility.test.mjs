import test from "node:test";
import assert from "node:assert/strict";

import { pointGroupMatches, segmentLayerVisible } from "../src/utils/overlayVisibility.js";

const enabledLayers = {
  measurements: true,
  angles: true,
  symmetry: true,
  thirds: true,
  eyes: true,
  nose: true,
  mouth: true,
  chinJaw: true,
};

test("segment visibility respects new visual groups", () => {
  assert.equal(
    segmentLayerVisible({ layers: ["measurements", "eyes"] }, enabledLayers),
    true,
  );
  assert.equal(
    segmentLayerVisible({ layers: ["thirds", "brows"] }, { ...enabledLayers, thirds: false }),
    false,
  );
  assert.equal(
    segmentLayerVisible({ layers: ["symmetry", "nose"] }, { ...enabledLayers, symmetry: false }),
    false,
  );
});

test("point group filtering respects chin_jaw and thirds groups", () => {
  assert.equal(pointGroupMatches("chin_jaw", enabledLayers), true);
  assert.equal(pointGroupMatches("chin_jaw", { ...enabledLayers, chinJaw: false }), false);
  assert.equal(pointGroupMatches("brows", { ...enabledLayers, thirds: false }), false);
  assert.equal(pointGroupMatches("midline", { ...enabledLayers, symmetry: false }), false);
});

test("unknown groups fail open instead of crashing", () => {
  assert.equal(pointGroupMatches("misc", enabledLayers), true);
  assert.equal(segmentLayerVisible({ layers: [] }, enabledLayers), true);
});
