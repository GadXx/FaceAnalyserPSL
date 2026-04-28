import test from "node:test";
import assert from "node:assert/strict";

import { getHelpEntry, getHelpLabel, resolveHelpKey } from "../src/content/helpText.js";

test("resolves stable aliases for metric keys", () => {
  assert.equal(resolveHelpKey("pitch"), "pose_pitch_deg");
  assert.equal(resolveHelpKey("quality_pass"), "quality_pass");
  assert.equal(resolveHelpKey("toggle.labels"), "toggle.labels");
});

test("returns localized help entry for known metric", () => {
  const entry = getHelpEntry("mahalanobis_distance", "en");
  assert.ok(entry);
  assert.match(entry.short, /statistical distance/i);
});

test("returns null when help text is missing", () => {
  assert.equal(getHelpEntry("totally_unknown_metric", "en"), null);
});

test("returns fallback label when localized label is not defined", () => {
  assert.equal(getHelpLabel("quality_pass", "en", "Quality Pass"), "Quality Pass");
});
