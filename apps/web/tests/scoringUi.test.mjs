import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");

test("api client uses relative score-upload endpoint without localhost fallback", () => {
  const source = fs.readFileSync(path.join(root, "src/api/client.ts"), "utf-8");
  assert.match(source, /fetch\(`\/score-upload`|fetch\(\s*`\/score-upload`|fetch\(\s*"\/score-upload"/);
  assert.doesNotMatch(source, /VITE_API_BASE_URL \|\| "http:\/\/127\.0\.0\.1:8000"/);
});

test("analyze view includes isolated experimental score section", () => {
  const source = fs.readFileSync(path.join(root, "src/views/AnalyzeView.vue"), "utf-8");
  assert.match(source, /help-key="section\.experimental_score"/);
  assert.match(source, /ExperimentalScoreCard/);
  assert.match(source, /section\.expanded_geometry/);
  assert.match(source, /ExpandedGeometryCard/);
});

test("score card exposes low-confidence and warnings UI", () => {
  const source = fs.readFileSync(path.join(root, "src/components/ExperimentalScoreCard.vue"), "utf-8");
  assert.match(source, /score-card__confidence--low/);
  assert.match(source, /scoreWarningsTitle/);
  assert.match(source, /primary_score/);
  assert.match(source, /local_explanation/);
  assert.match(source, /component_contributions/);
  assert.match(source, /learned_weights/);
  assert.match(source, /scoring_model_info/);
  assert.match(source, /scoreDiagnosticTitle/);
  assert.match(source, /scoreWeightsTitle/);
  assert.match(source, /anatomical_component_weights/);
  assert.match(source, /transform_family_weights/);
  assert.match(source, /source_weights/);
  assert.doesNotMatch(source, /Точка отсчёта объяснения/);
});

test("score card uses primary score instead of classifier as main value", () => {
  const source = fs.readFileSync(path.join(root, "src/components/ExperimentalScoreCard.vue"), "utf-8");
  assert.match(source, /props\.score\?\.primary_score \?\? props\.score\?\.regression/);
  assert.match(source, /primaryScore\.value\.display_score \?\? primaryScore\.value\.predicted_score/);
  assert.match(source, /auxiliaryScores\.value\.classifier\?\.predicted_group/);
  assert.doesNotMatch(source, /score\.classification\.predicted_group[^]*score-card__value/);
});

test("nearest score group is rendered as metadata and not the main score", () => {
  const source = fs.readFileSync(path.join(root, "src/components/ExperimentalScoreCard.vue"), "utf-8");
  assert.match(source, /scoreNearestGroupLabel/);
  assert.match(source, /scoreDistanceToGroupLabel/);
  assert.match(source, /scoreNearestGroupDisclaimer/);
});

test("morphometric overlay exposes mediapipe toggles", () => {
  const source = fs.readFileSync(path.join(root, "src/components/MorphometricOverlay.vue"), "utf-8");
  assert.match(source, /toggleMediapipePoints/);
  assert.match(source, /toggleFaceOval/);
  assert.match(source, /toggleBrows/);
  assert.match(source, /toggleCheekbones/);
  assert.match(source, /toggleSymmetryAxes/);
  assert.doesNotMatch(source, /toggleInsightfacePoints/);
  assert.match(source, /faceOval:\s*true/);
  assert.match(source, /eyes:\s*true/);
  assert.match(source, /brows:\s*true/);
  assert.match(source, /mediapipePoints:\s*false/);
});
