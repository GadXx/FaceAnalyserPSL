import test from "node:test";
import assert from "node:assert/strict";

import { selectResultImage } from "../src/utils/resultImageSelection.js";

const buildArtifactFileUrl = (path) => (path ? `http://127.0.0.1:8000/artifact-file?path=${encodeURIComponent(path)}` : null);

test("does not use annotated image as the primary UI result when analysis image is available", () => {
  const result = {
    generated_artifacts: {
      annotated_image_path: "reports/examples/annotated_analysis.jpg",
      analysis_image_path: "reports/examples/aligned_face.jpg",
      aligned_image_path: "reports/examples/aligned_face.jpg",
    },
    artifacts: {
      original_image_path: "data/interim/runtime_uploads/original.jpg",
    },
  };

  const selected = selectResultImage(result, 111, buildArtifactFileUrl);
  assert.equal(selected.source, "generated_artifacts.analysis_image_path");
  assert.match(selected.url, /aligned_face\.jpg/);
  assert.match(selected.url, /[?&]v=111/);
});

test("uses analysis image when annotated image is missing", () => {
  const result = {
    generated_artifacts: {
      analysis_image_path: "reports/examples/analysis_image.jpg",
    },
    artifacts: {
      aligned_image_path: "reports/examples/aligned_face.jpg",
      original_image_path: "data/interim/runtime_uploads/original.jpg",
    },
  };

  const selected = selectResultImage(result, 222, buildArtifactFileUrl);
  assert.equal(selected.source, "generated_artifacts.analysis_image_path");
  assert.match(selected.url, /analysis_image\.jpg/);
  assert.match(selected.url, /[?&]v=222/);
});

test("uses aligned image when only aligned image is available", () => {
  const result = {
    generated_artifacts: {
      aligned_image_path: "reports/examples/aligned_face.jpg",
    },
    artifacts: {},
  };

  const selected = selectResultImage(result, 333, buildArtifactFileUrl);
  assert.equal(selected.source, "generated_artifacts.aligned_image_path");
  assert.match(selected.url, /aligned_face\.jpg/);
});

test("falls back to original image path only after result artifacts are exhausted", () => {
  const result = {
    generated_artifacts: {},
    artifacts: {
      original_image_path: "data/interim/runtime_uploads/original.jpg",
    },
  };

  const selected = selectResultImage(result, 444, buildArtifactFileUrl);
  assert.equal(selected.source, "artifacts.original_image_path");
  assert.match(selected.url, /original\.jpg/);
});

test("returns no result image when backend did not provide any result artifact path", () => {
  const selected = selectResultImage({ generated_artifacts: {}, artifacts: {} }, 555, buildArtifactFileUrl);
  assert.equal(selected.source, "none");
  assert.equal(selected.url, null);
});

test("cache-busting token changes between repeated analyses", () => {
  const result = {
    generated_artifacts: {
      analysis_image_path: "reports/examples/analysis_image.jpg",
    },
  };

  const first = selectResultImage(result, 1000, buildArtifactFileUrl);
  const second = selectResultImage(result, 2000, buildArtifactFileUrl);
  assert.notEqual(first.url, second.url);
});
