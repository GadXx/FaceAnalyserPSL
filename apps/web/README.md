# Morphometric Web App

Vue 3 + Vite frontend for the existing objective morphometric API.

The UI supports:

- Russian / English language toggle
- real file upload via drag-and-drop or file picker
- image preview and clear action
- compact, section-based report presentation instead of full raw JSON by default
- SVG morphometric overlay on top of the uploaded image
- layer toggles for landmarks, eyes, nose, mouth, chin/jaw, symmetry, thirds,
  angles, measurements, debug mode, and labels
- reusable info-tooltips for sections, metrics, selectors, and overlay controls

## What It Uses

- `GET /health`
- `GET /reference-groups`
- `POST /analyze-upload`

The upload flow sends `multipart/form-data` with a real image file. The backend
stores the upload in its configured runtime directory and then reuses the
existing inference service. The older `POST /analyze` path-based mode remains
available for CLI and internal workflows.

## Morphometric Overlay

The overlay is drawn from real coordinates returned by the backend. The browser
does not guess or invent feature locations.

The current UI uses these report fields:

- `input_quality.detection.bbox`
- `visualization.image_size`
- `visualization.anchors`
- `visualization.named_points`
- `visualization.derived_points`
- `visualization.lines`
- `visualization.axes`
- `visualization.guides`
- `raw_measurements`
- `symmetry_metrics`
- `angle_metrics`
- `feature_level_deviations.top_contributing_features`

What is visualized:

- face bounding box
- 5 anchor landmarks:
  - `left_eye`
  - `right_eye`
  - `nose`
  - `left_mouth`
  - `right_mouth`
- main measurement lines:
  - eye line
  - mouth line
  - eye center to nose
  - nose to mouth center
  - eye center to mouth center
- symmetry axis and left/right asymmetry distances
- vertical thirds and horizontal proportion guides
- angle-related lines for eye line, mouth line, brow line, nose axis,
  philtrum axis, chin axis, and eye-to-mouth / nose-to-mouth direction
- dense 106-landmark debug mode with optional landmark indices

What stays numeric-only:

- features that do not map cleanly to the returned 2D coordinates
- ratios and indices that are derived from already drawn primitives but do not
  introduce a new trustworthy 2D anchor by themselves
- descriptive reference statistics that are not directly drawable on the image

If landmarks are missing, the UI degrades gracefully and shows only what can be
drawn safely. If the quality gate failed, the overlay is still shown when
coordinates are available, but with a warning banner.

## Help Tooltips

The web app includes a reusable explanation system for metrics and important UI
controls.

Files involved:

- `src/components/InfoTooltip.vue`
- `src/components/MetricLabel.vue`
- `src/content/helpText.js`

How it works:

- the UI passes a stable `helpKey`
- `MetricLabel` renders the visible label plus the circular info icon
- `InfoTooltip` resolves the localized explanation from `helpText.js`
- if no entry exists, the info icon is omitted and the UI does not break

What is covered now:

- major report sections
- quality / preprocessing metrics
- pose fields
- common raw measurements
- normalized ratios and symmetry metrics
- reference comparison terms
- composite indices
- reference-group controls
- overlay layer toggles

To add a new explanation:

1. add a new entry to `src/content/helpText.js`
2. add an alias if the backend key and the help key differ
3. pass the matching `helpKey` from the component that renders the label

## Install

```bash
cd apps/web
npm install
```

## Configure Backend URL

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Set:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Run Dev Server

```bash
npm run dev
```

## Production Build

```bash
npm run build
npm run preview
```

## Smoke Checks

```bash
npm run test:smoke
npm run build
```

## Backend Requirements

The backend must be running and must have:

- `python-multipart` installed for `POST /analyze-upload`
- InsightFace 2D models in `data/interim/insightface_models/models/buffalo_l/`
- reference artifacts in `models/reference/`

## Upload Flow

1. Select or drag an image file.
2. The app validates type and size before sending.
3. The frontend calls `POST /analyze-upload`.
4. The backend stores the file in its configured runtime upload directory.
5. The backend reuses the existing morphometric inference service and returns the
   same structured report used by the CLI/API JSON path.
