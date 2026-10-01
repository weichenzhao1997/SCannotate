# SCannotate: Interactive scRNA-seq Clustering and Annotation Tool

SCannotate is an interactive web application for exploratory single-cell RNA sequencing (scRNA-seq) analysis. It loads the PBMC 3k dataset, runs a standard preprocessing pipeline on startup, and exposes an interactive three-panel interface for clustering, driver gene interpretation, and cell type annotation with no coding required.

User can also upload their own dataset for custom analysis.

## Overview

SCannotate provides three tightly coupled points of interaction:

1. **Model steering** — Users tune parameters for two clustering algorithms: Leiden community detection (resolution slider) and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN, with min cluster size and min samples sliders). The UMAP projection updates live as parameters change.

2. **Model output validation and interpretability** — A Light Gradient-Boosting Machine (LightGBM) multiclass classifier is trained on the current cluster assignments, and Shapley Additive Explanations (SHAP) values are computed via `TreeExplainer` to surface the driver genes most discriminatively distinguishing each cluster's identity.

3. **Annotation interface** — Users can assign cell type labels manually or accept automated suggestions computed by matching each cluster's top-50 expressed genes against the PanglaoDB marker gene database. Confirmed annotations persist in the server session and are reflected in real time across the annotation list.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (served via Uvicorn) |
| Frontend | React 18 + TypeScript (Vite) |
| Clustering | Scanpy (Leiden via `leidenalg` / `python-igraph`), HDBSCAN |
| Preprocessing | Scanpy (filter, normalize, log1p, HVG selection, PCA, KNN graph) |
| Driver genes | LightGBM, SHAP |
| Annotation | PanglaoDB marker gene reference (downloaded at startup) |
| Visualization | react-plotly.js |

## Prerequisites

### Python environment

Python 3.11+ is required. The recommended setup uses conda:

```bash
conda env create -f backend/environment.yml
conda activate scannotate
```

Or install directly with pip:

```bash
pip install -r backend/requirements.txt
```

### Node.js (only needed if modifying the frontend)

Download the LTS release from [nodejs.org](https://nodejs.org). Verify with:

```bash
node -v
npm -v
```

## Running the App

The compiled frontend is already bundled into `backend/dist/`. To start the application:

```bash
cd backend
uvicorn main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000). The server preprocesses the PBMC 3k dataset and downloads the PanglaoDB marker reference on first startup — this takes about 30–60 seconds.

The `--reload` flag is optional; omit it in production.

## Modifying the Frontend

If you change any file under `frontend/src/`, you need to rebuild and re-bundle before the backend serves the updated UI:

```bash
# From the frontend directory
npm install        # first time only
npm run build      # outputs to frontend/dist/

# Copy the build output to where the backend expects it
cp -r dist/ ../backend/dist/
```

Then restart (or let `--reload` pick it up) the backend server.

For live development with hot-module replacement, run both servers simultaneously:

```bash
# Terminal 1 — backend
cd backend && uvicorn main:app --reload

# Terminal 2 — frontend dev server
cd frontend && npm run dev
# UI available at http://localhost:5173
```

The Vite config proxies the API routes (`/cluster`, `/shap`, `/annotate`, `/annotations`, `/load-dataset`, `/upload-dataset`, `/dataset-info`, `/export`) to the backend automatically.

## Testing

Backend tests use a small synthetic dataset with three known cell groups, so they run offline in about 20 seconds (no PBMC 3k or PanglaoDB download):

```bash
cd backend
pip install -r requirements-dev.txt   # pytest + httpx, on top of environment.yml
pytest
```

They cover preprocessing, file parsing (`.h5ad`, CSV/TSV orientation, 10x `.zip`), every endpoint, and check that clustering, SHAP driver genes, and marker-based suggestions recover the planted groups.

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push to `main` and on pull requests:

- **Backend**: builds the conda environment from `environment.yml` and runs `pytest`
- **Frontend**: `npm run lint`, then `npm run build` (type-check + bundle), and fails if the committed `backend/dist/` doesn't match the fresh build

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/cluster` | Re-run clustering; returns UMAP coordinates and cluster labels |
| `POST` | `/shap` | Train LightGBM + compute SHAP values; returns top-10 driver genes per cluster |
| `POST` | `/annotate` | Return top-5 PanglaoDB cell type suggestions for a cluster |
| `POST` | `/annotations/save` | Persist a user-confirmed or suggested label for a cluster |
| `GET`  | `/annotations` | Retrieve all saved annotations for the current session |
| `POST` | `/upload-dataset` | Upload a custom dataset (`.h5ad`, `.csv`, `.tsv`, `.zip`) to replace the default PBMC 3k data |

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) while the server is running.
