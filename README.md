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

Python 3.9+ is required. The recommended setup uses conda:

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

The Vite config proxies `/cluster`, `/shap`, `/annotate`, and `/annotations` to the backend automatically.

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
