from contextlib import asynccontextmanager
import gzip
import io
import os
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import anndata
import igraph  # verify python-igraph is present (required by leiden igraph flavor)
import pandas as pd
import scipy.sparse
from scipy.io import mmread
import numpy as np
from hdbscan import HDBSCAN
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder
import scanpy as sc


PANGLAO_URL = (
    "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
)


def preprocess(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    return adata


@asynccontextmanager
async def lifespan(app: FastAPI):
    adata = sc.datasets.pbmc3k()
    app.state.adata = preprocess(adata)
    app.state.dataset_name = "pbmc3k"

    # Download PanglaoDB human marker gene reference (human "Hs" entries only)
    try:
        with urllib.request.urlopen(PANGLAO_URL, timeout=30) as resp:
            raw = resp.read()
        with gzip.open(io.BytesIO(raw)) as f:
            content = f.read().decode("utf-8")
        marker_db: dict[str, set[str]] = {}
        for line in content.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            species, gene, cell_type = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if "Hs" in species and gene and cell_type:
                marker_db.setdefault(cell_type, set()).add(gene)
        app.state.marker_db = marker_db
    except Exception:
        app.state.marker_db = {}

    app.state.annotations: dict = {}
    yield  # server runs here; cleanup (if any) goes after this line


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusterRequest(BaseModel):
    resolution: float = 0.5
    algorithm: str = "leiden"       # "leiden" | "hdbscan"
    min_cluster_size: int = 50
    min_samples: int = 5


@app.post("/cluster")
async def cluster(req: ClusterRequest):
    """
    Re-run clustering on the pre-processed AnnData stored in app.state.
    Cluster labels are always written to adata.obs["labels"] so that
    downstream endpoints (e.g. /shap) have a single stable key to read from.

    Returns:
        {
            "points": [{"x": float, "y": float, "cluster": str}, ...],
            "n_clusters": int,   # unique real-cluster count (excludes noise)
            "n_noise": int       # cells labeled "-1" by HDBSCAN; 0 for Leiden
        }
    """
    adata = app.state.adata

    if req.algorithm == "hdbscan":
        X_pca = adata.obsm["X_pca"][:, :30]
        clusterer = HDBSCAN(
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
        )
        raw_labels = clusterer.fit_predict(X_pca)
        adata.obs["labels"] = [str(l) for l in raw_labels]
    else:
        sc.tl.leiden(adata, resolution=req.resolution, flavor="igraph", n_iterations=2)
        adata.obs["labels"] = adata.obs["leiden"].astype(str)

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    xs = adata.obsm["X_umap"][:, 0].tolist()
    ys = adata.obsm["X_umap"][:, 1].tolist()
    label_col = adata.obs["labels"]

    points = [
        {"x": x, "y": y, "cluster": c}
        for x, y, c in zip(xs, ys, label_col.tolist())
    ]
    n_noise    = int((label_col == "-1").sum())
    n_clusters = int(label_col[label_col != "-1"].nunique())

    return {"points": points, "n_clusters": n_clusters, "n_noise": n_noise}


@app.post("/shap")
def shap_endpoint():
    """
    Train a LightGBM multiclass classifier on the current cluster labels
    (whatever algorithm last ran is stored in adata.obs["leiden"]) and
    compute SHAP values via TreeExplainer.

    For each cluster, returns the top 10 genes ranked by mean absolute
    SHAP value across all cells assigned to that cluster.

    Results are cached in app.state.shap_cache keyed by a hash of the
    label array; repeated calls with unchanged cluster assignments return
    the cached result immediately without retraining.

    Returns:
        {
            "clusters": {
                "0": [{"gene": str, "shap": float}, ...],  # top 10
                "1": [...],
                ...
            }
        }
    """
    adata = app.state.adata
    # Use the unified "labels" column written by /cluster (works for both
    # Leiden and HDBSCAN). Exclude noise cells ("-1") before training.
    all_labels = adata.obs["labels"].to_numpy()
    noise_mask = all_labels == "-1"
    labels = all_labels[~noise_mask]

    # ── Cache check ──────────────────────────────────────────────────────────
    label_hash = hash(labels.tobytes())
    cached = getattr(app.state, "shap_cache", None)
    if cached is not None and cached["hash"] == label_hash:
        return cached["result"]

    # ── Feature matrix (noise rows already excluded via label mask) ──────────
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)[~noise_mask]

    gene_names = list(adata.var_names)

    # ── Encode cluster labels ────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    cluster_ids = [str(c) for c in le.classes_]  # original string labels

    # ── Train LightGBM ───────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1,
    )
    model.fit(X, y_encoded)

    # ── SHAP values ──────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Newer SHAP versions return a 3-D array (n_cells, n_genes, n_classes).
    # Older versions return a list of (n_cells, n_genes) arrays, one per class.
    # Normalise to 3-D so the indexing below is version-independent.
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)  # → (n_cells, n_genes, n_classes)

    # ── Top-10 genes per cluster ─────────────────────────────────────────────
    clusters_out: dict[str, list[dict]] = {}
    for class_idx, cluster_id in enumerate(cluster_ids):
        cell_mask = y_encoded == class_idx
        # Mean absolute SHAP across cells in this cluster, for this class
        mean_abs = np.abs(shap_values[:, :, class_idx][cell_mask]).mean(axis=0)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        clusters_out[cluster_id] = [
            {"gene": gene_names[i], "shap": float(mean_abs[i])}
            for i in top10_idx
        ]

    result = {"clusters": clusters_out}

    # ── Store in cache ───────────────────────────────────────────────────────
    app.state.shap_cache = {"hash": label_hash, "result": result}

    return result


class AnnotateRequest(BaseModel):
    cluster_id: str


@app.post("/annotate")
def annotate(req: AnnotateRequest):
    """
    Given a cluster ID, return the top 5 cell-type suggestions from PanglaoDB
    ranked by overlap between the cluster's top-50 expressed genes and each
    cell type's known marker set.
    """
    adata = app.state.adata
    marker_db: dict[str, set[str]] = getattr(app.state, "marker_db", {})

    if not marker_db:
        return {"cluster_id": req.cluster_id, "suggestions": []}

    mask = adata.obs["labels"] == req.cluster_id
    if not mask.any():
        return {"cluster_id": req.cluster_id, "suggestions": []}

    X_cluster = adata[mask].X
    if hasattr(X_cluster, "toarray"):
        X_cluster = X_cluster.toarray()
    mean_expr = np.asarray(X_cluster, dtype=np.float32).mean(axis=0).flatten()

    gene_names = list(adata.var_names)
    top50_idx = np.argsort(mean_expr)[::-1][:50]
    top50_genes = {gene_names[i] for i in top50_idx}

    scores = []
    for cell_type, markers in marker_db.items():
        if not markers:
            continue
        overlap = len(top50_genes & markers) / len(markers)
        if overlap > 0:
            scores.append({"cell_type": cell_type, "score": round(overlap, 4)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"cluster_id": req.cluster_id, "suggestions": scores[:5]}


class SaveAnnotationRequest(BaseModel):
    cluster_id: str
    label: str
    status: str  # "confirmed" | "suggested" | "unannotated"


@app.post("/annotations/save")
def save_annotation(req: SaveAnnotationRequest):
    if not hasattr(app.state, "annotations"):
        app.state.annotations = {}
    app.state.annotations[req.cluster_id] = {"label": req.label, "status": req.status}
    return {"ok": True}


@app.get("/annotations")
def get_annotations():
    return {"annotations": getattr(app.state, "annotations", {})}


class LoadDatasetRequest(BaseModel):
    dataset: str


@app.post("/load-dataset")
async def load_dataset(req: LoadDatasetRequest):
    if req.dataset != "pbmc3k":
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.dataset}")
    adata = sc.datasets.pbmc3k()
    app.state.adata = preprocess(adata)
    app.state.dataset_name = "pbmc3k"
    app.state.annotations = {}
    return {
        "ok": True,
        "dataset": "pbmc3k",
        "n_cells": int(app.state.adata.n_obs),
        "n_genes": int(app.state.adata.n_vars),
    }


def _read_uploaded(tmp_path: str, filename: str) -> anndata.AnnData:
    """Parse an uploaded file into AnnData. Supports .h5ad, .csv/.tsv, .mtx, .zip."""
    ext = pathlib.Path(filename.lower()).suffix

    if ext == '.h5ad':
        return anndata.read_h5ad(tmp_path)

    if ext in ('.csv', '.tsv'):
        df = pd.read_csv(tmp_path, index_col=0, sep=None, engine='python')
        # Transpose if more columns than rows (genes × cells → cells × genes)
        if df.shape[1] > df.shape[0]:
            df = df.T
        return anndata.AnnData(
            X=np.asarray(df.values, dtype=np.float32),
            obs=pd.DataFrame(index=df.index.astype(str)),
            var=pd.DataFrame(index=df.columns.astype(str)),
        )

    if ext == '.mtx':
        mat = mmread(tmp_path)
        X = mat.toarray() if scipy.sparse.issparse(mat) else np.asarray(mat)
        X = X.astype(np.float32)
        # MTX convention is genes × cells; transpose if rows > cols
        if X.shape[0] > X.shape[1]:
            X = X.T
        n_obs, n_vars = X.shape
        return anndata.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)]),
            var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars)]),
        )

    if ext == '.zip':
        tmpdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(tmpdir)
            mtx_files = list(pathlib.Path(tmpdir).rglob('*.mtx*'))
            if not mtx_files:
                raise ValueError("No .mtx file found in the zip archive.")
            return sc.read_10x_mtx(
                str(mtx_files[0].parent), var_names='gene_symbols', cache=False,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    raise ValueError(
        f"Unsupported format '{ext}'. Accepted: .h5ad, .csv, .tsv, .mtx, .zip"
    )


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = pathlib.Path(filename.lower()).suffix
    allowed = {'.h5ad', '.csv', '.tsv', '.mtx', '.zip'}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: .h5ad, .csv, .tsv, .mtx, .zip",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        adata = _read_uploaded(tmp_path, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if adata.n_obs < 50:
        raise HTTPException(status_code=422, detail="Dataset must contain at least 50 cells.")
    if adata.n_vars < 100:
        raise HTTPException(status_code=422, detail="Dataset must contain at least 100 genes.")

    try:
        app.state.adata = preprocess(adata)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")

    app.state.dataset_name = filename
    app.state.annotations = {}

    return {
        "ok": True,
        "dataset": filename,
        "n_cells": int(app.state.adata.n_obs),
        "n_genes": int(app.state.adata.n_vars),
    }


@app.get("/dataset-info")
def dataset_info():
    return {
        "dataset": getattr(app.state, "dataset_name", "pbmc3k"),
        "n_cells": int(app.state.adata.n_obs),
        "n_genes": int(app.state.adata.n_vars),
    }


# Serve the compiled React app from the dist/ folder
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
