from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import igraph  # verify python-igraph is present (required by leiden igraph flavor)
import numpy as np
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder
import scanpy as sc



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load PBMC 3k and run the standard preprocessing pipeline once on startup.
    # The resulting AnnData (with neighbor graph) is stored in app.state so it
    # persists across requests without re-downloading or re-computing.
    adata = sc.datasets.pbmc3k()

    # Quality-control filters
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalisation and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select highly variable genes, then subset and scale for PCA
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)

    # Dimensionality reduction and KNN graph (done once; leiden/UMAP reuse this)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    app.state.adata = adata
    yield  # server runs here; cleanup (if any) goes after this line


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusterRequest(BaseModel):
    resolution: float
    algorithm: str = "leiden"  # "leiden" | "hdbscan"


@app.post("/cluster")
async def cluster(req: ClusterRequest):
    """
    Re-run clustering at the requested resolution on the pre-processed AnnData
    stored in app.state.  UMAP coordinates are computed once and reused on
    subsequent calls (they depend only on the neighbor graph, not the labels).

    Returns:
        {
            "points": [{"x": float, "y": float, "cluster": str}, ...],
            "n_clusters": int
        }
    """
    adata = app.state.adata

    sc.tl.leiden(adata, resolution=req.resolution, flavor="igraph", n_iterations=2)

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    xs = adata.obsm["X_umap"][:, 0].tolist()
    ys = adata.obsm["X_umap"][:, 1].tolist()
    clusters = adata.obs["leiden"].tolist()

    points = [
        {"x": x, "y": y, "cluster": c}
        for x, y, c in zip(xs, ys, clusters)
    ]
    n_clusters = int(adata.obs["leiden"].nunique())

    return {"points": points, "n_clusters": n_clusters}


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
    labels = adata.obs["leiden"].to_numpy()  # Categorical → plain numpy array

    # ── Cache check ──────────────────────────────────────────────────────────
    label_hash = hash(labels.tobytes())
    cached = getattr(app.state, "shap_cache", None)
    if cached is not None and cached["hash"] == label_hash:
        return cached["result"]

    # ── Feature matrix ───────────────────────────────────────────────────────
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

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
        n_jobs=-1,
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


# Serve the compiled React app from the dist/ folder
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
