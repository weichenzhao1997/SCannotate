from contextlib import asynccontextmanager
import gzip
import io
import urllib.request

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import igraph  # verify python-igraph is present (required by leiden igraph flavor)
import numpy as np
from hdbscan import HDBSCAN
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder
import scanpy as sc


PANGLAO_URL = (
    "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
)



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
            core_dist_n_jobs=-1,
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


# Serve the compiled React app from the dist/ folder
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
