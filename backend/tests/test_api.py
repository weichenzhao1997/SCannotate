"""Endpoint tests against the FastAPI app using a synthetic dataset."""
import csv
import io

import anndata
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import main
from conftest import GROUPS, N_PER_GROUP, marker_genes, majority_truth
from test_data_io import write_10x_zip

N_CELLS = N_PER_GROUP * len(GROUPS)
HDBSCAN_WITH_NOISE = {"algorithm": "hdbscan", "min_cluster_size": 20, "min_samples": 1}


# ── /cluster ─────────────────────────────────────────────────────────────────

def test_leiden_recovers_known_groups(client):
    body = client.post("/cluster", json={"algorithm": "leiden", "resolution": 0.5}).json()

    assert len(body["points"]) == N_CELLS
    assert body["n_noise"] == 0
    assert body["n_clusters"] == len(GROUPS)
    obs = main.app.state.adata.obs
    assert adjusted_rand_score(obs["truth"], obs["labels"]) > 0.95


def test_higher_resolution_gives_at_least_as_many_clusters(client):
    low = client.post("/cluster", json={"resolution": 0.1}).json()["n_clusters"]
    high = client.post("/cluster", json={"resolution": 2.0}).json()["n_clusters"]
    assert high >= low


def test_cluster_points_have_coordinates_and_labels(client):
    points = client.post("/cluster", json={}).json()["points"]
    assert set(points[0]) == {"x", "y", "cluster"}
    assert all(np.isfinite([p["x"] for p in points]))
    assert all(isinstance(p["cluster"], str) for p in points)


def test_hdbscan_reports_noise_consistently(client):
    body = client.post("/cluster", json=HDBSCAN_WITH_NOISE).json()

    labels = [p["cluster"] for p in body["points"]]
    assert body["n_noise"] == labels.count("-1")
    assert body["n_noise"] > 0
    assert body["n_clusters"] == len(set(labels) - {"-1"})


# ── /shap ────────────────────────────────────────────────────────────────────

def test_shap_returns_top10_sorted_per_cluster(clustered_client):
    clusters = clustered_client.post("/shap").json()["clusters"]

    labels = set(main.app.state.adata.obs["labels"])
    assert set(clusters) == labels
    for genes in clusters.values():
        assert len(genes) == 10
        scores = [g["shap"] for g in genes]
        assert scores == sorted(scores, reverse=True)


def test_shap_driver_genes_are_markers(clustered_client):
    clusters = clustered_client.post("/shap").json()["clusters"]

    all_markers = {g for grp in GROUPS for g in marker_genes(grp)}
    for cluster_id, genes in clusters.items():
        top = [g["gene"] for g in genes]
        own = set(marker_genes(majority_truth(cluster_id)))
        assert set(top) <= all_markers, f"cluster {cluster_id} driven by background genes: {top}"
        assert own & set(top), f"cluster {cluster_id} has none of its own markers: {top}"


def test_shap_excludes_noise_cluster(client):
    client.post("/cluster", json=HDBSCAN_WITH_NOISE)
    clusters = client.post("/shap").json()["clusters"]
    assert "-1" not in clusters
    assert clusters  # the non-noise clusters are still explained


def test_shap_handles_exactly_two_clusters(clustered_client):
    # Two classes make LightGBM train a binary model, which changes the
    # shape of the SHAP output; this used to crash with an IndexError.
    adata = main.app.state.adata
    adata.obs["labels"] = np.where(adata.obs["truth"] == "A", "0", "1")

    resp = clustered_client.post("/shap")

    assert resp.status_code == 200
    clusters = resp.json()["clusters"]
    assert set(clusters) == {"0", "1"}
    assert set(marker_genes("A")) & {g["gene"] for g in clusters["0"]}


@pytest.fixture
def fit_counter(monkeypatch):
    """Counts LightGBM fits so tests can tell cache hits from retraining."""
    calls = {"n": 0}
    real_fit = main.lgb.LGBMClassifier.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(main.lgb.LGBMClassifier, "fit", counting_fit)
    return calls


def test_shap_cache_hit_for_unchanged_labels(clustered_client, fit_counter):
    first = clustered_client.post("/shap").json()
    second = clustered_client.post("/shap").json()
    assert first == second
    assert fit_counter["n"] == 1


def test_shap_cache_hit_when_labels_are_rebuilt_with_same_content(clustered_client, fit_counter):
    adata = main.app.state.adata
    # Multi-character labels, rebuilt as fresh string objects each time
    adata.obs["labels"] = [f"c{l}0" for l in adata.obs["leiden"]]
    clustered_client.post("/shap")
    adata.obs["labels"] = [f"c{l}0" for l in adata.obs["leiden"]]
    clustered_client.post("/shap")
    assert fit_counter["n"] == 1


def test_shap_cache_invalidated_when_labels_change(clustered_client, fit_counter):
    clustered_client.post("/shap")
    clustered_client.post("/cluster", json={"resolution": 2.0})
    clustered_client.post("/shap")
    assert fit_counter["n"] == 2


# ── /annotate ────────────────────────────────────────────────────────────────

def test_annotate_suggests_matching_cell_type(clustered_client):
    for cluster_id in set(main.app.state.adata.obs["labels"]):
        body = clustered_client.post("/annotate", json={"cluster_id": cluster_id}).json()
        assert body["cluster_id"] == cluster_id
        assert 0 < len(body["suggestions"]) <= 5
        assert body["suggestions"][0]["cell_type"] == f"Type {majority_truth(cluster_id)}"


def test_annotate_unknown_cluster_returns_no_suggestions(clustered_client):
    body = clustered_client.post("/annotate", json={"cluster_id": "999"}).json()
    assert body["suggestions"] == []


def test_annotate_without_marker_db_returns_no_suggestions(clustered_client):
    main.app.state.marker_db = {}
    body = clustered_client.post("/annotate", json={"cluster_id": "0"}).json()
    assert body["suggestions"] == []


# ── /annotations ─────────────────────────────────────────────────────────────

def test_save_and_list_annotations(client):
    assert client.get("/annotations").json() == {"annotations": {}}

    client.post("/annotations/save", json={"cluster_id": "0", "label": "T cell", "status": "confirmed"})
    client.post("/annotations/save", json={"cluster_id": "0", "label": "CD8 T cell", "status": "confirmed"})
    client.post("/annotations/save", json={"cluster_id": "1", "label": "B cell", "status": "suggested"})

    assert client.get("/annotations").json()["annotations"] == {
        "0": {"label": "CD8 T cell", "status": "confirmed"},
        "1": {"label": "B cell", "status": "suggested"},
    }


def test_save_annotation_rejects_missing_fields(client):
    resp = client.post("/annotations/save", json={"cluster_id": "0"})
    assert resp.status_code == 422


# ── /upload-dataset, /load-dataset, /dataset-info ────────────────────────────

def _csv_bytes(adata) -> bytes:
    df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    return df.to_csv().encode()


def test_upload_csv_replaces_dataset_and_resets_state(client, raw_adata):
    main.app.state.annotations = {"0": {"label": "old", "status": "confirmed"}}
    main.app.state.shap_cache = {"hash": 0, "result": {}}

    resp = client.post("/upload-dataset", files={"file": ("mine.csv", _csv_bytes(raw_adata), "text/csv")})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset"] == "mine.csv"
    assert body["n_cells"] == N_CELLS
    assert client.get("/dataset-info").json() == {k: body[k] for k in ("dataset", "n_cells", "n_genes")}
    assert main.app.state.annotations == {}
    assert main.app.state.shap_cache is None
    # The new dataset can be clustered straight away
    assert client.post("/cluster", json={}).status_code == 200


def test_upload_10x_zip(client, raw_adata, tmp_path):
    path = tmp_path / "tenx.zip"
    write_10x_zip(path, raw_adata)
    resp = client.post("/upload-dataset", files={"file": ("tenx.zip", path.read_bytes(), "application/zip")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_cells"] == N_CELLS


def test_upload_rejects_unsupported_extension(client):
    resp = client.post("/upload-dataset", files={"file": ("data.xlsx", b"x", "application/octet-stream")})
    assert resp.status_code == 400
    assert main.app.state.dataset_name == "synthetic"


def test_upload_rejects_too_few_cells(client, raw_adata):
    resp = client.post("/upload-dataset", files={"file": ("small.csv", _csv_bytes(raw_adata[:10]), "text/csv")})
    assert resp.status_code == 422
    assert "at least 50 cells" in resp.json()["detail"]


def test_upload_rejects_too_few_genes(client, raw_adata):
    resp = client.post("/upload-dataset", files={"file": ("narrow.csv", _csv_bytes(raw_adata[:, :50]), "text/csv")})
    assert resp.status_code == 422
    assert "at least 100 genes" in resp.json()["detail"]


def test_upload_rejects_corrupt_file(client):
    resp = client.post("/upload-dataset", files={"file": ("broken.h5ad", b"not hdf5", "application/octet-stream")})
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("Could not read file")
    assert main.app.state.dataset_name == "synthetic"


def test_load_unknown_builtin_dataset(client):
    resp = client.post("/load-dataset", json={"dataset": "not-a-dataset"})
    assert resp.status_code == 400


# ── /export/* ────────────────────────────────────────────────────────────────

def _read_csv(resp) -> list[dict]:
    return list(csv.DictReader(io.StringIO(resp.text)))


def test_export_annotations_csv(clustered_client):
    clustered_client.post("/annotations/save", json={"cluster_id": "0", "label": "T cell", "status": "confirmed"})

    resp = clustered_client.get("/export/annotations-csv")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    rows = _read_csv(resp)
    assert [r["cluster_id"] for r in rows] == sorted({r["cluster_id"] for r in rows}, key=int)
    assert sum(int(r["n_cells"]) for r in rows) == N_CELLS
    by_id = {r["cluster_id"]: r for r in rows}
    assert by_id["0"]["label"] == "T cell" and by_id["0"]["status"] == "confirmed"
    assert by_id["1"]["status"] == "unannotated"


def test_export_umap_csv(clustered_client):
    clustered_client.post("/annotations/save", json={"cluster_id": "1", "label": "B cell", "status": "suggested"})

    rows = _read_csv(clustered_client.get("/export/umap-csv"))

    assert len(rows) == N_CELLS
    assert set(rows[0]) == {"cell_barcode", "umap_1", "umap_2", "cluster_id", "cell_type", "annotation_status"}
    labelled = [r for r in rows if r["cluster_id"] == "1"]
    assert labelled and all(r["cell_type"] == "B cell" for r in labelled)


def test_export_full_h5ad_carries_annotations(clustered_client, tmp_path):
    clustered_client.post("/annotations/save", json={"cluster_id": "0", "label": "T cell", "status": "confirmed"})

    resp = clustered_client.get("/export/full-h5ad")

    assert resp.status_code == 200
    path = tmp_path / "out.h5ad"
    path.write_bytes(resp.content)
    exported = anndata.read_h5ad(path)
    assert exported.n_obs == N_CELLS
    in_zero = exported.obs["labels"] == "0"
    assert (exported.obs.loc[in_zero, "cell_type"] == "T cell").all()
    assert (exported.obs.loc[~in_zero, "cell_type"] == "unannotated").all()
    assert "X_umap" in exported.obsm
