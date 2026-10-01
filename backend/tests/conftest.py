"""
Shared fixtures for the SCannotate backend tests.

The tests never enter the FastAPI lifespan, so nothing is downloaded
(no PBMC 3k, no PanglaoDB). Instead each test gets a small synthetic
dataset with three known cell groups, preprocessed once per session,
and a hand-built marker database that maps onto those groups.
"""
import numpy as np
import pandas as pd
import anndata
import pytest
from fastapi.testclient import TestClient

import main

N_PER_GROUP = 100
N_GENES = 2000
N_MARKERS = 50
GROUPS = ["A", "B", "C"]


def marker_genes(group: str) -> list[str]:
    return [f"MK{group}{i}" for i in range(N_MARKERS)]


def synthetic_counts(seed: int = 0) -> anndata.AnnData:
    """
    Raw count matrix (cells x genes) with three groups of N_PER_GROUP cells.
    Each group over-expresses its own block of marker genes (Poisson mean 6
    vs. 1 background), which is strong enough for Leiden to recover the
    groups exactly while keeping the markers inside the HVG mean window
    used by main.preprocess.
    """
    rng = np.random.default_rng(seed)
    truth = np.repeat(GROUPS, N_PER_GROUP)
    mu = np.ones((len(GROUPS), N_GENES))
    for g in range(len(GROUPS)):
        mu[g, g * N_MARKERS:(g + 1) * N_MARKERS] = 6.0
    group_idx = np.repeat(np.arange(len(GROUPS)), N_PER_GROUP)
    X = rng.poisson(mu[group_idx]).astype(np.float32)

    gene_names = [g for grp in GROUPS for g in marker_genes(grp)]
    gene_names += [f"BG{i}" for i in range(N_GENES - len(gene_names))]
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"truth": truth}, index=[f"cell-{i}" for i in range(len(truth))]),
        var=pd.DataFrame(index=gene_names),
    )


@pytest.fixture(scope="session")
def raw_adata() -> anndata.AnnData:
    return synthetic_counts()


@pytest.fixture(scope="session")
def preprocessed(raw_adata) -> anndata.AnnData:
    return main.preprocess(raw_adata.copy())


@pytest.fixture
def client(preprocessed):
    """
    TestClient with fresh app.state for every test. Not used as a context
    manager on purpose: that would run the lifespan and hit the network.
    """
    state = main.app.state
    state.adata = preprocessed.copy()
    state.dataset_name = "synthetic"
    state.annotations = {}
    state.shap_cache = None
    state.marker_db = {f"Type {grp}": set(marker_genes(grp)[:20]) for grp in GROUPS}
    return TestClient(main.app)


@pytest.fixture
def clustered_client(client):
    resp = client.post("/cluster", json={"algorithm": "leiden", "resolution": 0.5})
    assert resp.status_code == 200
    return client


def majority_truth(cluster_id: str) -> str:
    """The ground-truth group most cells in a cluster belong to."""
    obs = main.app.state.adata.obs
    return obs.loc[obs["labels"] == cluster_id, "truth"].value_counts().idxmax()
