"""Unit tests for preprocessing, input parsing helpers, and file readers."""
import gzip
import zipfile

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse

import main
from conftest import N_PER_GROUP, GROUPS, marker_genes

BARCODES = [f"AAACATACAACC{i:02d}-1" for i in range(20)]
GENES = ["CD3D", "MS4A1", "LYZ", "NKG7", "GNLY", "CD14", "PPBP", "FCGR3A",
         "HLA-DRA", "CST3", "S100A8", "IL7R", "CCR7", "CD8A", "FCER1A",
         "ENSG00000163736", "ENSG00000105374", "ENSG00000101439", "ACTB", "MALAT1"]


# ── preprocess ───────────────────────────────────────────────────────────────

def test_preprocess_produces_pca_and_neighbors(preprocessed):
    assert preprocessed.n_obs == N_PER_GROUP * len(GROUPS)
    assert preprocessed.var["highly_variable"].all()
    assert "X_pca" in preprocessed.obsm
    assert "connectivities" in preprocessed.obsp


def test_preprocess_keeps_marker_genes_as_hvgs(preprocessed):
    # If this fails, the synthetic data no longer exercises the HVG window
    # in preprocess() and the downstream biology assertions are meaningless.
    for grp in GROUPS:
        kept = set(marker_genes(grp)) & set(preprocessed.var_names)
        assert len(kept) >= 10, f"only {len(kept)} markers of group {grp} survived HVG selection"


def test_preprocess_filters_low_quality_cells(raw_adata):
    adata = raw_adata.copy()
    adata.X[:5, 150:] = 0  # these five cells now express < 200 genes
    out = main.preprocess(adata)
    assert out.n_obs == raw_adata.n_obs - 5
    assert not set(raw_adata.obs_names[:5]) & set(out.obs_names)


# ── tabular orientation heuristics ───────────────────────────────────────────

def test_cells_by_genes_is_not_transposed():
    df = pd.DataFrame(np.zeros((20, 20)), index=BARCODES, columns=GENES)
    assert main._should_transpose_tabular(df) is False


def test_genes_by_cells_is_transposed():
    df = pd.DataFrame(np.zeros((20, 20)), index=GENES, columns=BARCODES)
    assert main._should_transpose_tabular(df) is True


def test_ambiguous_layout_defaults_to_cells_by_genes():
    labels = [str(i) for i in range(20)]
    df = pd.DataFrame(np.zeros((20, 20)), index=labels, columns=labels)
    assert main._should_transpose_tabular(df) is False


def test_cluster_sort_key_orders_numerically_then_by_name():
    ids = ["10", "2", "-1", "abc", "0"]
    assert sorted(ids, key=main._cluster_sort_key) == ["-1", "0", "2", "10", "abc"]


# ── _read_uploaded ───────────────────────────────────────────────────────────

def _counts_frame(raw_adata):
    return pd.DataFrame(raw_adata.X, index=raw_adata.obs_names, columns=raw_adata.var_names)


def test_read_csv_cells_by_genes(tmp_path, raw_adata):
    path = tmp_path / "counts.csv"
    _counts_frame(raw_adata).to_csv(path)
    adata = main._read_uploaded(str(path), "counts.csv")
    assert adata.shape == raw_adata.shape
    assert list(adata.var_names[:3]) == list(raw_adata.var_names[:3])


def test_read_tsv_genes_by_cells_is_transposed(tmp_path, raw_adata):
    path = tmp_path / "counts.tsv"
    _counts_frame(raw_adata).T.to_csv(path, sep="\t")
    adata = main._read_uploaded(str(path), "counts.tsv")
    assert adata.shape == raw_adata.shape
    assert adata.obs_names[0] == raw_adata.obs_names[0]


def test_read_h5ad_roundtrip(tmp_path, raw_adata):
    path = tmp_path / "data.h5ad"
    raw_adata.write_h5ad(path)
    adata = main._read_uploaded(str(path), "data.h5ad")
    assert adata.shape == raw_adata.shape
    np.testing.assert_array_equal(adata.X, raw_adata.X)


def write_10x_zip(path, raw_adata):
    """Write a Cell Ranger v3-style matrix folder into a zip archive."""
    folder = "filtered_feature_bc_matrix"
    mtx = path.parent / "matrix.mtx"
    # 10x stores genes x cells
    scipy.io.mmwrite(str(mtx), scipy.sparse.csr_matrix(raw_adata.X.T.astype(np.int32)))
    features = "".join(f"ID{i}\t{g}\tGene Expression\n" for i, g in enumerate(raw_adata.var_names))
    barcodes = "".join(f"{b}\n" for b in raw_adata.obs_names)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{folder}/matrix.mtx.gz", gzip.compress(mtx.read_bytes()))
        zf.writestr(f"{folder}/features.tsv.gz", gzip.compress(features.encode()))
        zf.writestr(f"{folder}/barcodes.tsv.gz", gzip.compress(barcodes.encode()))


def test_read_10x_zip(tmp_path, raw_adata):
    path = tmp_path / "tenx.zip"
    write_10x_zip(path, raw_adata)
    adata = main._read_uploaded(str(path), "tenx.zip")
    assert adata.shape == raw_adata.shape
    assert list(adata.var_names[:3]) == list(raw_adata.var_names[:3])


def test_read_zip_without_mtx_raises(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(ValueError, match="No .mtx file"):
        main._read_uploaded(str(path), "empty.zip")


def test_read_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not really excel")
    with pytest.raises(ValueError, match="Unsupported format"):
        main._read_uploaded(str(path), "data.xlsx")
