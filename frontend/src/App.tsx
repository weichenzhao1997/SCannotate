import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

// ── Types ──────────────────────────────────────────────────────────────────

interface UmapPoint { x: number; y: number; cluster: string; }
interface ShapGene   { gene: string; shap: number; }
interface Suggestion { cell_type: string; score: number; }
type ShapData    = Record<string, ShapGene[]>;
type Annotations = Record<string, { label: string; status: string }>;

// ── Constants ──────────────────────────────────────────────────────────────

const PLOTLY_COLORS = [
  '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
];

type Algorithm = 'leiden' | 'hdbscan';

const ALGO_DEFS: { value: string; label: string }[] = [
  { value: 'leiden',  label: 'Leiden'  },
  { value: 'hdbscan', label: 'HDBSCAN' },
];

const ALGO_ACTIVE_COLOR: Record<string, string> = {
  leiden:  '#7F77DD',
  hdbscan: '#BA7517',
};

const ALGO_HINT: Record<string, string> = {
  leiden:  'Leiden offers better community detection & is generally faster',
  hdbscan: 'HDBSCAN finds clusters of arbitrary shape without a fixed resolution',
};

// ── Shared styles ──────────────────────────────────────────────────────────

const sectionLabel: React.CSSProperties = {
  display: 'block',
  fontSize: '11px',
  textTransform: 'uppercase',
  letterSpacing: '0.04em',
  color: '#888',
  marginBottom: '6px',
};

const hintBox: React.CSSProperties = {
  marginTop: '6px',
  background: '#E1F5EE',
  border: '1px solid #5DCAA5',
  color: '#085041',
  borderRadius: '6px',
  padding: '6px 8px',
  fontSize: '11px',
  lineHeight: '1.5',
};

// ── Onboarding data ────────────────────────────────────────────────────────

const OB_HEADERS = [
  { bg: '#E1F5EE', symbol: '🧬', subtitle: 'Single-Cell RNA Sequencing',        subtitleColor: '#085041' },
  { bg: '#EEEDFE', symbol: '🔬', subtitle: 'Applications in Biomedical Research', subtitleColor: '#3C3489' },
  { bg: '#FAEEDA', symbol: '⚙️', subtitle: 'From Raw Counts to Biological Insight', subtitleColor: '#633806' },
  { bg: '#EEEDFE', symbol: '🗺️', subtitle: 'Leiden & HDBSCAN',                  subtitleColor: '#3C3489' },
  { bg: '#E6F1FB', symbol: '🏷️', subtitle: 'Annotating Cell Identities',         subtitleColor: '#0C447C' },
  { bg: '#EAF3DE', symbol: '📊', subtitle: 'LightGBM + SHAP Interpretability',   subtitleColor: '#27500A' },
];

const OB_TITLES = [
  'Understanding Single-Cell RNA Sequencing',
  'What is scRNA-seq Used For?',
  'The scRNA-seq Analysis Pipeline',
  'Clustering: Leiden and HDBSCAN',
  'Annotating Clusters with Cell Identities',
  'Understanding Driver Genes',
];

const obP: React.CSSProperties = {
  fontSize: '14px', color: '#444', lineHeight: '1.75', marginBottom: '12px', marginTop: 0,
};
const obTermPill: React.CSSProperties = {
  background: '#EEEDFE', color: '#3C3489', borderRadius: '4px',
  padding: '2px 7px', fontSize: '12px', fontWeight: 500,
  marginRight: '8px', whiteSpace: 'nowrap', display: 'inline-block',
};
const obTermRow: React.CSSProperties = {
  marginBottom: '8px', fontSize: '14px', color: '#444', lineHeight: '1.6',
};

function renderOnboardingBody(step: number): React.ReactNode {
  if (step === 0) return (
    <>
      <p style={obP}>
        Traditional RNA sequencing measures the average gene expression of an entire tissue
        sample — blending together whatever mix of cell types is present. Single-cell RNA
        sequencing (scRNA-seq) resolves this at the resolution of individual cells, producing
        one gene expression profile per cell.
      </p>
      <p style={obP}>
        A typical scRNA-seq experiment generates a count matrix: rows are cells, columns are
        genes, and each entry records how many RNA molecules of that gene were detected in that
        cell. The result is a high-dimensional, sparse dataset — most entries are zero, a
        property called dropout.
      </p>
      <div>
        {[
          { term: 'Count matrix', def: 'A cells × genes table of RNA molecule counts' },
          { term: 'Dropout',      def: 'Zero counts caused by technical capture limits, not true absence of expression' },
          { term: 'Transcriptome', def: 'The full set of RNA molecules expressed in a cell at a given moment' },
        ].map(({ term, def }) => (
          <div key={term} style={obTermRow}><span style={obTermPill}>{term}</span>{def}</div>
        ))}
      </div>
    </>
  );

  if (step === 1) return (
    <>
      <p style={obP}>
        scRNA-seq has become a foundational tool across biomedical research and industrial drug
        discovery. By profiling thousands of cells simultaneously, researchers can ask questions
        that bulk sequencing simply cannot answer.
      </p>
      <div style={{ marginBottom: '12px' }}>
        {[
          { term: 'Cell type discovery',       def: 'Identifying distinct populations in a tissue, including rare types invisible in bulk measurements' },
          { term: 'Disease characterization',  def: 'Comparing healthy and diseased tissue at single-cell resolution to find which populations change' },
          { term: 'Developmental biology',     def: 'Reconstructing how stem cells differentiate into specialized types over time' },
          { term: 'Drug response profiling',   def: 'Measuring how individual cells within a tumor respond to therapy, identifying resistant subpopulations' },
        ].map(({ term, def }) => (
          <div key={term} style={obTermRow}><span style={obTermPill}>{term}</span>{def}</div>
        ))}
      </div>
      <p style={obP}>
        The technology is now routine even in labs without dedicated bioinformaticians — which
        is exactly the gap Scannotate is designed to address.
      </p>
    </>
  );

  if (step === 2) {
    const pipeline = [
      { name: 'Quality control',             desc: 'Filter out low-quality cells and rarely expressed genes',          badge: 'auto' },
      { name: 'Normalization & log-transform', desc: 'Correct for sequencing depth differences between cells',          badge: 'auto' },
      { name: 'Feature selection',           desc: 'Select the most variable genes to reduce noise',                   badge: 'auto' },
      { name: 'PCA + KNN graph',             desc: 'Compress the data and build a cell similarity graph',              badge: 'auto' },
      { name: 'Clustering',                  desc: 'Partition cells into groups using Leiden or HDBSCAN',              badge: 'interactive' },
      { name: 'Annotation',                  desc: 'Assign biological cell type identities to each cluster',           badge: 'interactive' },
    ];
    return (
      <>
        <p style={obP}>
          Raw count matrices go through a standard series of computational steps before
          biological interpretation is possible. Scannotate handles the first four steps
          automatically when you load a dataset.
        </p>
        <div>
          {pipeline.map((s, i) => (
            <React.Fragment key={i}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                <div style={{
                  flexShrink: 0, width: 20, height: 20, borderRadius: '50%',
                  background: '#FAEEDA', color: '#633806',
                  fontSize: '11px', fontWeight: 600,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>{i + 1}</div>
                <div>
                  <span style={{ fontWeight: 500, fontSize: '13px', color: '#222' }}>{s.name}</span>
                  {s.badge === 'auto'
                    ? <span style={{ background: '#E1F5EE', color: '#085041', borderRadius: '20px', fontSize: '10px', padding: '1px 6px', marginLeft: '6px' }}>auto</span>
                    : <span style={{ background: '#EEEDFE', color: '#3C3489', borderRadius: '20px', fontSize: '10px', padding: '1px 6px', marginLeft: '6px' }}>interactive</span>
                  }
                  <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>{s.desc}</div>
                </div>
              </div>
              {i < pipeline.length - 1 && (
                <div style={{ height: '16px', width: '1px', background: '#e0e0e0', marginLeft: '9px', marginTop: '2px', marginBottom: '2px' }} />
              )}
            </React.Fragment>
          ))}
        </div>
      </>
    );
  }

  if (step === 3) return (
    <>
      <p style={obP}>
        Clustering is the step where cells are grouped by transcriptional similarity. Scannotate
        offers two fundamentally different approaches, and choosing between them is a biological
        judgment call — not something an algorithm can decide for you.
      </p>
      <div style={{ marginBottom: '12px' }}>
        {[
          { term: 'Leiden',  def: 'A graph-based algorithm that partitions a cell similarity network into communities. Controlled by a resolution parameter: higher values produce more, finer clusters. Every cell is assigned to a cluster.' },
          { term: 'HDBSCAN', def: 'A density-based algorithm that finds clusters as regions of high cell density. Controlled by minimum cluster size. Cells in sparse regions are labeled as noise rather than forced into a cluster.' },
        ].map(({ term, def }) => (
          <div key={term} style={obTermRow}><span style={obTermPill}>{term}</span>{def}</div>
        ))}
      </div>
      <p style={obP}>
        There is no universally correct answer — the right granularity depends on your biological
        question. Use the resolution slider (Leiden) or minimum cluster size slider (HDBSCAN) to
        explore different partitions, and look for a configuration where the clusters match your
        biological expectations.
      </p>
      <div style={{ ...hintBox, marginTop: '4px' }}>
        The UMAP plot updates in real time as you adjust parameters. Try different values and
        observe how the cluster boundaries shift.
      </div>
    </>
  );

  if (step === 4) return (
    <>
      <p style={obP}>
        Once clusters are defined, each one must be assigned a biological cell type label. This
        is the most knowledge-intensive step in the pipeline — automated tools can suggest
        candidates, but the final call requires domain expertise.
      </p>
      <p style={{ ...obP, marginBottom: '8px' }}>
        In Scannotate, click any cluster on the UMAP to select it. The right panel will show:
      </p>
      <div style={{ marginBottom: '12px' }}>
        {[
          'Automated suggestions from the PanglaoDB reference atlas, ranked by marker gene overlap',
          'A confidence score for each suggestion',
          'Accept, override, or type a custom label',
        ].map((item, i) => (
          <div key={i} style={{ fontSize: '13px', lineHeight: 2, color: '#444' }}>
            <span style={{ color: '#aaa', marginRight: '8px' }}>›</span>{item}
          </div>
        ))}
      </div>
      <p style={obP}>
        Every annotation is tracked with a status — Confirmed (you validated it), Suggested
        (awaiting your review), or Unannotated. The annotation list in the right panel shows
        the status of all clusters at a glance.
      </p>
      <div style={{
        marginTop: '4px', background: '#FAEEDA', border: '1px solid #FAC775',
        color: '#633806', borderRadius: '6px', padding: '6px 8px',
        fontSize: '11px', lineHeight: '1.5',
      }}>
        Automated suggestions are hypotheses, not ground truth. Always cross-reference with
        known marker genes before confirming an annotation.
      </div>
    </>
  );

  // step 5
  return (
    <>
      <p style={obP}>
        The driver gene panel gives you a second layer of evidence for annotation decisions,
        beyond the statistical marker genes. Click 'Compute driver genes' to train a LightGBM
        classifier on the current cluster assignments and compute SHAP values.
      </p>
      <div style={{ marginBottom: '12px' }}>
        {[
          { term: 'Marker genes',  def: 'Genes with significantly higher expression in one cluster versus all others, identified by a statistical test (Wilcoxon rank-sum)' },
          { term: 'Driver genes',  def: 'Genes that most strongly help a machine learning classifier distinguish one cluster from all others simultaneously, accounting for the full gene expression profile' },
          { term: 'SHAP value',    def: "A per-gene score reflecting how much that gene contributed to the classifier's decision for a given cluster" },
        ].map(({ term, def }) => (
          <div key={term} style={obTermRow}><span style={obTermPill}>{term}</span>{def}</div>
        ))}
      </div>
      <p style={obP}>
        When marker genes and driver genes agree, annotation confidence is high. When they
        disagree, it is a signal worth investigating — the cluster may be biologically ambiguous
        or poorly separated from a neighbor.
      </p>
      <div style={{ ...hintBox, marginTop: '4px' }}>
        Driver gene analysis retrains the model on the current clustering. Re-run it after
        changing resolution or switching algorithms to keep results current.
      </div>
      <p style={{ fontSize: '15px', color: '#3C3489', textAlign: 'center', marginTop: '16px', marginBottom: '4px' }}>
        You're ready to start. Load a dataset, explore the clustering, and annotate your cells. 🎉
      </p>
      <p style={{ fontSize: '12px', color: '#aaa', textAlign: 'center', margin: 0 }}>
        You can reopen this guide any time using the ? button in the top bar.
      </p>
    </>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────

function buildTracesWithColors(points: UmapPoint[]): {
  traces: object[];
  colorMap: Record<string, string>;
} {
  const byCluster = new Map<string, { x: number[]; y: number[] }>();
  for (const p of points) {
    if (!byCluster.has(p.cluster)) byCluster.set(p.cluster, { x: [], y: [] });
    byCluster.get(p.cluster)!.x.push(p.x);
    byCluster.get(p.cluster)!.y.push(p.y);
  }
  const realIds = [...byCluster.keys()]
    .filter(id => id !== '-1')
    .sort((a, b) => parseInt(a) - parseInt(b));

  const colorMap: Record<string, string> = {};
  const traces: object[] = realIds.map((id, idx) => {
    const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
    colorMap[id] = color;
    return {
      x: byCluster.get(id)!.x,
      y: byCluster.get(id)!.y,
      mode: 'markers',
      type: 'scatter',
      name: `Cluster ${id}`,
      marker: { size: 3, opacity: 0.7, color },
    };
  });

  if (byCluster.has('-1')) {
    colorMap['-1'] = '#aaaaaa';
    traces.push({
      x: byCluster.get('-1')!.x,
      y: byCluster.get('-1')!.y,
      mode: 'markers',
      type: 'scatter',
      name: 'Noise',
      marker: { size: 3, opacity: 0.4, color: '#aaaaaa' } as object,
    });
  }
  return { traces, colorMap };
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, React.CSSProperties> = {
    confirmed:   { background: '#EAF3DE', color: '#27500A', border: '1px solid #C0DD97' },
    suggested:   { background: '#FAEEDA', color: '#633806', border: '1px solid #FAC775' },
    unannotated: { background: '#F1EFE8', color: '#5F5E5A', border: '1px solid #D3D1C7' },
  };
  const s = styles[status] ?? styles.unannotated;
  const text = status === 'confirmed' ? 'confirmed' : status === 'suggested' ? 'suggested' : '—';
  return (
    <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '20px', flexShrink: 0, ...s }}>
      {text}
    </span>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

const App: React.FC = () => {
  // Clustering params
  const [resolution, setResolution]         = useState<number>(0.5);
  const [algorithm, setAlgorithm]           = useState<Algorithm>('leiden');
  const [minClusterSize, setMinClusterSize] = useState<number>(50);
  const [minSamples, setMinSamples]         = useState<number>(5);

  // Plot data
  const [traces, setTraces]                     = useState<object[]>([]);
  const [clusterColorMap, setClusterColorMap]   = useState<Record<string, string>>({});
  const [activeClusterIds, setActiveClusterIds] = useState<Set<string>>(new Set());
  const [nClusters, setNClusters]               = useState<number | null>(null);
  const [nNoise, setNNoise]                     = useState<number>(0);
  const [loading, setLoading]                   = useState<boolean>(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Annotation state
  const [annotations, setAnnotations]               = useState<Annotations>({});
  const [selectedCluster, setSelectedCluster]       = useState<string | null>(null);
  const [suggestions, setSuggestions]               = useState<Suggestion[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState<boolean>(false);
  const [annotationInput, setAnnotationInput]       = useState<string>('');

  // SHAP / driver genes state
  const [shapData, setShapData]       = useState<ShapData | null>(null);
  const [shapLoading, setShapLoading] = useState<boolean>(false);
  const [shapStale, setShapStale]     = useState<boolean>(false);

  // Dataset / import state
  const [datasetName, setDatasetName]         = useState<string>('PBMC 3k (built-in)');
  const [datasetInfo, setDatasetInfo]         = useState<{ n_cells: number; n_genes: number } | null>(null);
  const [showImportModal, setShowImportModal] = useState<boolean>(false);
  const [importLoading, setImportLoading]     = useState<boolean>(false);
  const [importError, setImportError]         = useState<string | null>(null);
  const [selectedBuiltin, setSelectedBuiltin] = useState<string>('pbmc3k');
  const [importTab, setImportTab]             = useState<'builtin' | 'upload'>('builtin');
  const [uploadFile, setUploadFile]           = useState<File | null>(null);
  const [dragOver, setDragOver]               = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Onboarding state
  const [showOnboarding, setShowOnboarding]   = useState<boolean>(false);
  const [onboardingStep, setOnboardingStep]   = useState<number>(0);
  const [onboardingDir, setOnboardingDir]     = useState<'forward' | 'backward'>('forward');
  const [helpBtnHover, setHelpBtnHover]       = useState<boolean>(false);

  // ── Helpers ────────────────────────────────────────────────────────────

  const completeOnboarding = () => {
    localStorage.setItem('scannotate_onboarding_seen', 'true');
    setShowOnboarding(false);
  };

  const goToStep = (step: number, dir: 'forward' | 'backward') => {
    setOnboardingDir(dir);
    setOnboardingStep(step);
  };

  // ── Effects ────────────────────────────────────────────────────────────

  // Mount: data fetches + onboarding check
  useEffect(() => {
    fetchCluster(resolution, algorithm, minClusterSize, minSamples);
    fetch('http://localhost:8000/annotations')
      .then(r => r.json())
      .then(d => setAnnotations(d.annotations ?? {}))
      .catch(() => {});
    fetch('http://localhost:8000/dataset-info')
      .then(r => r.json())
      .then(d => {
        setDatasetName(d.dataset === 'pbmc3k' ? 'PBMC 3k (built-in)' : d.dataset);
        setDatasetInfo({ n_cells: d.n_cells, n_genes: d.n_genes });
      })
      .catch(() => {});
    if (localStorage.getItem('scannotate_onboarding_seen') !== 'true') {
      setShowOnboarding(true);
    }
  }, []);

  // Keyboard nav for onboarding
  useEffect(() => {
    if (!showOnboarding) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (onboardingStep < 5) {
          setOnboardingDir('forward');
          setOnboardingStep(onboardingStep + 1);
        } else {
          localStorage.setItem('scannotate_onboarding_seen', 'true');
          setShowOnboarding(false);
        }
      } else if (e.key === 'ArrowLeft') {
        if (onboardingStep > 0) {
          setOnboardingDir('backward');
          setOnboardingStep(onboardingStep - 1);
        }
      } else if (e.key === 'Escape') {
        localStorage.setItem('scannotate_onboarding_seen', 'true');
        setShowOnboarding(false);
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [showOnboarding, onboardingStep]);

  // ── Data fetching ──────────────────────────────────────────────────────

  const fetchCluster = async (
    res: number, alg: Algorithm, mcs: number, ms: number,
  ) => {
    setLoading(true);
    setSelectedCluster(null);
    setSuggestions([]);
    try {
      const response = await fetch('http://localhost:8000/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resolution: res, algorithm: alg,
          min_cluster_size: mcs, min_samples: ms,
        }),
      });
      const result = await response.json();
      const { traces: newTraces, colorMap } = buildTracesWithColors(result.points);
      setTraces(newTraces);
      setClusterColorMap(colorMap);
      setNClusters(result.n_clusters);
      setNNoise(result.n_noise ?? 0);
      setActiveClusterIds(
        new Set<string>(
          (result.points as UmapPoint[]).map(p => p.cluster).filter(c => c !== '-1'),
        ),
      );
    } catch (error) {
      console.error('Backend unreachable.', error);
    } finally {
      setLoading(false);
    }
  };

  const selectCluster = async (clusterId: string) => {
    if (clusterId === '-1') return;
    setSelectedCluster(clusterId);
    setSuggestionsLoading(true);
    try {
      const resp = await fetch('http://localhost:8000/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: clusterId }),
      });
      const data = await resp.json();
      setSuggestions(data.suggestions ?? []);
    } catch (err) {
      console.error('Annotate fetch failed.', err);
      setSuggestions([]);
    } finally {
      setSuggestionsLoading(false);
    }
  };

  const handleAcceptSuggestion = async (cellType: string) => {
    if (!selectedCluster) return;
    setAnnotations(prev => ({
      ...prev,
      [selectedCluster]: { label: cellType, status: 'confirmed' },
    }));
    try {
      await fetch('http://localhost:8000/annotations/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: selectedCluster, label: cellType, status: 'confirmed' }),
      });
    } catch (err) {
      console.error('Save annotation failed.', err);
    }
  };

  const handleConfirmAnnotation = async () => {
    if (!selectedCluster || !annotationInput.trim()) return;
    const label = annotationInput.trim();
    setAnnotations(prev => ({
      ...prev,
      [selectedCluster]: { label, status: 'confirmed' },
    }));
    setAnnotationInput('');
    try {
      await fetch('http://localhost:8000/annotations/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: selectedCluster, label, status: 'confirmed' }),
      });
    } catch (err) {
      console.error('Save annotation failed.', err);
    }
  };

  const fetchShap = async () => {
    setShapLoading(true);
    try {
      const response = await fetch('http://localhost:8000/shap', { method: 'POST' });
      const result = await response.json();
      setShapData(result.clusters);
      setShapStale(false);
    } catch (error) {
      console.error('SHAP fetch failed.', error);
    } finally {
      setShapLoading(false);
    }
  };

  // ── Slider / algorithm handlers ────────────────────────────────────────

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setResolution(val);
    setShapStale(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(
      () => fetchCluster(val, algorithm, minClusterSize, minSamples), 400,
    );
  };

  const handleMinClusterSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setMinClusterSize(val);
    setShapStale(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(
      () => fetchCluster(resolution, algorithm, val, minSamples), 400,
    );
  };

  const handleMinSamplesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value);
    setMinSamples(val);
    setShapStale(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(
      () => fetchCluster(resolution, algorithm, minClusterSize, val), 400,
    );
  };

  const handleAlgorithmChange = (alg: Algorithm) => {
    setAlgorithm(alg);
    setShapStale(true);
    fetchCluster(resolution, alg, minClusterSize, minSamples);
  };

  const handlePlotClick = (event: any) => {
    if (!event?.points?.length) return;
    const traceName: string = event.points[0].data.name;
    if (traceName === 'Noise') return;
    selectCluster(traceName.replace('Cluster ', ''));
  };

  // ── Import handlers ────────────────────────────────────────────────────

  const resetDatasetState = () => {
    setTraces([]);
    setNClusters(null);
    setNNoise(0);
    setSelectedCluster(null);
    setAnnotations({});
    setSuggestions([]);
    setShapData(null);
  };

  const handleLoadBuiltin = async () => {
    setImportLoading(true);
    setImportError(null);
    try {
      const resp = await fetch('http://localhost:8000/load-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset: selectedBuiltin }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        setImportError(err.detail ?? 'Failed to load dataset.');
        return;
      }
      const data = await resp.json();
      setDatasetName('PBMC 3k (built-in)');
      setDatasetInfo({ n_cells: data.n_cells, n_genes: data.n_genes });
      setShowImportModal(false);
      resetDatasetState();
      fetchCluster(resolution, algorithm, minClusterSize, minSamples);
    } catch {
      setImportError('Network error. Is the backend running?');
    } finally {
      setImportLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setImportLoading(true);
    setImportError(null);
    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      const resp = await fetch('http://localhost:8000/upload-dataset', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json();
        setImportError(err.detail ?? 'Upload failed.');
        return;
      }
      const data = await resp.json();
      setDatasetName(data.dataset);
      setDatasetInfo({ n_cells: data.n_cells, n_genes: data.n_genes });
      setShowImportModal(false);
      setUploadFile(null);
      resetDatasetState();
      fetchCluster(resolution, algorithm, minClusterSize, minSamples);
    } catch {
      setImportError('Network error. Is the backend running?');
    } finally {
      setImportLoading(false);
    }
  };

  // ── Derived values ─────────────────────────────────────────────────────

  const allClusterIds = Array.from(
    new Set([
      ...Array.from(activeClusterIds),
      ...Object.keys(annotations).filter(id => id !== '-1'),
    ]),
  ).sort((a, b) => parseInt(a) - parseInt(b));

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100vh', background: '#f8f8f8',
      fontFamily: 'system-ui, Inter, sans-serif',
    }}>
      <style>{`
        body { margin: 0; padding: 0; }
        @keyframes lassoPulse {
          0%, 100% { opacity: 0.15; }
          50%       { opacity: 0.45; }
        }
        input[type=range] { accent-color: #7F77DD; }
        @keyframes obSlideRight {
          from { transform: translateX(40px); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
        @keyframes obSlideLeft {
          from { transform: translateX(-40px); opacity: 0; }
          to   { transform: translateX(0);     opacity: 1; }
        }
        .ob-slide-right { animation: obSlideRight 220ms ease-out; }
        .ob-slide-left  { animation: obSlideLeft  220ms ease-out; }
      `}</style>

      {/* ── Top navbar ── */}
      <div style={{
        height: '44px', flexShrink: 0,
        background: '#fff', borderBottom: '0.5px solid #e0e0e0',
        display: 'flex', alignItems: 'center', padding: '0 16px', gap: '12px',
      }}>
        <span style={{ fontWeight: 500, fontSize: '14px' }}>Scannotate</span>
        {datasetInfo && (
          <span style={{ fontSize: '12px', color: '#888', fontWeight: 400 }}>
            {datasetName} · {datasetInfo.n_cells.toLocaleString()} cells · {datasetInfo.n_genes.toLocaleString()} genes
          </span>
        )}
        <div style={{ flex: 1 }} />
        {/* Help / re-open onboarding */}
        <button
          onClick={() => { setOnboardingStep(0); setOnboardingDir('forward'); setShowOnboarding(true); }}
          onMouseEnter={() => setHelpBtnHover(true)}
          onMouseLeave={() => setHelpBtnHover(false)}
          title="Show introduction"
          style={{
            width: 24, height: 24, borderRadius: '50%', padding: 0,
            border: `0.5px solid ${helpBtnHover ? '#AFA9EC' : '#D3D1C7'}`,
            background: helpBtnHover ? '#EEEDFE' : '#fff',
            color: helpBtnHover ? '#3C3489' : '#888',
            fontSize: '13px', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'inherit', flexShrink: 0,
            transition: 'background 0.12s, color 0.12s, border-color 0.12s',
          }}
        >?</button>
        <button
          onClick={() => {
            setShowImportModal(true);
            setImportError(null);
            setImportTab('builtin');
            setUploadFile(null);
          }}
          style={{
            padding: '5px 12px', fontSize: '12px',
            background: '#fff', color: '#444',
            border: '1px solid #ddd', borderRadius: '6px',
            cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          Load Data
        </button>
      </div>

      {/* ── Onboarding modal ── */}
      {showOnboarding && (
        <div style={{
          position: 'fixed', inset: 0,
          background: 'rgba(0,0,0,0.5)', zIndex: 2000,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#fff', borderRadius: '14px',
            width: '600px', maxWidth: '92vw', maxHeight: '88vh',
            overflow: 'hidden', boxShadow: '0 12px 40px rgba(0,0,0,0.2)',
            display: 'flex', flexDirection: 'column',
          }}>
            {/* Illustrated header band */}
            <div style={{
              height: '140px', flexShrink: 0,
              background: OB_HEADERS[onboardingStep].bg,
              borderRadius: '14px 14px 0 0',
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ fontSize: '48px', lineHeight: 1 }}>{OB_HEADERS[onboardingStep].symbol}</span>
              <span style={{
                fontSize: '13px', fontWeight: 500, marginTop: '10px',
                color: OB_HEADERS[onboardingStep].subtitleColor,
              }}>
                {OB_HEADERS[onboardingStep].subtitle}
              </span>
            </div>

            {/* Text content — key forces remount & re-triggers animation */}
            <div
              key={onboardingStep}
              className={onboardingDir === 'forward' ? 'ob-slide-right' : 'ob-slide-left'}
              style={{ flex: 1, overflowY: 'auto', padding: '28px 32px 0 32px' }}
            >
              <div style={{ fontSize: '11px', color: '#888', fontWeight: 400, marginBottom: '6px' }}>
                Step {onboardingStep + 1} of 6
              </div>
              <div style={{ fontSize: '20px', fontWeight: 600, color: '#1a1a1a', marginBottom: '12px' }}>
                {OB_TITLES[onboardingStep]}
              </div>
              {renderOnboardingBody(onboardingStep)}
              {/* padding buffer so content doesn't touch footer */}
              <div style={{ height: '24px' }} />
            </div>

            {/* Footer */}
            <div style={{
              flexShrink: 0, padding: '20px 32px',
              borderTop: '0.5px solid #e0e0e0',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              {/* Skip */}
              <button
                onClick={completeOnboarding}
                onMouseEnter={e => (e.currentTarget.style.textDecoration = 'underline')}
                onMouseLeave={e => (e.currentTarget.style.textDecoration = 'none')}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: '12px', color: '#888', fontFamily: 'inherit',
                  padding: 0, textDecoration: 'none',
                }}
              >
                Skip introduction
              </button>

              {/* Step dots */}
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                {[0, 1, 2, 3, 4, 5].map(i => (
                  <div key={i} style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: i === onboardingStep ? '#7F77DD' : '#D3D1C7',
                  }} />
                ))}
              </div>

              {/* Back + Next/Get started */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                {onboardingStep > 0 && (
                  <button
                    onClick={() => goToStep(onboardingStep - 1, 'backward')}
                    style={{
                      padding: '8px 16px', fontSize: '13px',
                      background: '#fff', color: '#666',
                      border: '1px solid #ddd', borderRadius: '6px',
                      cursor: 'pointer', fontFamily: 'inherit',
                    }}
                  >Back</button>
                )}
                <button
                  onClick={() => {
                    if (onboardingStep < 5) goToStep(onboardingStep + 1, 'forward');
                    else completeOnboarding();
                  }}
                  style={{
                    padding: '8px 20px', fontSize: '13px',
                    background: '#7F77DD', color: '#fff',
                    border: 'none', borderRadius: '6px',
                    cursor: 'pointer', fontFamily: 'inherit',
                  }}
                >
                  {onboardingStep < 5 ? 'Next' : 'Get started'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Import modal ── */}
      {showImportModal && (
        <div style={{
          position: 'fixed', inset: 0,
          background: 'rgba(0,0,0,0.4)', zIndex: 1000,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            background: '#fff', borderRadius: '12px',
            padding: '28px 32px', width: '520px', maxWidth: '90vw',
            boxShadow: '0 8px 32px rgba(0,0,0,0.18)',
            position: 'relative',
          }}>
            <button
              onClick={() => { setShowImportModal(false); setImportError(null); }}
              style={{
                position: 'absolute', top: '16px', right: '20px',
                fontSize: '18px', color: '#888', cursor: 'pointer',
                background: 'none', border: 'none', lineHeight: 1,
                fontFamily: 'inherit',
              }}
            >×</button>

            <div style={{ fontSize: '16px', fontWeight: 500, marginBottom: '16px' }}>
              Load dataset
            </div>

            <div style={{
              display: 'flex', gap: '16px',
              marginBottom: '20px', borderBottom: '1px solid #e0e0e0',
            }}>
              {(['builtin', 'upload'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => { setImportTab(tab); setImportError(null); }}
                  style={{
                    padding: '8px 0', fontSize: '13px',
                    background: 'none', border: 'none', cursor: 'pointer',
                    fontFamily: 'inherit', marginBottom: '-1px',
                    borderBottom: importTab === tab ? '2px solid #7F77DD' : '2px solid transparent',
                    color: importTab === tab ? '#3C3489' : '#888',
                    fontWeight: importTab === tab ? 500 : 400,
                  }}
                >
                  {tab === 'builtin' ? 'Built-in datasets' : 'Upload your own'}
                </button>
              ))}
            </div>

            {importError && (
              <div style={{
                background: '#FDE8E8', border: '1px solid #F5C4B3',
                color: '#712B13', borderRadius: '6px',
                padding: '8px 10px', fontSize: '12px', marginBottom: '12px',
              }}>
                {importError}
              </div>
            )}

            {importTab === 'builtin' ? (
              <>
                <select
                  value={selectedBuiltin}
                  onChange={e => setSelectedBuiltin(e.target.value)}
                  disabled={importLoading}
                  style={{
                    width: '100%', border: '0.5px solid #D3D1C7',
                    borderRadius: '6px', padding: '8px 10px', fontSize: '13px',
                    fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box',
                  }}
                >
                  <option value="pbmc3k">
                    PBMC 3k · 2,700 cells · Peripheral blood, 10x Genomics (default)
                  </option>
                </select>

                <div style={hintBox}>
                  PBMC 3k is the canonical single-cell benchmark dataset from 10x Genomics,
                  containing ~2,700 peripheral blood mononuclear cells across 8 well-characterized
                  populations. It is used as the default dataset for this tool.
                </div>

                <button
                  onClick={handleLoadBuiltin}
                  disabled={importLoading}
                  style={{
                    width: '100%', padding: '10px', fontSize: '13px',
                    background: importLoading ? '#B2DFD2' : '#1D9E75',
                    color: '#fff', border: 'none', borderRadius: '6px',
                    cursor: importLoading ? 'not-allowed' : 'pointer',
                    marginTop: '16px', fontFamily: 'inherit',
                  }}
                >
                  {importLoading ? 'Processing…' : 'Load dataset'}
                </button>
              </>
            ) : (
              <>
                <div style={{
                  background: '#f8f8f8', borderRadius: '8px',
                  padding: '12px 14px', marginBottom: '14px',
                }}>
                  <div style={{ fontSize: '13px', fontWeight: 500, marginBottom: '10px' }}>
                    Accepted formats
                  </div>
                  {[
                    {
                      fmt: '.h5ad',
                      color: '#7F77DD', bg: '#EEEDFE',
                      desc: 'AnnData object — recommended. Preserves gene names, metadata, and any existing obs/var columns.',
                    },
                    {
                      fmt: '.csv / .tsv',
                      color: '#085041', bg: '#E1F5EE',
                      desc: 'Plain count matrix with cells as rows and genes as columns (or genes × cells — the tool auto-detects orientation). First row = gene names, first column = cell barcodes.',
                    },
                    {
                      fmt: '.mtx',
                      color: '#633806', bg: '#FAEEDA',
                      desc: 'Sparse Matrix Market format produced by tools like Cell Ranger or STARsolo. Gene names default to gene_0, gene_1, … unless supplied via a separate barcodes/features file.',
                    },
                    {
                      fmt: '.zip',
                      color: '#0C447C', bg: '#E6F1FB',
                      desc: '10x Genomics bundle — zip the folder containing matrix.mtx(.gz), barcodes.tsv(.gz), and features.tsv(.gz). Gene symbols are used automatically.',
                    },
                  ].map(({ fmt, color, bg, desc }) => (
                    <div key={fmt} style={{ marginBottom: '10px' }}>
                      <span style={{
                        display: 'inline-block', background: bg, color,
                        borderRadius: '4px', padding: '1px 7px',
                        fontSize: '11px', fontWeight: 600, marginBottom: '3px',
                      }}>{fmt}</span>
                      <div style={{ fontSize: '12px', color: '#555', lineHeight: '1.6' }}>{desc}</div>
                    </div>
                  ))}
                  <div style={{
                    marginTop: '8px', borderTop: '0.5px solid #e0e0e0',
                    paddingTop: '8px', fontSize: '11px', color: '#888', lineHeight: '1.6',
                  }}>
                    All formats require at least 50 cells and 100 genes. The tool automatically runs
                    normalization, log-transform, HVG selection, PCA, and KNN graph construction.
                  </div>
                </div>

                <div
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={e => {
                    e.preventDefault();
                    setDragOver(false);
                    const f = e.dataTransfer.files[0];
                    if (f) setUploadFile(f);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: `2px dashed ${dragOver ? '#7F77DD' : '#D3D1C7'}`,
                    borderRadius: '8px', padding: '28px', textAlign: 'center',
                    cursor: 'pointer',
                    background: dragOver ? '#EEEDFE' : 'transparent',
                    transition: 'border-color 0.15s, background 0.15s',
                  }}
                >
                  <span style={{ fontSize: '24px', color: '#888', display: 'block', marginBottom: '8px' }}>↑</span>
                  <div style={{ fontSize: '13px', color: '#555' }}>Drag and drop your file here</div>
                  <div style={{ fontSize: '12px', color: '#aaa', marginTop: '4px' }}>
                    .h5ad · .csv · .tsv · .mtx · .zip (10x)
                  </div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>or click to browse</div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".h5ad,.csv,.tsv,.mtx,.zip"
                    style={{ display: 'none' }}
                    onChange={e => {
                      const f = e.target.files?.[0];
                      if (f) setUploadFile(f);
                    }}
                  />
                </div>

                {uploadFile && (
                  <>
                    <div style={{ fontSize: '12px', color: '#888', marginTop: '8px' }}>
                      {uploadFile.name} · {(uploadFile.size / 1024 / 1024).toFixed(1)} MB
                    </div>
                    <button
                      onClick={handleUpload}
                      disabled={importLoading}
                      style={{
                        width: '100%', padding: '10px', fontSize: '13px',
                        background: importLoading ? '#B2DFD2' : '#1D9E75',
                        color: '#fff', border: 'none', borderRadius: '6px',
                        cursor: importLoading ? 'not-allowed' : 'pointer',
                        marginTop: '12px', fontFamily: 'inherit',
                      }}
                    >
                      {importLoading ? 'Processing…' : 'Upload and analyze'}
                    </button>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Three-column body ── */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ── Left panel ── */}
        <div style={{
          flex: '0 0 20%',
          background: '#fff', borderRight: '0.5px solid #e0e0e0',
          padding: '16px', overflowY: 'auto',
          display: 'flex', flexDirection: 'column', gap: '18px',
        }}>

          <section>
            <span style={sectionLabel}>Algorithm</span>
            <div style={{ display: 'flex', gap: '4px' }}>
              {ALGO_DEFS.map(({ value, label }) => {
                const isActive = algorithm === value;
                const activeColor = ALGO_ACTIVE_COLOR[value] ?? '#aaa';
                return (
                  <button
                    key={value}
                    onClick={() => handleAlgorithmChange(value as Algorithm)}
                    style={{
                      flex: 1, padding: '5px 0', fontSize: '11px',
                      borderRadius: '5px', border: '1px solid',
                      borderColor: isActive ? activeColor : '#ddd',
                      background: isActive ? activeColor : '#fff',
                      color: isActive ? '#fff' : '#555',
                      cursor: 'pointer', fontFamily: 'inherit',
                      transition: 'background 0.15s, color 0.15s',
                    }}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            <div style={hintBox}>{ALGO_HINT[algorithm]}</div>
          </section>

          <section>
            <span style={sectionLabel}>
              {algorithm === 'leiden' ? 'Resolution' : 'Parameters'}
            </span>

            {algorithm === 'leiden' ? (
              <>
                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: '12px', color: '#444', marginBottom: '6px',
                }}>
                  <span>Resolution</span>
                  <span style={{ color: '#222', fontWeight: 500 }}>{resolution.toFixed(1)}</span>
                </div>
                <input
                  type="range" min="0.1" max="2.0" step="0.1"
                  value={resolution} onChange={handleSliderChange}
                  style={{ width: '100%' }}
                />
                <div style={hintBox}>Controls granularity of clustering. Try 0.4–1.2 for most datasets.</div>
              </>
            ) : (
              <>
                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: '12px', color: '#444', marginBottom: '6px',
                }}>
                  <span>Min cluster size</span>
                  <span style={{ color: '#222', fontWeight: 500 }}>{minClusterSize}</span>
                </div>
                <input
                  type="range" min="10" max="200" step="5"
                  value={minClusterSize} onChange={handleMinClusterSizeChange}
                  style={{ width: '100%' }}
                />
                <div style={hintBox}>
                  Minimum cells required to form a cluster. Smaller values surface rare populations.
                </div>

                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: '12px', color: '#444', marginBottom: '6px', marginTop: '10px',
                }}>
                  <span>Min samples</span>
                  <span style={{ color: '#222', fontWeight: 500 }}>{minSamples}</span>
                </div>
                <input
                  type="range" min="1" max="30" step="1"
                  value={minSamples} onChange={handleMinSamplesChange}
                  style={{ width: '100%' }}
                />
                <div style={hintBox}>
                  Higher values label more borderline cells as noise rather than assigning them to a cluster.
                </div>

                {nNoise > 0 && (
                  <div style={{
                    display: 'inline-block', marginTop: '10px',
                    background: '#FAEEDA', border: '1px solid #FAC775',
                    color: '#633806', borderRadius: '20px',
                    padding: '3px 10px', fontSize: '11px',
                  }}>
                    {nNoise} cells labeled as noise
                  </div>
                )}
              </>
            )}
          </section>

          <section>
            <span style={sectionLabel}>Cluster Tools</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <button style={{
                width: '100%', padding: '6px 0', fontSize: '12px',
                background: '#fff', border: '1px solid #ddd',
                borderRadius: '6px', cursor: 'pointer',
                color: '#444', fontFamily: 'inherit',
              }}>
                Merge clusters
              </button>
              <button style={{
                width: '100%', padding: '6px 0', fontSize: '12px',
                background: '#fff', border: '1px solid #ddd',
                borderRadius: '6px', cursor: 'pointer',
                color: '#444', fontFamily: 'inherit',
              }}>
                Lasso split
              </button>
            </div>
          </section>

          {nClusters !== null && (
            <section>
              <span style={sectionLabel}>Quality</span>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ color: '#888' }}>Clusters</span>
                <span style={{ color: '#222', fontWeight: 500 }}>{nClusters}</span>
              </div>
              {algorithm === 'hdbscan' && nNoise > 0 && (
                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: '12px', marginTop: '4px',
                }}>
                  <span style={{ color: '#888' }}>Noise cells</span>
                  <span style={{ color: '#222', fontWeight: 500 }}>{nNoise}</span>
                </div>
              )}
            </section>
          )}

        </div>

        {/* ── Center panel ── */}
        <div style={{
          flex: 1, minWidth: 0, background: '#fff',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Plot
              data={traces}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              layout={{
                autosize: true,
                height: undefined,
                margin: { t: 30, r: 150, b: 50, l: 60 },
                xaxis: { title: { text: 'UMAP 1' }, zeroline: false },
                yaxis: { title: { text: 'UMAP 2' }, zeroline: false },
                legend: { title: { text: 'Cluster' }, itemsizing: 'constant' },
                plot_bgcolor:  'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
              }}
              config={{ displayModeBar: false }}
              onClick={handlePlotClick}
            />

            {loading && (
              <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: 'rgba(255,255,255,0.7)',
                fontSize: '13px', fontWeight: 500, color: '#666',
              }}>
                Loading…
              </div>
            )}
          </div>

          <div style={{
            textAlign: 'center', fontSize: '11px',
            color: '#bbb', padding: '8px 0 10px', flexShrink: 0,
          }}>
            Click a cluster to inspect · Lasso to split
          </div>
        </div>

        {/* ── Right panel ── */}
        <div style={{
          flex: '0 0 260px',
          borderLeft: '0.5px solid #e0e0e0', background: '#fff',
          display: 'flex', flexDirection: 'column', overflowY: 'auto',
          padding: '14px', gap: '0',
        }}>

          <section>
            <span style={sectionLabel}>Annotations</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {allClusterIds.length === 0 ? (
                <div style={{ fontSize: '12px', color: '#bbb' }}>No clusters yet.</div>
              ) : allClusterIds.map(id => {
                const ann        = annotations[id];
                const isActive   = activeClusterIds.has(id);
                const isSelected = selectedCluster === id;
                const dotColor   = clusterColorMap[id] ?? '#aaaaaa';
                const label      = ann?.label;
                const status     = ann?.status ?? 'unannotated';
                return (
                  <div
                    key={id}
                    onClick={() => selectCluster(id)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '6px',
                      padding: '5px 6px', borderRadius: '5px',
                      background: isSelected ? '#EEEDFE' : 'transparent',
                      cursor: 'pointer',
                      opacity: isActive ? 1 : 0.4,
                    }}
                  >
                    <div style={{
                      width: 10, height: 10, borderRadius: '50%',
                      background: dotColor, flexShrink: 0,
                    }} />
                    <span style={{
                      flex: 1, fontSize: '12px',
                      color: label ? '#222' : '#aaa',
                      fontStyle: label ? 'normal' : 'italic',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {label ?? `Cluster ${id}`}
                    </span>
                    <StatusBadge status={status} />
                  </div>
                );
              })}
            </div>
          </section>

          {selectedCluster !== null && (
            <section style={{ marginTop: '12px' }}>
              <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '0 0 12px' }} />
              <span style={sectionLabel}>Annotate Cluster {selectedCluster}</span>

              {suggestionsLoading ? (
                <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '10px' }}>
                  Fetching suggestions…
                </div>
              ) : suggestions.length > 0 ? (
                <>
                  <div style={{
                    background: '#FAEEDA', border: '1px solid #FAC775',
                    borderRadius: '8px', padding: '8px 10px',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    marginBottom: '8px',
                  }}>
                    <span style={{
                      fontSize: '13px', color: '#633806', fontWeight: 500,
                      flex: 1, marginRight: '8px',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {suggestions[0].cell_type}
                    </span>
                    <span style={{ fontSize: '12px', color: '#854F0B', flexShrink: 0 }}>
                      {Math.round(suggestions[0].score * 100)}%
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                    <button
                      onClick={() => handleAcceptSuggestion(suggestions[0].cell_type)}
                      style={{
                        flex: 1, padding: '5px 0', fontSize: '12px',
                        background: '#1D9E75', color: '#fff',
                        border: 'none', borderRadius: '6px',
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >Accept</button>
                    <button
                      style={{
                        flex: 1, padding: '5px 0', fontSize: '12px',
                        background: '#fff', color: '#666',
                        border: '1px solid #ddd', borderRadius: '6px',
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >Override</button>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '10px' }}>
                    {suggestions.slice(1).map(s => (
                      <button
                        key={s.cell_type}
                        onClick={() => handleAcceptSuggestion(s.cell_type)}
                        style={{
                          border: '0.5px solid #D3D1C7', borderRadius: '20px',
                          fontSize: '11px', padding: '2px 8px',
                          background: '#fff', color: '#5F5E5A',
                          cursor: 'pointer', fontFamily: 'inherit',
                        }}
                        onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                          e.currentTarget.style.background = '#F1EFE8';
                        }}
                        onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                          e.currentTarget.style.background = '#fff';
                        }}
                      >
                        {s.cell_type}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: '12px', color: '#bbb', marginBottom: '10px' }}>
                  No suggestions available.
                </div>
              )}

              <input
                type="text"
                value={annotationInput}
                onChange={e => setAnnotationInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleConfirmAnnotation(); }}
                placeholder="or type a custom label…"
                style={{
                  width: '100%', fontSize: '12px',
                  border: '0.5px solid #D3D1C7', borderRadius: '6px',
                  padding: '6px 8px', boxSizing: 'border-box',
                  fontFamily: 'inherit', outline: 'none',
                }}
              />
              <button
                onClick={handleConfirmAnnotation}
                style={{
                  width: '100%', marginTop: '6px', padding: '6px 0', fontSize: '12px',
                  background: '#1D9E75', color: '#fff',
                  border: 'none', borderRadius: '6px',
                  cursor: 'pointer', fontFamily: 'inherit',
                }}
              >Confirm</button>
            </section>
          )}

          <section style={{ marginTop: '16px' }}>
            <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '0 0 12px' }} />
            <span style={sectionLabel}>Gene Search</span>
            <input
              type="text"
              placeholder="e.g. CD3E, FOXP3…"
              style={{
                width: '100%', fontSize: '12px',
                border: '0.5px solid #D3D1C7', borderRadius: '6px',
                padding: '6px 8px', boxSizing: 'border-box',
                fontFamily: 'inherit', outline: 'none',
              }}
            />
            <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
              Overlays expression on UMAP
            </div>
          </section>

          <section style={{ marginTop: '16px' }}>
            <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '0 0 12px' }} />
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={sectionLabel}>Driver Genes</span>
              <button
                onClick={fetchShap}
                disabled={shapLoading}
                style={{
                  padding: '5px 12px', fontSize: '11px',
                  background: shapLoading ? '#f0f0f0' : '#7F77DD',
                  color: shapLoading ? '#aaa' : '#fff',
                  border: 'none', borderRadius: '5px',
                  cursor: shapLoading ? 'not-allowed' : 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                {shapLoading ? 'Computing…' : 'Compute driver genes'}
              </button>
            </div>

            {shapStale && shapData !== null && (
              <div style={{
                ...hintBox,
                marginTop: 0, background: '#FFF8E1',
                border: '1px solid #F9A825', color: '#5D4037',
                marginBottom: '12px',
              }}>
                Clustering changed. Re-run driver gene analysis to refresh.
              </div>
            )}

            {selectedCluster !== null && shapData !== null && shapData[selectedCluster] && (() => {
              const genes  = shapData[selectedCluster].map(d => d.gene).reverse();
              const values = shapData[selectedCluster].map(d => d.shap).reverse();
              const top3   = shapData[selectedCluster].slice(0, 3).map(d => d.gene);
              return (
                <>
                  <div style={{ fontSize: '13px', fontWeight: 500, marginBottom: '8px', color: '#222' }}>
                    Driver genes — cluster {selectedCluster}
                  </div>
                  <Plot
                    data={[{
                      type: 'bar',
                      orientation: 'h',
                      x: values,
                      y: genes,
                      marker: { color: '#F59E0B' },
                    }]}
                    layout={{
                      height: 280,
                      width: 220,
                      margin: { t: 10, r: 16, b: 40, l: 80 },
                      xaxis: { title: { text: 'mean |SHAP|' }, zeroline: false },
                      yaxis: { automargin: true },
                      plot_bgcolor:  'rgba(0,0,0,0)',
                      paper_bgcolor: 'rgba(0,0,0,0)',
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '12px', color: '#555', marginTop: '4px' }}>
                    This cluster is most distinguished by{' '}
                    <strong>{top3[0]}</strong>, <strong>{top3[1]}</strong>, and <strong>{top3[2]}</strong>.
                  </div>
                </>
              );
            })()}

            {shapData !== null && selectedCluster === null && (
              <div style={{ fontSize: '12px', color: '#aaa' }}>
                Click a cluster on the UMAP to see its driver genes.
              </div>
            )}

            {shapData === null && !shapLoading && (
              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.5' }}>
                Run driver gene analysis to see which genes distinguish each cluster.
              </div>
            )}
          </section>

        </div>

      </div>
    </div>
  );
};

export default App;
