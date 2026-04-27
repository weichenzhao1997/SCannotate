import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

// ── Types ──────────────────────────────────────────────────────────────────

interface UmapPoint { x: number; y: number; cluster: string; }
interface ShapGene   { gene: string; shap: number; }
interface Suggestion { cell_type: string; score: number; }
type ShapData    = Record<string, ShapGene[]>;
type Annotations = Record<string, { label: string; status: string }>;

// ── Constants ──────────────────────────────────────────────────────────────

// Matches Plotly's default Plotly10 colour sequence so annotation dots
// stay in sync with UMAP trace colours.
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
  const [traces, setTraces]                   = useState<object[]>([]);
  const [clusterColorMap, setClusterColorMap] = useState<Record<string, string>>({});
  const [activeClusterIds, setActiveClusterIds] = useState<Set<string>>(new Set());
  const [nClusters, setNClusters]             = useState<number | null>(null);
  const [nNoise, setNNoise]                   = useState<number>(0);
  const [loading, setLoading]                 = useState<boolean>(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Annotation state
  const [annotations, setAnnotations]         = useState<Annotations>({});
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  const [suggestions, setSuggestions]         = useState<Suggestion[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState<boolean>(false);
  const [annotationInput, setAnnotationInput] = useState<string>('');

  // SHAP / driver genes state
  const [shapData, setShapData]       = useState<ShapData | null>(null);
  const [shapLoading, setShapLoading] = useState<boolean>(false);
  const [shapStale, setShapStale]     = useState<boolean>(false);

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

  // Fetch cluster + annotations on mount
  useEffect(() => {
    fetchCluster(resolution, algorithm, minClusterSize, minSamples);
    fetch('http://localhost:8000/annotations')
      .then(r => r.json())
      .then(d => setAnnotations(d.annotations ?? {}))
      .catch(() => {});
  }, []);

  // Select a cluster: set selection and fetch PanglaoDB suggestions
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
    // Optimistic update
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

  // ── Derived values ─────────────────────────────────────────────────────

  // All cluster IDs for the annotation list: active + any previously annotated
  // clusters that no longer exist (shown greyed out per spec).
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
      `}</style>

      {/* ── Top navbar ── */}
      <div style={{
        height: '44px', flexShrink: 0,
        background: '#fff', borderBottom: '0.5px solid #e0e0e0',
        display: 'flex', alignItems: 'center', padding: '0 16px',
      }}>
        <span style={{ fontWeight: 500, fontSize: '14px' }}>Scannotate</span>
      </div>

      {/* ── Three-column body ── */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ── Left panel ── */}
        <div style={{
          flex: '0 0 20%',
          background: '#fff', borderRight: '0.5px solid #e0e0e0',
          padding: '16px', overflowY: 'auto',
          display: 'flex', flexDirection: 'column', gap: '18px',
        }}>

          {/* ALGORITHM */}
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

          {/* PARAMETERS — conditional on active algorithm */}
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
                {/* Min cluster size */}
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

                {/* Min samples */}
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

                {/* Noise badge */}
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

          {/* CLUSTER TOOLS */}
          <section>
            <span style={sectionLabel}>Cluster Tools</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {/* TODO: Wire up merge-clusters functionality */}
              <button style={{
                width: '100%', padding: '6px 0', fontSize: '12px',
                background: '#fff', border: '1px solid #ddd',
                borderRadius: '6px', cursor: 'pointer',
                color: '#444', fontFamily: 'inherit',
              }}>
                Merge clusters
              </button>
              {/* TODO: Wire up lasso-split functionality */}
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

          {/* QUALITY — live cluster count + optional noise row for HDBSCAN */}
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

        {/* ── Center panel (fills remaining width) ── */}
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

          {/* ── ANNOTATIONS list ── */}
          <section>
            <span style={sectionLabel}>Annotations</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {allClusterIds.length === 0 ? (
                <div style={{ fontSize: '12px', color: '#bbb' }}>No clusters yet.</div>
              ) : allClusterIds.map(id => {
                const ann       = annotations[id];
                const isActive  = activeClusterIds.has(id);
                const isSelected = selectedCluster === id;
                const dotColor  = clusterColorMap[id] ?? '#aaaaaa';
                const label     = ann?.label;
                const status    = ann?.status ?? 'unannotated';
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

          {/* ── ANNOTATE CLUSTER detail (only when a cluster is selected) ── */}
          {selectedCluster !== null && (
            <section style={{ marginTop: '12px' }}>
              <hr style={{ border: 'none', borderTop: '0.5px solid #e0e0e0', margin: '0 0 12px' }} />
              <span style={sectionLabel}>Annotate Cluster {selectedCluster}</span>

              {/* Atlas suggestions */}
              {suggestionsLoading ? (
                <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '10px' }}>
                  Fetching suggestions…
                </div>
              ) : suggestions.length > 0 ? (
                <>
                  {/* Top suggestion card */}
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

                  {/* Accept / Override */}
                  <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                    <button
                      onClick={() => handleAcceptSuggestion(suggestions[0].cell_type)}
                      style={{
                        flex: 1, padding: '5px 0', fontSize: '12px',
                        background: '#1D9E75', color: '#fff',
                        border: 'none', borderRadius: '6px',
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >
                      Accept
                    </button>
                    <button
                      style={{
                        flex: 1, padding: '5px 0', fontSize: '12px',
                        background: '#fff', color: '#666',
                        border: '1px solid #ddd', borderRadius: '6px',
                        cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >
                      Override
                    </button>
                  </div>

                  {/* Secondary suggestions (rank 2–5) */}
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

              {/* Custom label input */}
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
              >
                Confirm
              </button>
            </section>
          )}

          {/* ── GENE SEARCH ── */}
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
            {/* TODO: wire gene expression overlay on UMAP */}
          </section>

          {/* ── DRIVER GENES ── */}
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
