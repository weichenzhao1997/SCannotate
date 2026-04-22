import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

interface UmapPoint {
  x: number;
  y: number;
  cluster: string;
}

interface ShapGene { gene: string; shap: number; }
type ShapData = Record<string, ShapGene[]>;

// Group UMAP points by cluster so each cluster becomes its own Plotly trace,
// which gives automatic qualitative coloring and a legend entry per cluster.
function buildTraces(points: UmapPoint[]) {
  const byCluster = new Map<string, { x: number[]; y: number[] }>();
  for (const p of points) {
    if (!byCluster.has(p.cluster)) byCluster.set(p.cluster, { x: [], y: [] });
    const bucket = byCluster.get(p.cluster)!;
    bucket.x.push(p.x);
    bucket.y.push(p.y);
  }
  // Sort cluster IDs numerically so legend order matches cluster numbering
  const sorted = [...byCluster.keys()].sort((a, b) => parseInt(a) - parseInt(b));
  return sorted.map((id) => ({
    x: byCluster.get(id)!.x,
    y: byCluster.get(id)!.y,
    mode: 'markers',
    type: 'scatter',
    name: `Cluster ${id}`,
    marker: { size: 3, opacity: 0.7 },
  }));
}

type Algorithm = 'leiden' | 'hdbscan';

// ── UI metadata (not state) ────────────────────────────────────────────────

const ALGO_DEFS: { value: string; label: string; disabled?: boolean }[] = [
  { value: 'leiden',  label: 'Leiden'  },
  { value: 'hdbscan', label: 'HDBSCAN', disabled: true },
];

const ALGO_ACTIVE_COLOR: Record<string, string> = {
  leiden:  '#7F77DD',
  hdbscan: '#7F77DD',
};

const ALGO_HINT: Record<string, string> = {
  leiden:  'Leiden offers better community detection & is generally faster',
  hdbscan: 'HDBSCAN finds clusters of arbitrary shape without a fixed resolution',
};

// ── Shared style objects ───────────────────────────────────────────────────

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

// ── Component ──────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const [resolution, setResolution] = useState<number>(0.5);
  const [algorithm, setAlgorithm] = useState<Algorithm>('leiden');
  const [traces, setTraces] = useState<object[]>([]);
  const [nClusters, setNClusters] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [shapLoading, setShapLoading] = useState<boolean>(false);
  const [shapStale, setShapStale] = useState<boolean>(false);

  const fetchCluster = async (res: number, alg: Algorithm) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution: res, algorithm: alg }),
      });
      const result = await response.json();
      setTraces(buildTraces(result.points));
      setNClusters(result.n_clusters);
    } catch (error) {
      console.error('Backend unreachable.', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch once on mount with the default resolution and algorithm
  useEffect(() => {
    fetchCluster(resolution, algorithm);
  }, []);

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

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setResolution(val);
    setShapStale(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchCluster(val, algorithm), 400);
  };

  const handleAlgorithmChange = (alg: Algorithm) => {
    setAlgorithm(alg);
    setShapStale(true);
    fetchCluster(resolution, alg);
  };

  const handlePlotClick = (event: any) => {
    if (!event?.points?.length) return;
    // trace name is "Cluster 0", "Cluster 1", etc.
    const traceName: string = event.points[0].data.name;
    setSelectedCluster(traceName.replace('Cluster ', ''));
  };

  // ── Render ───────────────────────────────────────────────────────────────

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
              {ALGO_DEFS.map(({ value, label, disabled }) => {
                const isActive = algorithm === value;
                const activeColor = ALGO_ACTIVE_COLOR[value] ?? '#aaa';
                return (
                  <button
                    key={value}
                    disabled={disabled}
                    onClick={() => !disabled && handleAlgorithmChange(value as Algorithm)}
                    style={{
                      flex: 1, padding: '5px 0', fontSize: '11px',
                      borderRadius: '5px', border: '1px solid',
                      borderColor: isActive ? activeColor : '#ddd',
                      background: isActive ? activeColor : '#fff',
                      color: isActive ? '#fff' : disabled ? '#ccc' : '#555',
                      cursor: disabled ? 'not-allowed' : 'pointer',
                      fontFamily: 'inherit',
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

          {/* RESOLUTION */}
          <section>
            <span style={sectionLabel}>Resolution</span>
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

          {/* QUALITY — cluster count from live state */}
          {nClusters !== null && (
            <section>
              <span style={sectionLabel}>Quality</span>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: '12px',
              }}>
                <span style={{ color: '#888' }}>Clusters</span>
                <span style={{ color: '#222', fontWeight: 500 }}>{nClusters}</span>
              </div>
            </section>
          )}

        </div>

        {/* ── Center panel (fills all remaining width) ── */}
        <div style={{
          flex: '0 0 50%', minWidth: 0, background: '#fff',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* Plot area */}
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

            

            {/* Loading overlay */}
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

          {/* Hint line below plot */}
          <div style={{
            textAlign: 'center', fontSize: '11px',
            color: '#bbb', padding: '8px 0 10px', flexShrink: 0,
          }}>
            Click a cluster to inspect · Lasso to split
          </div>
        </div>

        {/* ── Right panel — SHAP / driver genes ── */}
        <div style={{
          width: '0 0 20%',
          borderLeft: '0.5px solid #e0e0e0', background: '#fff',
          display: 'flex', flexDirection: 'column', overflowY: 'auto',
          padding: '16px',
        }}>
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

          {/* Staleness warning */}
          {shapStale && shapData !== null && (
            <div style={{
              ...hintBox,
              background: '#FFF8E1', border: '1px solid #F9A825',
              color: '#5D4037', marginBottom: '12px',
            }}>
              SHAP values reflect the current clustering. Re-run after changing resolution or algorithm.
            </div>
          )}

          {/* Chart + summary */}
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
                    width: 308,
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

          {/* Prompt to click a cluster once data is ready */}
          {shapData !== null && selectedCluster === null && (
            <div style={{ fontSize: '12px', color: '#aaa' }}>
              Click a cluster on the UMAP to see its driver genes.
            </div>
          )}

          {/* Initial prompt before any fetch */}
          {shapData === null && !shapLoading && (
            <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.5' }}>
              Run driver gene analysis to see which genes distinguish each cluster.
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default App;
