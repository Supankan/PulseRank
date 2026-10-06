import React, { useState, useEffect } from 'react';
import { fetchMonitoringStats, fetchDriftReport, fetchRegistryModels } from '../api/client';

export const MonitoringView: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [drift, setDrift] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = () => {
    setLoading(true);
    setError(null);
    Promise.all([fetchMonitoringStats(), fetchDriftReport(), fetchRegistryModels()])
      .then(([s, d, m]) => {
        setStats(s);
        setDrift(d);
        setModels(m.models || {});
      })
      .catch((err) => setError(err.message || 'Failed to load monitoring telemetry'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Inference Monitoring & MLOps Console</h2>
          <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Real-time inference telemetry, candidate channel attribution, and Population Stability Index (PSI) category drift.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          style={{ padding: '0.5rem 1rem', background: 'var(--surface-hover)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' }}
        >
          {loading ? 'Refreshing...' : 'Refresh Metrics'}
        </button>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      {/* Latency & Throughput KPIs */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Total Requests Monitored</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{stats.total_requests}</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Mean Serving Latency</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem', color: 'var(--primary)' }}>
              {stats.mean_latency_ms.toFixed(1)} ms
            </div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>P95 Tail Latency</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{stats.p95_latency_ms.toFixed(1)} ms</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Prediction Entropy</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{stats.mean_prediction_entropy.toFixed(3)}</div>
          </div>
        </div>
      )}

      {/* PSI Drift Alert & Category Shift */}
      {drift && (
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>Population Stability Index (PSI) Drift</h3>
              <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                Comparing training corpus distribution (P) against live serving impressions (Q).
              </p>
            </div>
            <span style={{
              padding: '0.35rem 0.85rem',
              borderRadius: '9999px',
              fontSize: '0.8rem',
              fontWeight: 700,
              background: drift.psi_score < 0.10 ? '#d1fae5' : drift.psi_score < 0.25 ? '#fef3c7' : '#fee2e2',
              color: drift.psi_score < 0.10 ? '#065f46' : drift.psi_score < 0.25 ? '#92400e' : '#991b1b',
            }}>
              PSI: {drift.psi_score.toFixed(4)} ({drift.status.toUpperCase()})
            </span>
          </div>

          <div style={{ padding: '0.75rem 1rem', background: 'var(--bg-card)', borderRadius: '0.5rem', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem', fontSize: '0.85rem' }}>
            <strong>Action:</strong> {drift.recommended_action}
          </div>

          {/* Category Distribution Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
            {Object.keys(drift.baseline_distribution || {}).map((cat) => {
              const basePct = (drift.baseline_distribution[cat] * 100).toFixed(1);
              const servePct = ((drift.serving_distribution?.[cat] || 0) * 100).toFixed(1);
              return (
                <div key={cat} style={{ padding: '0.75rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', textTransform: 'capitalize' }}>{cat}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    <span>Baseline: {basePct}%</span>
                    <span>Served: {servePct}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Model Registry Console */}
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1.1rem', fontWeight: 600 }}>Model Registry & Promotion Gates</h3>
        {models && Object.keys(models).length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {Object.entries(models).map(([mName, versions]: [string, any]) => (
              <div key={mName} style={{ padding: '1rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem' }}>
                <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.5rem' }}>{mName}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {versions.map((v: any) => (
                    <div key={v.version} style={{ padding: '0.5rem 0.75rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.35rem', fontSize: '0.8rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600 }}>{v.version}</span>
                        <span style={{
                          fontSize: '0.65rem',
                          fontWeight: 700,
                          padding: '0.1rem 0.4rem',
                          borderRadius: '0.25rem',
                          background: v.stage === 'production' ? '#10b981' : v.stage === 'staging' ? '#3b82f6' : '#6b7280',
                          color: '#fff',
                          textTransform: 'uppercase',
                        }}>
                          {v.stage}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                        AUC: {v.metrics?.auc?.toFixed(3) || 'N/A'} ? nDCG: {v.metrics?.ndcg?.toFixed(3) || 'N/A'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No models registered in local registry yet. Run pipeline to register checkpoints.</p>
        )}
      </div>
    </div>
  );
};
