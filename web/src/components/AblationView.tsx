import React, { useState, useEffect } from 'react';
import { fetchAblations } from '../api/client';

export const AblationView: React.FC = () => {
  const [ablations, setAblations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAblations = () => {
    setLoading(true);
    setError(null);
    fetchAblations(80)
      .then((res) => {
        setAblations(res.ablations || []);
      })
      .catch((err) => setError(err.message || 'Failed to load ablation study'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAblations();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Ablation Laboratory</h2>
          <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Systematic 9-configuration ablation benchmarking ranking accuracy, intra-list diversity (ILD), and novelty.
          </p>
        </div>
        <button
          onClick={loadAblations}
          disabled={loading}
          style={{ padding: '0.5rem 1rem', background: 'var(--surface-hover)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' }}
        >
          {loading ? 'Evaluating...' : 'Re-run Ablations'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={{ background: 'var(--surface)', border: '1px solid #3b82f640', borderRadius: '0.5rem', padding: '0.875rem 1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#3b82f6' }} />
            <strong style={{ fontSize: '0.85rem', color: '#60a5fa' }}>Track A: MIND Impression Relevance Benchmark</strong>
          </div>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Evaluates ranking metrics (AUC, MRR, nDCG@5) strictly within the impression candidate set provided in the MIND validation dataset.
          </p>
        </div>

        <div style={{ background: 'var(--surface)', border: '1px solid #10b98140', borderRadius: '0.5rem', padding: '0.875rem 1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }} />
            <strong style={{ fontSize: '0.85rem', color: '#34d399' }}>Track B: Whole-Catalog Slate Policy Benchmark</strong>
          </div>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            Evaluates the complete serving pipeline (multi-channel retrieval &rarr; scoring &rarr; MMR &rarr; exploration) across the full article catalog for slate diversity, category entropy, novelty, and coverage.
          </p>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>Evaluating 9 ablation configurations across dev behaviors...</div>
      ) : (
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th colSpan={2} style={{ padding: '0.6rem 1rem', fontWeight: 700, borderRight: '1px solid var(--border-subtle)' }}>
                    Architecture Policy
                  </th>
                  <th colSpan={4} style={{ padding: '0.6rem 1rem', fontWeight: 700, textAlign: 'center', background: '#3b82f612', color: '#60a5fa', borderRight: '1px solid var(--border-subtle)' }}>
                    Track A: MIND Impression Relevance
                  </th>
                  <th colSpan={4} style={{ padding: '0.6rem 1rem', fontWeight: 700, textAlign: 'center', background: '#10b98112', color: '#34d399', borderRight: '1px solid var(--border-subtle)' }}>
                    Track B: Whole-Catalog Slate Policy
                  </th>
                  <th style={{ padding: '0.6rem 1rem', fontWeight: 700, textAlign: 'center' }}>
                    Status
                  </th>
                </tr>
                <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  <th style={{ padding: '0.6rem 1rem', fontWeight: 600 }}>Configuration</th>
                  <th style={{ padding: '0.6rem 1rem', fontWeight: 600, borderRight: '1px solid var(--border-subtle)' }}>Architecture Scope</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>AUC</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>MRR</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>nDCG@5</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600, borderRight: '1px solid var(--border-subtle)' }}>&Delta; nDCG@5</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>ILD</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>Entropy</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600 }}>Novelty</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600, borderRight: '1px solid var(--border-subtle)' }}>Coverage</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 600, textAlign: 'center' }}>Pipeline</th>
                </tr>
              </thead>
              <tbody>
                {ablations.map((a, idx) => {
                  const deltaColor = a.delta_ndcg5_vs_full > 0 ? '#10b981' : a.delta_ndcg5_vs_full < 0 ? '#ef4444' : 'var(--text-muted)';
                  const statusLabel = a.status === 'fallback_non_ltr' ? 'Non-LTR Fallback' : a.status === 'measured' ? 'Measured' : a.status || 'Measured';
                  const statusBg = a.status === 'fallback_non_ltr' ? '#f59e0b20' : '#10b98120';
                  const statusColor = a.status === 'fallback_non_ltr' ? '#f59e0b' : '#10b981';

                  return (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '0.75rem 1rem', fontWeight: 600, whiteSpace: 'nowrap' }}>{a.config_name}</td>
                      <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)', maxWidth: '240px', borderRight: '1px solid var(--border-subtle)' }}>{a.description}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace' }}>{(a.auc * 100).toFixed(1)}%</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace' }}>{a.mrr.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace', fontWeight: 700 }}>{a.ndcg_at_5.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace', fontWeight: 700, color: deltaColor, borderRight: '1px solid var(--border-subtle)' }}>
                        {a.delta_ndcg5_vs_full > 0 ? `+${a.delta_ndcg5_vs_full.toFixed(4)}` : a.delta_ndcg5_vs_full.toFixed(4)}
                      </td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace' }}>{a.ild.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace' }}>{a.category_entropy ? a.category_entropy.toFixed(3) : '0.000'}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace' }}>{a.novelty.toFixed(2)}</td>
                      <td style={{ padding: '0.75rem 0.75rem', fontFamily: 'monospace', borderRight: '1px solid var(--border-subtle)' }}>{(a.catalog_coverage * 100).toFixed(1)}%</td>
                      <td style={{ padding: '0.75rem 0.75rem', textAlign: 'center' }}>
                        <span style={{ display: 'inline-block', padding: '0.2rem 0.5rem', borderRadius: '0.25rem', fontSize: '0.72rem', fontWeight: 600, background: statusBg, color: statusColor }}>
                          {statusLabel}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
