import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle, BarChart3, HelpCircle, AlertTriangle, Layers } from 'lucide-react';
import { ModelStatus, EvaluationResult } from '../types';
import { fetchModels, fetchEvaluations } from '../api/client';

export const ModelDashboardView: React.FC = () => {
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchModels(), fetchEvaluations()])
      .then(([modelsData, evalData]) => {
        setModels(modelsData);
        setEvaluations(evalData);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const impressionEvals = evaluations.filter((e) => !e.evaluation_mode || e.evaluation_mode === 'impression');
  const catalogEvals = evaluations.filter((e) => e.evaluation_mode === 'catalog_retrieval');
  const isSyntheticFixture = evaluations.some((e) => e.dataset_source === 'synthetic_fixture');

  // Compute best values for impression metrics
  const maxAuc = Math.max(...impressionEvals.map((e) => e.auc), 0);
  const maxMrr = Math.max(...impressionEvals.map((e) => e.mrr), 0);
  const maxNdcg5 = Math.max(...impressionEvals.map((e) => e.ndcg_at_5), 0);
  const maxNdcg10 = Math.max(...impressionEvals.map((e) => e.ndcg_at_10), 0);
  const maxRecall = Math.max(...impressionEvals.map((e) => e.recall_at_k), 0);

  // Compute best values for catalog metrics
  const maxHitRate = Math.max(...catalogEvals.map((e) => e.hit_rate_at_k || 0), 0);
  const maxCatRecall = Math.max(...catalogEvals.map((e) => e.recall_at_k), 0);
  const maxCatCov = Math.max(...catalogEvals.map((e) => e.catalog_coverage || e.coverage || 0), 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 8,
        padding: '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Cpu size={22} color="#38bdf8" />
          <div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Recommendation Model Benchmarks</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Offline evaluation framework supporting impression-level ranking and catalog-level retrieval
            </p>
          </div>
        </div>
        {isSyntheticFixture ? (
          <span style={{
            fontSize: '0.725rem',
            padding: '0.25rem 0.6rem',
            borderRadius: 4,
            background: 'rgba(245, 158, 11, 0.15)',
            color: '#fbbf24',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            fontWeight: 600,
          }}>
            DEMO FIXTURE
          </span>
        ) : (
          <span style={{
            fontSize: '0.725rem',
            padding: '0.25rem 0.6rem',
            borderRadius: 4,
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#34d399',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            fontWeight: 600,
          }}>
            MIND-SMALL
          </span>
        )}
      </div>

      {/* Synthetic Provenance Warning Banner */}
      {isSyntheticFixture && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: 8,
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
        }}>
          <AlertTriangle size={18} color="#fbbf24" style={{ flexShrink: 0, marginTop: 2 }} />
          <div style={{ fontSize: '0.825rem', color: '#fef3c7', lineHeight: 1.5 }}>
            <strong style={{ color: '#fbbf24' }}>[Demo Fixture - Not a MIND Benchmark Result]</strong>
            <div>
              Evaluation scores below were computed on synthetic test fixtures generated for pipeline smoke-testing. These figures verify pipeline math and ranker execution, but must not be cited as Microsoft News Dataset (MIND) benchmark results.
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading model benchmark metrics...
        </div>
      ) : error ? (
        <div style={{ padding: '1.5rem', color: '#f43f5e' }}>Error loading evaluations: {error}</div>
      ) : (
        <>
          {/* 1. Impression-Level Benchmark Table */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '1rem 1.25rem',
              borderBottom: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontWeight: 600,
              fontSize: '0.9rem',
              color: '#cbd5e1',
            }}>
              <BarChart3 size={16} color="#38bdf8" />
              <span>Impression-Level Ranking (MIND Benchmark Semantics)</span>
            </div>

            {impressionEvals.length === 0 ? (
              <div style={{ padding: '3rem 1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                No evaluation benchmarks recorded yet. Run <code style={{ color: '#38bdf8' }}>python -m pulserank.scripts.run_evaluation</code> to generate benchmark metrics.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-surface)', textAlign: 'left', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.75rem 1.25rem' }}>Recommender Model</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>AUC</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>MRR</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>nDCG@5</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>nDCG@10</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Recall@5</th>
                      <th style={{ padding: '0.75rem 1.25rem', textAlign: 'right' }}>Imp. Coverage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {impressionEvals.map((item, idx) => {
                      const isBestAuc = item.auc === maxAuc && maxAuc > 0;
                      const isBestMrr = item.mrr === maxMrr && maxMrr > 0;
                      const isBestNdcg5 = item.ndcg_at_5 === maxNdcg5 && maxNdcg5 > 0;
                      const isBestNdcg10 = item.ndcg_at_10 === maxNdcg10 && maxNdcg10 > 0;
                      const isBestRecall = item.recall_at_k === maxRecall && maxRecall > 0;
                      const impCov = item.impression_coverage ?? item.coverage;

                      return (
                        <tr
                          key={item.model_name}
                          style={{
                            borderTop: '1px solid var(--border-subtle)',
                            background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                          }}
                        >
                          <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600, color: '#f8fafc' }}>
                            {item.model_name}
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestAuc ? '#34d399' : 'var(--text-primary)', fontWeight: isBestAuc ? 700 : 400 }}>
                              {item.auc.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestMrr ? '#34d399' : 'var(--text-primary)', fontWeight: isBestMrr ? 700 : 400 }}>
                              {item.mrr.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestNdcg5 ? '#34d399' : 'var(--text-primary)', fontWeight: isBestNdcg5 ? 700 : 400 }}>
                              {item.ndcg_at_5.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestNdcg10 ? '#34d399' : 'var(--text-primary)', fontWeight: isBestNdcg10 ? 700 : 400 }}>
                              {item.ndcg_at_10.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestRecall ? '#34d399' : 'var(--text-primary)', fontWeight: isBestRecall ? 700 : 400 }}>
                              {item.recall_at_k.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1.25rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            {(impCov * 100).toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* 2. Whole-Catalog Retrieval Benchmark Table (if present) */}
          {catalogEvals.length > 0 && (
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              overflow: 'hidden',
            }}>
              <div style={{
                padding: '1rem 1.25rem',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontWeight: 600,
                fontSize: '0.9rem',
                color: '#cbd5e1',
              }}>
                <Layers size={16} color="#a855f7" />
                <span>Whole-Catalog Retrieval Benchmarks (Top-10 Discovery)</span>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-surface)', textAlign: 'left', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.75rem 1.25rem' }}>Candidate Stage</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>HitRate@10</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Recall@10</th>
                      <th style={{ padding: '0.75rem 1.25rem', textAlign: 'right' }}>Catalog Coverage</th>
                      <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>User Coverage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {catalogEvals.map((item, idx) => {
                      const hr = item.hit_rate_at_k ?? 0;
                      const catCov = item.catalog_coverage ?? item.coverage;
                      const isBestHr = hr === maxHitRate && maxHitRate > 0;
                      const isBestRec = item.recall_at_k === maxCatRecall && maxCatRecall > 0;
                      const isBestCov = catCov === maxCatCov && maxCatCov > 0;

                      return (
                        <tr
                          key={item.model_name}
                          style={{
                            borderTop: '1px solid var(--border-subtle)',
                            background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                          }}
                        >
                          <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600, color: '#f8fafc' }}>
                            {item.model_name}
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestHr ? '#34d399' : 'var(--text-primary)', fontWeight: isBestHr ? 700 : 400 }}>
                              {hr.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestRec ? '#34d399' : 'var(--text-primary)', fontWeight: isBestRec ? 700 : 400 }}>
                              {item.recall_at_k.toFixed(4)}
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1.25rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            <span style={{ color: isBestCov ? '#34d399' : 'var(--text-primary)', fontWeight: isBestCov ? 700 : 400 }}>
                              {(catCov * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            {((item.impression_coverage ?? 1.0) * 100).toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Model Status & Operational Cards */}
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '1rem' }}>
              Model Architecture & Operational Status
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
              {models.map((m) => (
                <div
                  key={m.name}
                  style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 8,
                    padding: '1.15rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '0.75rem',
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
                        {m.model_type}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.75rem', color: m.status === 'ready' ? '#34d399' : '#f59e0b' }}>
                        <CheckCircle size={13} />
                        <span style={{ textTransform: 'capitalize' }}>{m.status}</span>
                      </div>
                    </div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.35rem' }}>
                      {m.name}
                    </h4>
                    <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                      {m.description}
                    </p>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span>Artifact Disk Status</span>
                    <span style={{ color: m.artifact_available ? '#34d399' : '#94a3b8' }}>
                      {m.artifact_available ? 'Cached in artifacts/' : 'Not Cached'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Metric Documentation Card */}
          <div style={{
            background: 'var(--bg-surface)',
            borderRadius: 8,
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>
              <HelpCircle size={16} color="#60a5fa" />
              <span>Evaluation Methodology & Ground Truth Semantics</span>
            </div>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              PulseRank distinguishes between two evaluation regimes: (1) <strong>Impression-Level Benchmarks (MIND Standard)</strong>: Candidates are restricted to the exact slate of articles shown in the MSN impression log. <strong>AUC</strong> measures pairwise ranking discrimination between clicked and unclicked articles. <strong>MRR</strong> measures the reciprocal rank of the first clicked article. <strong>nDCG@K</strong> measures discounted cumulative gain against the ideal ranking. <strong>Impression Coverage</strong> measures the fraction of impressions scored. (2) <strong>Whole-Catalog Retrieval</strong>: Candidates are retrieved from the entire article catalog. <strong>HitRate@K</strong> and <strong>Recall@K</strong> measure whether positive validation articles are recovered, and <strong>Catalog Coverage</strong> measures the fraction of the total article corpus recommended to users.
            </p>
          </div>
        </>
      )}
    </div>
  );
};
