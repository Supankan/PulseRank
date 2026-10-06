import React from 'react';
import { X, Sparkles, Layers, Sliders, Compass } from 'lucide-react';
import { RecommendationExplanation } from '../types';

interface RecommendationModalProps {
  explanation: RecommendationExplanation | null;
  loading: boolean;
  onClose: () => void;
}

export const RecommendationModal: React.FC<RecommendationModalProps> = ({
  explanation,
  loading,
  onClose,
}) => {
  if (!explanation && !loading) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '1.25rem 1.5rem',
          borderBottom: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Compass size={20} color="#38bdf8" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Recommendation Inspector</h3>
          </div>
          <button
            onClick={onClose}
            style={{ color: 'var(--text-secondary)', padding: '0.2rem' }}
          >
            <X size={20} />
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '3rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            Retrieving recommendation decision graph...
          </div>
        ) : explanation ? (
          <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Top Score Card */}
            <div style={{
              background: 'var(--bg-surface)',
              borderRadius: 6,
              padding: '1rem 1.25rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Final Rank
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                  #{explanation.rank}
                </div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Model Score
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                  {explanation.final_score.toFixed(4)}
                </div>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Candidate Sources
                </span>
                <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.2rem' }}>
                  {explanation.sources.map((src) => (
                    <span key={src} className="badge badge-source" style={{ textTransform: 'capitalize' }}>
                      {src}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Natural Explanation Card */}
            <div style={{
              background: 'rgba(37, 99, 235, 0.08)',
              border: '1px solid rgba(37, 99, 235, 0.25)',
              borderRadius: 6,
              padding: '1rem 1.25rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#60a5fa', marginBottom: '0.4rem', fontSize: '0.85rem', fontWeight: 600 }}>
                <Sparkles size={16} />
                Recommendation Rationale
              </div>
              <p style={{ fontSize: '0.9rem', color: '#e2e8f0', lineHeight: 1.5 }}>
                {explanation.primary_reason}
              </p>
              {explanation.summary_text !== explanation.primary_reason && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.5rem' }}>
                  {explanation.summary_text}
                </p>
              )}
            </div>

            {/* Retrieval Channel Breakdown */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                <Layers size={16} />
                Multi-Channel Retrieval Scores
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {Object.entries(explanation.component_scores).map(([channel, val]) => (
                  <div key={channel}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                      <span style={{ textTransform: 'capitalize' }}>{channel}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{val.toFixed(3)}</span>
                    </div>
                    <div style={{ width: '100%', height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{
                        width: `${Math.min(100, Math.max(0, val * 100))}%`,
                        height: '100%',
                        background: channel === 'content' ? '#38bdf8' : channel === 'collaborative' ? '#a855f7' : '#f59e0b',
                        borderRadius: 3,
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Ranking Features Table */}
            {Object.keys(explanation.feature_breakdown).length > 0 && (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  <Sliders size={16} />
                  Calculated Ranking Features (LightGBM)
                </div>
                <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 6, overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem' }}>
                    <thead>
                      <tr style={{ background: 'var(--bg-surface)', textAlign: 'left', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '0.5rem 0.85rem' }}>Feature</th>
                        <th style={{ padding: '0.5rem 0.85rem', textAlign: 'right' }}>Normalized Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(explanation.feature_breakdown).map(([feat, val], idx) => (
                        <tr key={feat} style={{ borderTop: '1px solid var(--border-subtle)', background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                          <td style={{ padding: '0.45rem 0.85rem', fontFamily: 'var(--font-mono)', color: '#cbd5e1' }}>{feat}</td>
                          <td style={{ padding: '0.45rem 0.85rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                            {typeof val === 'number' ? val.toFixed(4) : val}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
};
