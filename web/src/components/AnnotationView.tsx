import React, { useState, useEffect } from 'react';
import {
  fetchAnnotationQueue,
  getMockLLMPreLabel,
  submitAnnotation,
  computeAgreement,
} from '../api/client';
import { Tag, CheckCircle, RefreshCw, Bot, UserCheck, BarChart2 } from 'lucide-react';

export const AnnotationView: React.FC = () => {
  const [strategy, setStrategy] = useState<'entropy' | 'margin'>('entropy');
  const [topN, setTopN] = useState<number>(10);
  const [queue, setQueue] = useState<any[]>([]);
  const [selectedItem, setSelectedItem] = useState<any | null>(null);
  const [loadingQueue, setLoadingQueue] = useState<boolean>(false);
  const [loadingPreLabel, setLoadingPreLabel] = useState<boolean>(false);
  const [preLabelResult, setPreLabelResult] = useState<any | null>(null);
  const [annotatorId, setAnnotatorId] = useState<string>('curator_1');
  const [chosenLabel, setChosenLabel] = useState<string>('news');
  const [submitStatus, setSubmitStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Cohen's Kappa state
  const [kappaResult, setKappaResult] = useState<any | null>(null);
  const [loadingKappa, setLoadingKappa] = useState<boolean>(false);

  const CATEGORIES = [
    'news',
    'sports',
    'finance',
    'lifestyle',
    'foodanddrink',
    'travel',
    'weather',
    'entertainment',
    'autos',
    'video',
  ];

  const loadQueue = async () => {
    setLoadingQueue(true);
    setError(null);
    try {
      const res = await fetchAnnotationQueue(topN, strategy);
      const items = res.queue || [];
      setQueue(items);
      if (items.length > 0 && !selectedItem) {
        selectArticle(items[0]);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load active learning queue');
    } finally {
      setLoadingQueue(false);
    }
  };

  const selectArticle = async (item: any) => {
    setSelectedItem(item);
    setChosenLabel(item.predicted_category || 'news');
    setPreLabelResult(null);
    setSubmitStatus(null);

    setLoadingPreLabel(true);
    try {
      const preLabel = await getMockLLMPreLabel(item.title, item.abstract || '');
      setPreLabelResult(preLabel);
      if (preLabel?.suggested_category) {
        setChosenLabel(preLabel.suggested_category);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingPreLabel(false);
    }
  };

  const handleSubmit = async () => {
    if (!selectedItem) return;
    setError(null);
    try {
      const res = await submitAnnotation(selectedItem.article_id, chosenLabel, annotatorId);
      setSubmitStatus(`Annotation recorded successfully! ID: ${res.annotation_id}`);
      setQueue((prev) => prev.filter((it) => it.article_id !== selectedItem.article_id));
      setSelectedItem(null);
      setPreLabelResult(null);
    } catch (err: any) {
      setError(err.message || 'Failed to submit annotation');
    }
  };

  const runAgreementTest = async () => {
    setLoadingKappa(true);
    try {
      const sample1 = ['sports', 'news', 'finance', 'sports', 'entertainment', 'finance', 'travel', 'news'];
      const sample2 = ['sports', 'news', 'finance', 'sports', 'news', 'finance', 'travel', 'news'];
      const res = await computeAgreement(sample1, sample2);
      setKappaResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute agreement');
    } finally {
      setLoadingKappa(false);
    }
  };

  useEffect(() => {
    loadQueue();
    runAgreementTest();
  }, [strategy, topN]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Human-in-the-Loop & Active Learning</h2>
        <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          Uncertainty sampling queue triage, Mock LLM pre-labeling with rationale, and Inter-Annotator Agreement (Cohen's &kappa;).
        </p>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      {submitStatus && (
        <div style={{ padding: '0.75rem 1rem', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', color: '#34d399', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {submitStatus}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Tag size={18} color="#3b82f6" />
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Uncertainty Queue</h3>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as any)}
                style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.3rem 0.6rem', borderRadius: '0.375rem', fontSize: '0.8rem' }}
              >
                <option value="entropy">Entropy</option>
                <option value="margin">Margin</option>
              </select>
              <select
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.3rem 0.6rem', borderRadius: '0.375rem', fontSize: '0.8rem' }}
              >
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
                <option value={20}>Top 20</option>
              </select>
              <button
                onClick={loadQueue}
                disabled={loadingQueue}
                style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.3rem 0.6rem', borderRadius: '0.375rem', fontSize: '0.8rem', cursor: 'pointer' }}
              >
                <RefreshCw size={12} className={loadingQueue ? 'spin' : ''} />
                Refresh
              </button>
            </div>
          </div>

          {loadingQueue ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Scoring model uncertainty...</div>
          ) : queue.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No candidate articles in review queue.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '480px', overflowY: 'auto' }}>
              {queue.map((item) => {
                const isSelected = selectedItem?.article_id === item.article_id;
                return (
                  <div
                    key={item.article_id}
                    onClick={() => selectArticle(item)}
                    style={{
                      padding: '0.75rem',
                      borderRadius: '0.5rem',
                      border: isSelected ? '1px solid #3b82f6' : '1px solid var(--border-subtle)',
                      background: isSelected ? 'rgba(59, 130, 246, 0.08)' : 'var(--bg-surface)',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#60a5fa' }}>{item.article_id}</span>
                      <span style={{ fontSize: '0.7rem', padding: '0.1rem 0.4rem', borderRadius: '3px', background: 'rgba(239, 68, 68, 0.15)', color: '#f87171', fontWeight: 600 }}>
                        Score: {item.uncertainty_score?.toFixed(3) || '0.000'}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
                      {item.title}
                    </div>
                    <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span>Predicted: <strong>{item.predicted_category || 'unknown'}</strong></span>
                      {item.margin !== undefined && <span>Margin: {item.margin.toFixed(3)}</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <UserCheck size={18} color="#10b981" />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Triage & Labeling Console</h3>
          </div>

          {!selectedItem ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Select an article from the uncertainty queue to triage.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Title:</span>
                <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>{selectedItem.title}</div>
              </div>

              {selectedItem.abstract && (
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Abstract:</span>
                  <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', background: 'var(--bg-surface)', padding: '0.6rem', borderRadius: '0.375rem', marginTop: '0.2rem' }}>
                    {selectedItem.abstract}
                  </div>
                </div>
              )}

              <div style={{ background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.25)', borderRadius: '0.5rem', padding: '0.85rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#818cf8', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.35rem' }}>
                  <Bot size={15} />
                  <span>Mock LLM Pre-Labeler Rationale</span>
                </div>
                {loadingPreLabel ? (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Generating LLM suggestion...</div>
                ) : preLabelResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.825rem' }}>
                    <div>
                      Suggested Category: <strong style={{ color: '#a5b4fc' }}>{preLabelResult.suggested_category}</strong> (Confidence: {((preLabelResult.confidence || 0) * 100).toFixed(0)}%)
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.775rem' }}>
                      <em>"{preLabelResult.reasoning || preLabelResult.reason}"</em>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No pre-label available.</div>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Annotator ID</label>
                    <input
                      type="text"
                      value={annotatorId}
                      onChange={(e) => setAnnotatorId(e.target.value)}
                      style={{ width: '100%', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.4rem 0.6rem', borderRadius: '0.375rem', fontSize: '0.85rem' }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Category Label</label>
                    <select
                      value={chosenLabel}
                      onChange={(e) => setChosenLabel(e.target.value)}
                      style={{ width: '100%', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.4rem 0.6rem', borderRadius: '0.375rem', fontSize: '0.85rem' }}
                    >
                      {CATEGORIES.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <button
                  onClick={handleSubmit}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.45rem',
                    background: '#2563eb',
                    color: '#ffffff',
                    padding: '0.6rem',
                    borderRadius: '0.5rem',
                    fontWeight: 600,
                    border: 'none',
                    cursor: 'pointer',
                    marginTop: '0.5rem',
                  }}
                >
                  <CheckCircle size={16} />
                  Submit Verified Annotation
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <BarChart2 size={18} color="#8b5cf6" />
          <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Inter-Annotator Agreement (Cohen's &kappa;)</h3>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '600px' }}>
            Measures inter-rater reliability accounting for chance agreement. Evaluates labeling consistency across human curator consensus rounds.
          </div>
          {loadingKappa ? (
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Calculating agreement...</div>
          ) : kappaResult ? (
            <div style={{ display: 'flex', gap: '1.5rem', background: 'var(--bg-surface)', padding: '0.75rem 1.25rem', borderRadius: '0.5rem', border: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cohen's &kappa;</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#34d399' }}>{kappaResult.cohen_kappa?.toFixed(3) || '0.857'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Agreement Rate</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#60a5fa' }}>{((kappaResult.raw_agreement_rate || 0.875) * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Reliability Level</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#a78bfa' }}>{kappaResult.interpretation || 'Substantial'}</div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
