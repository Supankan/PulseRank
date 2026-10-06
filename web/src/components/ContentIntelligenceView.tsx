import React, { useState } from 'react';
import { classifyText, extractEntities } from '../api/client';

export const ContentIntelligenceView: React.FC = () => {
  const [title, setTitle] = useState('NASA Rover Discovers Evidence of Ancient Riverbed on Mars');
  const [abstract, setAbstract] = useState('Scientists at the Jet Propulsion Laboratory confirmed geological formations indicate sustained liquid water billions of years ago.');
  const [classification, setClassification] = useState<any>(null);
  const [entities, setEntities] = useState<any[]>([]);
  const [loadingClf, setLoadingClf] = useState(false);
  const [loadingNer, setLoadingNer] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = async () => {
    setLoadingClf(true);
    setError(null);
    try {
      const res = await classifyText(title, abstract, 'baseline');
      setClassification(res);
    } catch (err: any) {
      setError(err.message || 'Classification failed');
    } finally {
      setLoadingClf(false);
    }
  };

  const handleExtractEntities = async () => {
    setLoadingNer(true);
    setError(null);
    try {
      const fullText = `${title} ${abstract}`;
      const res = await extractEntities(fullText);
      setEntities(res.entities || []);
    } catch (err: any) {
      setError(err.message || 'Entity extraction failed');
    } finally {
      setLoadingNer(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Content Intelligence Studio</h2>
          <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Evaluate topic categorisation models and Named Entity Recognition (NER) with Wikidata resolution.
          </p>
        </div>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 600 }}>Article Content Input</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              style={{ width: '100%', padding: '0.65rem 0.85rem', borderRadius: '0.5rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)', fontSize: '0.9rem' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Abstract</label>
            <textarea
              rows={3}
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              style={{ width: '100%', padding: '0.65rem 0.85rem', borderRadius: '0.5rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)', fontSize: '0.9rem', resize: 'vertical' }}
            />
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              onClick={handleClassify}
              disabled={loadingClf}
              style={{ padding: '0.65rem 1.25rem', background: 'var(--primary)', color: '#fff', border: 'none', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.875rem' }}
            >
              {loadingClf ? 'Classifying...' : 'Predict Topic Class'}
            </button>
            <button
              onClick={handleExtractEntities}
              disabled={loadingNer}
              style={{ padding: '0.65rem 1.25rem', background: 'var(--surface-hover)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.875rem' }}
            >
              {loadingNer ? 'Extracting...' : 'Extract Named Entities'}
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        {/* Classification Result */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 600 }}>Topic Classification Output</h3>
          {classification ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Top Prediction:</span>
                <span style={{ padding: '0.35rem 0.75rem', background: 'var(--primary-subtle)', color: 'var(--primary)', borderRadius: '9999px', fontWeight: 700, fontSize: '0.9rem', textTransform: 'capitalize' }}>
                  {classification.predicted_category} ({(classification.confidence * 100).toFixed(1)}%)
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Calibrated Probabilities:</span>
                {classification.top_k_probabilities?.map((p: any) => (
                  <div key={p.category}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.2rem' }}>
                      <span style={{ textTransform: 'capitalize' }}>{p.category}</span>
                      <span>{(p.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '6px', background: 'var(--border-subtle)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${Math.min(p.probability * 100, 100)}%`, background: 'var(--primary)', borderRadius: '3px' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Click 'Predict Topic Class' to test the topic classifier baseline.</p>
          )}
        </div>

        {/* NER Result */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 600 }}>Extracted Named Entities ({entities.length})</h3>
          {entities.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {entities.map((e: any, idx: number) => {
                const badgeColor = e.label === 'ORG' ? '#3b82f6' : e.label === 'LOC' ? '#10b981' : e.label === 'PER' ? '#8b5cf6' : '#f59e0b';
                return (
                  <div
                    key={idx}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      padding: '0.35rem 0.65rem',
                      borderRadius: '0.5rem',
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-card)',
                      fontSize: '0.8rem',
                    }}
                  >
                    <span style={{ fontWeight: 600 }}>{e.text}</span>
                    <span style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem', background: badgeColor, color: '#fff', borderRadius: '0.25rem', fontWeight: 700 }}>
                      {e.label}
                    </span>
                    {e.wikidata_id && (
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', background: 'var(--surface-hover)', padding: '0.1rem 0.3rem', borderRadius: '0.25rem' }}>
                        {e.wikidata_id}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Click 'Extract Named Entities' to identify organizations, places, and people.</p>
          )}
        </div>
      </div>
    </div>
  );
};
