import React, { useState, useEffect } from 'react';
import { createOnboardingPersona, fetchColdStartSegments } from '../api/client';

export const ColdStartView: React.FC = () => {
  const [availableCategories] = useState(['sports', 'news', 'finance', 'technology', 'entertainment', 'lifestyle']);
  const [selectedCats, setSelectedCats] = useState<string[]>(['technology', 'sports']);
  const [onboardingResult, setOnboardingResult] = useState<any>(null);
  const [segments, setSegments] = useState<any>(null);
  const [loadingOnboard, setLoadingOnboard] = useState(false);
  const [loadingSegments, setLoadingSegments] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchColdStartSegments()
      .then(setSegments)
      .catch((err) => setError(err.message || 'Failed to load segment counts'))
      .finally(() => setLoadingSegments(false));
  }, []);

  const toggleCategory = (cat: string) => {
    if (selectedCats.includes(cat)) {
      setSelectedCats(selectedCats.filter((c) => c !== cat));
    } else {
      setSelectedCats([...selectedCats, cat]);
    }
  };

  const handleGeneratePersona = async () => {
    if (selectedCats.length === 0) {
      setError('Please select at least one topic category.');
      return;
    }
    setLoadingOnboard(true);
    setError(null);
    try {
      const res = await createOnboardingPersona(selectedCats);
      setOnboardingResult(res);
    } catch (err: any) {
      setError(err.message || 'Onboarding failed');
    } finally {
      setLoadingOnboard(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Cold-Start Studio</h2>
        <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          Simulate user onboarding personas and observe cold-item exploration boosts and segment distributions.
        </p>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      {/* Segment Metrics Bar */}
      {loadingSegments ? (
        <div style={{ padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Loading user segment diagnostics...</div>
      ) : segments ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Cold Users (|history| ? 2)</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{segments.cold_users_count}</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Warm Users (|history| &gt; 2)</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{segments.warm_users_count}</div>
          </div>
          <div style={{ padding: '1rem', background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Cold Item Impressions</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.25rem' }}>{segments.cold_item_impressions_count}</div>
          </div>
        </div>
      ) : null}

      {/* Interactive Onboarding Persona Builder */}
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
        <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.1rem', fontWeight: 600 }}>Onboarding Persona Generator</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', margin: '0 0 1.25rem' }}>
          Select target interest topics to synthesize initial category preference priors for a cold-start visitor:
        </p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.65rem', marginBottom: '1.25rem' }}>
          {availableCategories.map((cat) => {
            const isSelected = selectedCats.includes(cat);
            return (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                style={{
                  padding: '0.5rem 1rem',
                  borderRadius: '9999px',
                  border: isSelected ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'var(--primary-subtle)' : 'var(--bg-card)',
                  color: isSelected ? 'var(--primary)' : 'var(--text-main)',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                }}
              >
                {isSelected ? '? ' : '+ '}{cat}
              </button>
            );
          })}
        </div>

        <button
          onClick={handleGeneratePersona}
          disabled={loadingOnboard}
          style={{ padding: '0.65rem 1.25rem', background: 'var(--primary)', color: '#fff', border: 'none', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer' }}
        >
          {loadingOnboard ? 'Synthesizing...' : 'Generate Onboarding Slate'}
        </button>

        {onboardingResult && (
          <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem' }}>
            <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.95rem', fontWeight: 600 }}>Synthesized Initial Feed Slate</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              {onboardingResult.initial_slate?.map((item: any) => (
                <div key={item.article_id} style={{ padding: '1rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--primary)', fontWeight: 700, textTransform: 'uppercase' }}>
                    {item.category} / {item.subcategory}
                  </div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, margin: '0.35rem 0' }}>{item.title}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Prior Affinity: {(item.prior_weight * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
