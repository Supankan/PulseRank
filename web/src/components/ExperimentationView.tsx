import React, { useState } from 'react';
import { computeSampleSize, runBootstrapTest } from '../api/client';

export const ExperimentationView: React.FC = () => {
  // Sizing state
  const [baselineCtr, setBaselineCtr] = useState(0.04);
  const [mdeRelative, setMdeRelative] = useState(0.05);
  const [alpha, setAlpha] = useState(0.05);
  const [power, setPower] = useState(0.80);
  const [sampleResult, setSampleResult] = useState<any>(null);
  const [loadingSize, setLoadingSize] = useState(false);

  // Bootstrap state
  const [modelA, setModelA] = useState('hybrid');
  const [modelB, setModelB] = useState('ranker');
  const [bootstraps, setBootstraps] = useState(200);
  const [bootstrapResult, setBootstrapResult] = useState<any>(null);
  const [loadingBoot, setLoadingBoot] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleComputeSampleSize = async () => {
    setLoadingSize(true);
    setError(null);
    try {
      const res = await computeSampleSize(baselineCtr, mdeRelative, alpha, power);
      setSampleResult(res);
    } catch (err: any) {
      setError(err.message || 'Sample size calculation failed');
    } finally {
      setLoadingSize(false);
    }
  };

  const handleRunBootstrap = async () => {
    setLoadingBoot(true);
    setError(null);
    try {
      const res = await runBootstrapTest(modelA, modelB, bootstraps);
      setBootstrapResult(res);
    } catch (err: any) {
      setError(err.message || 'Bootstrap testing failed');
    } finally {
      setLoadingBoot(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Experimentation & Statistical Testing</h2>
        <p style={{ margin: '0.25rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          Plan sample sizes with two-proportion z-tests and run non-parametric paired bootstrap comparisons on impression metrics.
        </p>
      </div>

      {error && (
        <div style={{ padding: '0.75rem 1rem', background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
        {/* A/B Sample Size Calculator */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h3 style={{ margin: '0 0 1rem', fontSize: '1.1rem', fontWeight: 600 }}>A/B Test Sample Size Sizing</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                Baseline Click-Through Rate (CTR): {(baselineCtr * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min={0.01}
                max={0.20}
                step={0.005}
                value={baselineCtr}
                onChange={(e) => setBaselineCtr(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                Minimum Detectable Effect (MDE): {(mdeRelative * 100).toFixed(1)}% relative lift
              </label>
              <input
                type="range"
                min={0.01}
                max={0.25}
                step={0.01}
                value={mdeRelative}
                onChange={(e) => setMdeRelative(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Significance (?)</label>
                <select
                  value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0.35rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)' }}
                >
                  <option value={0.05}>0.05 (95% Conf)</option>
                  <option value={0.01}>0.01 (99% Conf)</option>
                  <option value={0.10}>0.10 (90% Conf)</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Power (1 - ?)</label>
                <select
                  value={power}
                  onChange={(e) => setPower(parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0.35rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)' }}
                >
                  <option value={0.80}>0.80 (80% Power)</option>
                  <option value={0.90}>0.90 (90% Power)</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleComputeSampleSize}
              disabled={loadingSize}
              style={{ padding: '0.65rem 1rem', background: 'var(--primary)', color: '#fff', border: 'none', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', marginTop: '0.5rem' }}
            >
              {loadingSize ? 'Computing...' : 'Calculate Sample Size'}
            </button>

            {sampleResult && (
              <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Required Sample Size per Variant:</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>
                  {sampleResult.required_sample_size_per_variant.toLocaleString()} users
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Total A/B test population: {sampleResult.total_sample_size.toLocaleString()} impressions to detect target CTR of {((sampleResult.baseline_ctr + sampleResult.mde_absolute) * 100).toFixed(2)}%.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Paired Bootstrap Comparison */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem', padding: '1.5rem' }}>
          <h3 style={{ margin: '0 0 1rem', fontSize: '1.1rem', fontWeight: 600 }}>Paired Bootstrap Hypothesis Testing</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Model A (Control)</label>
                <select
                  value={modelA}
                  onChange={(e) => setModelA(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0.35rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)' }}
                >
                  <option value="hybrid">Pointwise Hybrid</option>
                  <option value="content">Content-Based</option>
                  <option value="collaborative">Collaborative ALS</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Model B (Treatment)</label>
                <select
                  value={modelB}
                  onChange={(e) => setModelB(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '0.35rem', border: '1px solid var(--border-subtle)', background: 'var(--bg-card)', color: 'var(--text-main)' }}
                >
                  <option value="ranker">LightGBM LambdaMART</option>
                  <option value="hybrid">Pointwise Hybrid</option>
                </select>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                Bootstrap Resampling Iterations: {bootstraps}
              </label>
              <input
                type="range"
                min={100}
                max={500}
                step={50}
                value={bootstraps}
                onChange={(e) => setBootstraps(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <button
              onClick={handleRunBootstrap}
              disabled={loadingBoot}
              style={{ padding: '0.65rem 1rem', background: 'var(--surface-hover)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', marginTop: '0.5rem' }}
            >
              {loadingBoot ? 'Resampling...' : 'Run Paired Bootstrap'}
            </button>

            {bootstrapResult && (
              <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {Object.entries(bootstrapResult.metrics || {}).map(([key, m]: [string, any]) => (
                  <div key={key} style={{ padding: '0.75rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '0.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{m.metric_name}</span>
                      <span style={{
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        padding: '0.2rem 0.5rem',
                        borderRadius: '0.25rem',
                        background: m.is_statistically_significant ? '#d1fae5' : '#f3f4f6',
                        color: m.is_statistically_significant ? '#065f46' : '#6b7280',
                      }}>
                        {m.is_statistically_significant ? 'Statistically Significant (p < 0.05)' : 'Not Significant'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <span>Model A: {m.model_a_mean.toFixed(4)} ? Model B: {m.model_b_mean.toFixed(4)}</span>
                      <span style={{ fontWeight: 700, color: m.delta_mean >= 0 ? '#10b981' : '#ef4444' }}>
                        ?: {m.delta_mean >= 0 ? `+${m.delta_mean.toFixed(4)}` : m.delta_mean.toFixed(4)}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      95% CI: [{m.ci_lower_95.toFixed(4)}, {m.ci_upper_95.toFixed(4)}] ? p-val: {m.p_value.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
