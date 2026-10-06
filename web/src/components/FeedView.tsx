import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Bookmark,
  Heart,
  EyeOff,
  ExternalLink,
  ChevronRight,
  Filter,
  Users,
} from 'lucide-react';
import { UserSummary, FeedItem, RecommendationExplanation } from '../types';
import { fetchUsers, fetchFeed, fetchExplanation, recordInteraction } from '../api/client';
import { RecommendationModal } from './RecommendationModal';

export const FeedView: React.FC = () => {
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('ranker');
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Interaction feedback tracking
  const [likedArticles, setLikedArticles] = useState<Set<string>>(new Set());
  const [savedArticles, setSavedArticles] = useState<Set<string>>(new Set());
  const [dismissedArticles, setDismissedArticles] = useState<Set<string>>(new Set());

  // Recommendation Inspector Modal
  const [selectedExplanation, setSelectedExplanation] = useState<RecommendationExplanation | null>(null);
  const [explLoading, setExplLoading] = useState<boolean>(false);

  // Load available users
  useEffect(() => {
    fetchUsers(20)
      .then((data) => {
        setUsers(data);
        if (data.length > 0) {
          setSelectedUser(data[0].user_id);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  // Fetch feed when user or model changes
  useEffect(() => {
    if (!selectedUser) return;
    setLoading(true);
    setError(null);
    fetchFeed(selectedUser, selectedModel, 20)
      .then((res) => {
        setFeed(res.items);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [selectedUser, selectedModel]);

  const handleInspect = async (recommendationId: string) => {
    setExplLoading(true);
    try {
      const expl = await fetchExplanation(recommendationId);
      setSelectedExplanation(expl);
    } catch (err: any) {
      alert(`Could not load explanation: ${err.message}`);
    } finally {
      setExplLoading(false);
    }
  };

  const handleInteraction = async (articleId: string, type: 'like' | 'save' | 'dismiss') => {
    if (type === 'like') {
      setLikedArticles((prev) => new Set(prev).add(articleId));
    } else if (type === 'save') {
      setSavedArticles((prev) => new Set(prev).add(articleId));
    } else if (type === 'dismiss') {
      setDismissedArticles((prev) => new Set(prev).add(articleId));
    }

    try {
      await recordInteraction(selectedUser, articleId, type);
    } catch {
      // Local state is preserved
    }
  };

  const models = [
    { id: 'ranker', label: 'LightGBM Ranker (Final)' },
    { id: 'hybrid', label: 'Hybrid Fusion' },
    { id: 'content', label: 'Content-Based (FAISS)' },
    { id: 'collaborative', label: 'Collaborative (ALS)' },
    { id: 'popularity', label: 'Popularity Baseline' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Control Bar: User Selector & Model Switcher */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 8,
        padding: '1.25rem',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
      }}>
        {/* User Selection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <Users size={16} />
            <span>Target User:</span>
          </div>
          <select
            value={selectedUser}
            onChange={(e) => setSelectedUser(e.target.value)}
            style={{
              background: 'var(--bg-surface)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-active)',
              padding: '0.45rem 0.75rem',
              borderRadius: 6,
              fontSize: '0.875rem',
              outline: 'none',
              cursor: 'pointer',
            }}
          >
            {users.map((u) => (
              <option key={u.user_id} value={u.user_id}>
                {u.user_id} ({u.history_count} reads • {u.top_categories.join(', ') || 'general'})
              </option>
            ))}
          </select>
        </div>

        {/* Model Switcher Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginRight: '0.25rem' }}>
            <Filter size={14} />
            <span>Engine:</span>
          </div>
          {models.map((m) => {
            const isActive = selectedModel === m.id;
            return (
              <button
                key={m.id}
                onClick={() => setSelectedModel(m.id)}
                style={{
                  fontSize: '0.8rem',
                  padding: '0.35rem 0.65rem',
                  borderRadius: 5,
                  fontWeight: 500,
                  transition: 'all 0.15s ease',
                  background: isActive ? '#2563eb' : 'var(--bg-surface)',
                  color: isActive ? '#ffffff' : 'var(--text-secondary)',
                  border: isActive ? '1px solid #3b82f6' : '1px solid var(--border-subtle)',
                }}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Feed List */}
      {error ? (
        <div style={{
          background: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          color: '#f43f5e',
          padding: '1rem',
          borderRadius: 6,
        }}>
          Error loading feed: {error}
        </div>
      ) : loading ? (
        <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Retrieving personalized candidates and reranking...
        </div>
      ) : feed.length === 0 ? (
        <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          No articles retrieved for this user.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {feed
            .filter((item) => !dismissedArticles.has(item.article_id))
            .map((item) => {
              const isLiked = likedArticles.has(item.article_id);
              const isSaved = savedArticles.has(item.article_id);

              return (
                <div
                  key={item.recommendation_id}
                  style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 8,
                    padding: '1.25rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem',
                    transition: 'border-color 0.15s ease',
                  }}
                >
                  {/* Article Card Header */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <span className="badge badge-rank">#{item.rank}</span>
                      <span className="badge badge-category">{item.category}</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        / {item.subcategory}
                      </span>
                      {item.sources.map((src) => (
                        <span key={src} className="badge badge-source" style={{ textTransform: 'capitalize' }}>
                          {src}
                        </span>
                      ))}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontSize: '0.775rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        score: <strong style={{ color: '#38bdf8' }}>{item.score.toFixed(4)}</strong>
                      </span>
                    </div>
                  </div>

                  {/* Title & Abstract */}
                  <div>
                    <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.35rem', lineHeight: 1.35 }}>
                      {item.title}
                    </h3>
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', lineHeight: 1.5 }}>
                      {item.abstract || 'No abstract available for this article.'}
                    </p>
                  </div>

                  {/* Recommendation Primary Reason Callout */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.55rem 0.85rem',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: 6,
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.8rem', color: '#cbd5e1' }}>
                      <Sparkles size={14} color="#60a5fa" />
                      <span>{item.primary_reason}</span>
                    </div>

                    <button
                      onClick={() => handleInspect(item.recommendation_id)}
                      className="btn-secondary"
                      style={{ fontSize: '0.775rem', padding: '0.3rem 0.6rem' }}
                    >
                      <span>Inspect Why</span>
                      <ChevronRight size={13} />
                    </button>
                  </div>

                  {/* Interactive Action Bar */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderTop: '1px solid var(--border-subtle)',
                    paddingTop: '0.75rem',
                    marginTop: '0.25rem',
                  }}>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        onClick={() => handleInteraction(item.article_id, 'like')}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          fontSize: '0.8rem',
                          color: isLiked ? '#f43f5e' : 'var(--text-muted)',
                          padding: '0.35rem 0.65rem',
                          borderRadius: 4,
                          background: isLiked ? 'rgba(244, 63, 94, 0.1)' : 'transparent',
                        }}
                      >
                        <Heart size={15} fill={isLiked ? '#f43f5e' : 'none'} />
                        <span>{isLiked ? 'Liked' : 'Like'}</span>
                      </button>

                      <button
                        onClick={() => handleInteraction(item.article_id, 'save')}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          fontSize: '0.8rem',
                          color: isSaved ? '#38bdf8' : 'var(--text-muted)',
                          padding: '0.35rem 0.65rem',
                          borderRadius: 4,
                          background: isSaved ? 'rgba(56, 189, 248, 0.1)' : 'transparent',
                        }}
                      >
                        <Bookmark size={15} fill={isSaved ? '#38bdf8' : 'none'} />
                        <span>{isSaved ? 'Saved' : 'Save'}</span>
                      </button>

                      <button
                        onClick={() => handleInteraction(item.article_id, 'dismiss')}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          fontSize: '0.8rem',
                          color: 'var(--text-muted)',
                          padding: '0.35rem 0.65rem',
                          borderRadius: 4,
                        }}
                      >
                        <EyeOff size={15} />
                        <span>Dismiss</span>
                      </button>
                    </div>

                    {item.url && (
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.3rem',
                          fontSize: '0.775rem',
                          color: 'var(--text-muted)',
                        }}
                      >
                        <span>Original News</span>
                        <ExternalLink size={13} />
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      )}

      {/* Recommendation Inspector Modal */}
      <RecommendationModal
        explanation={selectedExplanation}
        loading={explLoading}
        onClose={() => setSelectedExplanation(null)}
      />
    </div>
  );
};
