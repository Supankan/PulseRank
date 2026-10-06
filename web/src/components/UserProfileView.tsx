import React, { useState, useEffect } from 'react';
import { User, Tag, Database, BookOpen, ExternalLink, Flame } from 'lucide-react';
import { UserSummary, UserProfile } from '../types';
import { fetchUsers, fetchUserProfile } from '../api/client';

export const UserProfileView: React.FC = () => {
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchUsers(50)
      .then((data) => {
        setUsers(data);
        if (data.length > 0) {
          setSelectedUser(data[0].user_id);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedUser) return;
    setLoading(true);
    fetchUserProfile(selectedUser)
      .then((data) => {
        setProfile(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [selectedUser]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* User Selector Header */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 8,
        padding: '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <User size={20} color="#60a5fa" />
          <h2 style={{ fontSize: '1.15rem', fontWeight: 600 }}>User Interest Intelligence</h2>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Select User:</span>
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
                {u.user_id} ({u.history_count} articles)
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Computing user interaction profile and entity graph...
        </div>
      ) : error ? (
        <div style={{ padding: '1.5rem', color: '#f43f5e' }}>Error loading profile: {error}</div>
      ) : profile ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {/* Left Column: Stats & Category Distributions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Summary Card */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              padding: '1.25rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#38bdf8', marginBottom: '0.5rem', fontSize: '0.85rem', fontWeight: 600 }}>
                <Flame size={16} />
                Inferred Reading Persona
              </div>
              <p style={{ fontSize: '0.95rem', color: '#e2e8f0', lineHeight: 1.5, marginBottom: '1rem' }}>
                {profile.semantic_summary}
              </p>
              <div style={{ display: 'flex', gap: '1.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Consumed</span>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                    {profile.history_size}
                  </div>
                </div>
                <div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Dominant Topic</span>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, textTransform: 'capitalize', color: '#60a5fa' }}>
                    {profile.top_categories[0] || 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Category Preferences Breakdown */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              padding: '1.25rem',
            }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '1rem' }}>
                Category Distribution
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {Object.entries(profile.category_distribution).map(([cat, ratio]) => {
                  const pct = Math.round(ratio * 100);
                  return (
                    <div key={cat}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', marginBottom: '0.3rem' }}>
                        <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>{cat}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{pct}%</span>
                      </div>
                      <div style={{ width: '100%', height: 7, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: '#3b82f6',
                          borderRadius: 4,
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Subcategories */}
            {Object.keys(profile.subcategory_distribution).length > 0 && (
              <div style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 8,
                padding: '1.25rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.85rem', fontWeight: 600 }}>
                  <Tag size={15} />
                  Top Subcategories
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                  {Object.entries(profile.subcategory_distribution).map(([subcat, ratio]) => (
                    <span key={subcat} className="badge badge-category" style={{ fontSize: '0.75rem', padding: '0.3rem 0.55rem' }}>
                      {subcat} ({Math.round(ratio * 100)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Entities & Recent Interaction History */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Interacted Entities */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              padding: '1.25rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.85rem', fontWeight: 600 }}>
                <Database size={15} />
                Knowledge Graph Entities (Wikidata)
              </div>
              {profile.top_entities.length === 0 ? (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No explicit entities detected in reading history.</p>
              ) : (
                <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 6, overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem' }}>
                    <thead>
                      <tr style={{ background: 'var(--bg-surface)', textAlign: 'left', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '0.5rem 0.75rem' }}>Entity / Topic</th>
                        <th style={{ padding: '0.5rem 0.75rem' }}>Category</th>
                        <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Clicks</th>
                      </tr>
                    </thead>
                    <tbody>
                      {profile.top_entities.map((e, idx) => (
                        <tr key={idx} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                          <td style={{ padding: '0.45rem 0.75rem', fontWeight: 500, color: '#f8fafc' }}>
                            {e.name}
                            {e.wikidata_id && (
                              <a
                                href={`https://www.wikidata.org/wiki/${e.wikidata_id}`}
                                target="_blank"
                                rel="noreferrer"
                                style={{ marginLeft: '0.35rem', color: '#60a5fa' }}
                              >
                                <ExternalLink size={10} style={{ display: 'inline' }} />
                              </a>
                            )}
                          </td>
                          <td style={{ padding: '0.45rem 0.75rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                            {e.category || 'general'}
                          </td>
                          <td style={{ padding: '0.45rem 0.75rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                            {e.count}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Recent Consumed Articles */}
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              padding: '1.25rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', fontSize: '0.85rem', fontWeight: 600 }}>
                <BookOpen size={15} />
                Recent Reading History
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {profile.recent_articles.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '0.65rem 0.85rem',
                      background: 'rgba(255, 255, 255, 0.02)',
                      borderRadius: 6,
                      border: '1px solid rgba(255, 255, 255, 0.05)',
                    }}
                  >
                    <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.2rem' }}>
                      <span className="badge badge-category" style={{ fontSize: '0.7rem' }}>
                        {item.category}
                      </span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        {item.subcategory}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#f1f5f9', fontWeight: 500 }}>
                      {item.title}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
