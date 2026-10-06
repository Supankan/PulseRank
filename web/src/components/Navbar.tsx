import React from 'react';
import {
  Activity,
  Layers,
  User,
  Cpu,
  Sparkles,
  FlaskConical,
  GitCompare,
  Gauge,
  UserPlus,
  FileCheck,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { HealthStatus } from '../types';

export type NavTab =
  | 'feed'
  | 'profile'
  | 'models'
  | 'nlp'
  | 'ablations'
  | 'experiments'
  | 'coldstart'
  | 'monitoring'
  | 'annotations';

interface NavbarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  health: HealthStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab, health }) => {
  const navItems: Array<{ id: NavTab; label: string; icon: React.ReactNode }> = [
    { id: 'feed', label: 'Feed', icon: <Layers size={15} /> },
    { id: 'profile', label: 'Profiles', icon: <User size={15} /> },
    { id: 'models', label: 'Benchmarks', icon: <Cpu size={15} /> },
    { id: 'nlp', label: 'Content AI', icon: <Sparkles size={15} /> },
    { id: 'ablations', label: 'Ablation Lab', icon: <FlaskConical size={15} /> },
    { id: 'experiments', label: 'A/B Stats', icon: <GitCompare size={15} /> },
    { id: 'coldstart', label: 'Cold-Start', icon: <UserPlus size={15} /> },
    { id: 'monitoring', label: 'Monitoring', icon: <Gauge size={15} /> },
    { id: 'annotations', label: 'Active Learning', icon: <FileCheck size={15} /> },
  ];

  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(11, 15, 25, 0.95)',
      backdropFilter: 'blur(8px)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <div style={{
        maxWidth: 1400,
        margin: '0 auto',
        padding: '0.65rem 1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem',
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #1e3a8a, #2563eb)',
            width: 32,
            height: 32,
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
          }}>
            <Activity size={18} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ fontWeight: 700, fontSize: '1.05rem', letterSpacing: '-0.01em' }}>PulseRank</span>
              <span style={{ fontSize: '0.65rem', color: '#60a5fa', background: 'rgba(37,99,235,0.15)', padding: '0.1rem 0.35rem', borderRadius: 3, fontWeight: 600 }}>PORTFOLIO PASS 2</span>
            </div>
            <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', margin: 0 }}>Content Intelligence & Multi-Stage RecSys</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
          {navItems.map((item) => {
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  padding: '0.35rem 0.65rem',
                  borderRadius: 6,
                  fontSize: '0.8rem',
                  fontWeight: isActive ? 600 : 500,
                  background: isActive ? 'var(--bg-surface)' : 'transparent',
                  color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                  border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Health */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
          {health?.status === 'healthy' ? (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'rgba(16, 185, 129, 0.1)',
              color: '#34d399',
              padding: '0.2rem 0.5rem',
              borderRadius: 4,
              border: '1px solid rgba(16, 185, 129, 0.25)',
            }}>
              <CheckCircle2 size={12} />
              <span>Pipeline Active</span>
            </div>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'rgba(245, 158, 11, 0.1)',
              color: '#fbbf24',
              padding: '0.2rem 0.5rem',
              borderRadius: 4,
              border: '1px solid rgba(245, 158, 11, 0.25)',
            }}>
              <AlertCircle size={12} />
              <span>Degraded / Initializing</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
