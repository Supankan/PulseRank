import React, { useState, useEffect } from 'react';
import { Navbar, NavTab } from './components/Navbar';
import { FeedView } from './components/FeedView';
import { UserProfileView } from './components/UserProfileView';
import { ModelDashboardView } from './components/ModelDashboardView';
import { ContentIntelligenceView } from './components/ContentIntelligenceView';
import { AblationView } from './components/AblationView';
import { ExperimentationView } from './components/ExperimentationView';
import { ColdStartView } from './components/ColdStartView';
import { MonitoringView } from './components/MonitoringView';
import { AnnotationView } from './components/AnnotationView';
import { HealthStatus } from './types';
import { fetchHealth } from './api/client';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('feed');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const checkHealth = () => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Navbar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        health={health}
      />

      <main className="main-content" style={{ maxWidth: 1400, margin: '0 auto', padding: '1.5rem 1.25rem' }}>
        {currentTab === 'feed' && <FeedView />}
        {currentTab === 'profile' && <UserProfileView />}
        {currentTab === 'models' && <ModelDashboardView />}
        {currentTab === 'nlp' && <ContentIntelligenceView />}
        {currentTab === 'ablations' && <AblationView />}
        {currentTab === 'experiments' && <ExperimentationView />}
        {currentTab === 'coldstart' && <ColdStartView />}
        {currentTab === 'monitoring' && <MonitoringView />}
        {currentTab === 'annotations' && <AnnotationView />}
      </main>

      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '1.25rem',
        textAlign: 'center',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
      }}>
        PulseRank • Production Multi-Stage Recommendation & Content Intelligence Platform • Microsoft News Dataset (MIND-small)
      </footer>
    </div>
  );
};
export default App;
