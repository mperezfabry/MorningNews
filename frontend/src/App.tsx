import { useEffect, useState } from 'react';
import { Bot, Newspaper, Mic, TrendingUp } from 'lucide-react';
import { fetchArticles, fetchDailySummaries } from './lib/db';
import type { Article, DailySummary } from './lib/db';
import SmartFeed from './components/SmartFeed';
import DailyBrief from './components/DailyBrief';
import Trends from './components/Trends';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);
  const [articles, setArticles] = useState<Article[]>([]);
  const [summaries, setSummaries] = useState<DailySummary[]>([]);
  const [activeTab, setActiveTab] = useState<'feed' | 'brief' | 'trends'>('feed');
  
  // Filters state
  const [watchlist, setWatchlist] = useState('');
  const [positiveOnly, setPositiveOnly] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const loadData = async () => {
      try {
        const [arts, sums] = await Promise.all([
          fetchArticles(),
          fetchDailySummaries()
        ]);
        setArticles(arts);
        setSummaries(sums);
      } catch (err) {
        console.error("Error loading data:", err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <h2>Initializing Serverless AI...</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Downloading MorningNews DB</p>
      </div>
    );
  }

  // Derived filter logic
  const filteredArticles = articles.filter(a => {
    if (positiveOnly && (a.sentiment_score ?? 0.5) < 0.6) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      if (!a.title.toLowerCase().includes(q) && !(a.ai_summary || '').toLowerCase().includes(q)) {
        return false;
      }
    }
    return true;
  });

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <Bot size={28} color="var(--accent-primary)" />
          MorningNews AI
        </div>
        <p className="brand-caption">Serverless edge intelligence.</p>

        <div className="sidebar-section">
          <h3>Personalize</h3>
          <div className="input-group">
            <label>Watchlist (comma separated)</label>
            <input 
              type="text" 
              className="text-input" 
              placeholder="Crypto, AI, SpaceX"
              value={watchlist}
              onChange={(e) => setWatchlist(e.target.value)}
            />
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Filters</h3>
          <div className="input-group">
            <label>Search Feed</label>
            <input 
              type="text" 
              className="text-input" 
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <label className="checkbox-label">
            <input 
              type="checkbox" 
              checked={positiveOnly}
              onChange={(e) => setPositiveOnly(e.target.checked)}
            />
            Positive sentiment only ({'>='} 0.6)
          </label>
        </div>

        <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '2rem 0' }} />
        
        <button 
          className="btn" 
          style={{ width: '100%', justifyContent: 'center', marginBottom: '1rem' }}
          title="While it is possible to update daily, automatically, or at any more frequent interval, the system is a demo so we have chosen to trigger updates manually to save costs."
          onClick={() => alert("Update triggered! (Simulated for serverless frontend)")}
        >
          Update News Feed
        </button>

        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
          Created by Michael Perez, Ray Odian-Floyd, and Mark Crisci.
        </p>

      </aside>

      <main className="main-content">
        <div className="tabs">
          <button 
            className={`tab-btn ${activeTab === 'feed' ? 'active' : ''}`}
            onClick={() => setActiveTab('feed')}
          >
            <Newspaper size={20} /> Smart Feed
          </button>
          <button 
            className={`tab-btn ${activeTab === 'brief' ? 'active' : ''}`}
            onClick={() => setActiveTab('brief')}
          >
            <Mic size={20} /> Daily Brief
          </button>
          <button 
            className={`tab-btn ${activeTab === 'trends' ? 'active' : ''}`}
            onClick={() => setActiveTab('trends')}
          >
            <TrendingUp size={20} /> Trends
          </button>
        </div>

        {activeTab === 'feed' && <SmartFeed articles={filteredArticles} watchlist={watchlist} />}
        {activeTab === 'brief' && <DailyBrief summaries={summaries} />}
        {activeTab === 'trends' && <Trends articles={filteredArticles} />}
      </main>
    </div>
  );
}

export default App;
