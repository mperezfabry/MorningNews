import { useState, useMemo } from 'react';
import type { Article } from '../lib/db';
import { AlertTriangle, AlertCircle, CheckCircle } from 'lucide-react';

interface SmartFeedProps {
  articles: Article[];
  watchlist: string;
}

const PAGE_SIZE = 20;

export default function SmartFeed({ articles, watchlist }: SmartFeedProps) {
  const [page, setPage] = useState(1);

  const watchTerms = useMemo(() => {
    return watchlist.split(',').map(t => t.trim().toLowerCase()).filter(t => t);
  }, [watchlist]);

  const totalPages = Math.max(1, Math.ceil(articles.length / PAGE_SIZE));
  const pageData = articles.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Top Stories ({articles.length})</h2>
      
      {pageData.length === 0 && (
        <p>No articles found for current filters.</p>
      )}

      {pageData.map(article => {
        const isWatchlist = watchTerms.some(term => 
          article.title.toLowerCase().includes(term) || 
          (article.description || '').toLowerCase().includes(term)
        );

        const biasRaw = article.bias_score || 0;
        const biasClamped = Math.max(-1.0, Math.min(1.0, biasRaw));
        const biasPct = ((biasClamped + 1) / 2) * 100;

        const hasWarning = article.is_clickbait || article.misinformation_flag || article.extreme_bias_flag;

        return (
          <div key={article.id} className={`glass-panel article-card ${isWatchlist ? 'highlight-card' : ''}`}>
            {isWatchlist && (
              <div style={{ color: 'var(--accent-primary)', fontSize: '0.875rem', marginBottom: '0.5rem', fontWeight: 600 }}>
                👀 Matches your watchlist
              </div>
            )}
            <div className="article-header">
              <a href={article.url} target="_blank" rel="noreferrer" className="article-title">
                {hasWarning && '⚠️ '} {article.title}
              </a>
            </div>

            <div className="article-meta">
              <span className="badge provider">{article.source || 'Unknown'}</span>
              <span className="badge topic">{article.topic || 'Uncategorized'}</span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {new Date(article.published_at).toLocaleString(undefined, {
                  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                })}
              </span>

              {/* Backend specific AI Tags missing in streamlit */}
              {article.misinformation_flag && (
                <span className="badge warning" title="AI flagged potential misinformation">
                  <AlertTriangle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }}/> Misinformation Risk
                </span>
              )}
              {article.extreme_bias_flag && (
                <span className="badge warning" title="AI flagged extreme political bias">
                  <AlertCircle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }}/> Extreme Bias
                </span>
              )}
              {article.reliability_score !== undefined && article.reliability_score < 0.4 && (
                <span className="badge warning">Low Reliability ({Math.round(article.reliability_score * 100)}%)</span>
              )}
            </div>

            <div className="bias-meter" style={{ marginBottom: '1rem' }}>
              <span>Left</span>
              <div className="bias-track">
                <div className="bias-thumb" style={{ left: `${biasPct}%` }}></div>
              </div>
              <span>Right</span>
            </div>

            <p className="article-summary">
              {article.ai_summary || article.description}
            </p>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {article.sentiment_score >= 0.6 && (
                <span className="badge" style={{ background: 'var(--positive-tag)', color: 'var(--positive-text)' }}>
                  <CheckCircle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }}/> Positive
                </span>
              )}
              {article.sentiment_score < 0.4 && (
                <span className="badge negative">Negative</span>
              )}
              {article.is_clickbait && (
                <span className="badge warning">Clickbait</span>
              )}
            </div>
          </div>
        );
      })}

      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', alignItems: 'center', marginTop: '2rem' }}>
          <button 
            className="btn" 
            disabled={page === 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            style={{ opacity: page === 1 ? 0.5 : 1 }}
          >
            Prev
          </button>
          <span>Page {page} of {totalPages}</span>
          <button 
            className="btn" 
            disabled={page === totalPages}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            style={{ opacity: page === totalPages ? 0.5 : 1 }}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
