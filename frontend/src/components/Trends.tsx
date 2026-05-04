import { useMemo } from 'react';
import type { Article } from '../lib/db';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

interface TrendsProps {
  articles: Article[];
}

export default function Trends({ articles }: TrendsProps) {
  
  // Prepare sentiment data
  const sentimentBins = useMemo(() => {
    const bins = Array(10).fill(0);
    articles.forEach(a => {
      const s = a.sentiment_score ?? 0.5;
      const idx = Math.min(9, Math.floor(s * 10));
      bins[idx]++;
    });
    return bins.map((count, i) => ({
      range: `${(i/10).toFixed(1)}-${((i+1)/10).toFixed(1)}`,
      count
    }));
  }, [articles]);

  // Topic Mix Over Time
  const topicData = useMemo(() => {
    // Group by day
    const days: Record<string, any> = {};
    articles.forEach(a => {
      const day = new Date(a.published_at).toISOString().split('T')[0];
      if (!days[day]) days[day] = { date: day };
      const topic = a.topic || 'Uncategorized';
      days[day][topic] = (days[day][topic] || 0) + 1;
    });

    return Object.values(days).sort((a, b) => a.date.localeCompare(b.date)).slice(-14); // Last 14 days
  }, [articles]);

  const topics = useMemo(() => {
    const s = new Set<string>();
    articles.forEach(a => { if (a.topic) s.add(a.topic); });
    return Array.from(s);
  }, [articles]);

  const colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#6366f1'];

  return (
    <div>
      <h2 style={{ marginBottom: '2rem' }}>Market & Topic Trends</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        
        <div className="glass-panel" style={{ padding: '1.5rem', height: 400 }}>
          <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', color: 'var(--text-secondary)' }}>Sentiment Distribution</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sentimentBins}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="range" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-color)', borderRadius: 8 }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
              <Bar dataKey="count" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem', height: 400 }}>
          <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem', color: 'var(--text-secondary)' }}>Topic Volume (Last 14 Days)</h3>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={topicData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="date" stroke="var(--text-secondary)" tick={{fontSize: 10}} />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-color)', borderRadius: 8 }}
              />
              {topics.map((t, i) => (
                <Area key={t} type="monotone" dataKey={t} stackId="1" stroke={colors[i%colors.length]} fill={colors[i%colors.length]} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
}
