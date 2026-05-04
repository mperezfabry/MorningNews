import initSqlJs from 'sql.js';
import type { Database, SqlJsStatic } from 'sql.js';

let dbInstance: Database | null = null;
let initPromise: Promise<Database> | null = null;

export const initDb = async (): Promise<Database> => {
  if (dbInstance) return dbInstance;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      const SQL: SqlJsStatic = await initSqlJs({
        locateFile: () => `/sql-wasm-2.wasm`
      });

      // The DB is hosted in the same S3 bucket/CloudFront distribution
      // so we can just fetch it relatively if served from the root.
      // During local dev, we might need a proxy or we can download it locally
      // For this portfolio piece, assuming it's available at /morningnews.db
      // Or we can point to the absolute URL. We'll use absolute URL for simplicity
      // and robust local dev.
      const dbUrl = window.location.hostname === 'localhost' 
        ? 'https://morningnews.fabryperez.com/morningnews.db'
        : `/morningnews.db?v=${Date.now()}`;

      const response = await fetch(dbUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch database: ${response.statusText}`);
      }
      
      const buffer = await response.arrayBuffer();
      dbInstance = new SQL.Database(new Uint8Array(buffer));
      return dbInstance;
    } catch (err) {
      console.error("Database initialization failed:", err);
      throw err;
    }
  })();

  return initPromise;
};

export const getDb = (): Database => {
  if (!dbInstance) throw new Error("Database not initialized yet.");
  return dbInstance;
};

// Types
export interface Article {
  id: string;
  title: string;
  description: string;
  author: string;
  source: string;
  published_at: string;
  url: string;
  content: string;
  sentiment_score: number;
  bias_score: number;
  is_clickbait: boolean;
  ai_summary: string;
  topic: string;
  provider: string;
  // Tags from article_ai_tags
  quality_score?: number;
  reliability_score?: number;
  misinformation_flag?: boolean;
  extreme_bias_flag?: boolean;
}

export interface DailySummary {
  summary_date: string;
  category: string;
  summary_text: string;
}

export const fetchArticles = async (): Promise<Article[]> => {
  const db = await initDb();
  const query = `
    SELECT 
      a.id, a.title, a.description, a.author, a.source, a.published_at, a.url,
      a.sentiment_score, a.bias_score, a.is_clickbait, a.ai_summary, a.topic, a.provider,
      t.quality_score, t.reliability_score, t.misinformation_flag, t.extreme_bias_flag
    FROM articles a
    LEFT JOIN article_ai_tags t ON a.id = t.article_id
    ORDER BY a.published_at DESC
    LIMIT 2000
  `;
  const res = db.exec(query);
  if (res.length === 0) return [];
  
  const columns = res[0].columns;
  const values = res[0].values;
  
  return values.map(row => {
    const article: any = {};
    columns.forEach((col, idx) => {
      article[col] = row[idx];
    });
    // Type coercions
    article.is_clickbait = Boolean(article.is_clickbait);
    article.misinformation_flag = Boolean(article.misinformation_flag);
    article.extreme_bias_flag = Boolean(article.extreme_bias_flag);
    return article as Article;
  });
};

export const fetchDailySummaries = async (): Promise<DailySummary[]> => {
  const db = await initDb();
  // Fetch the latest summary date available
  const dateQuery = `SELECT MAX(summary_date) as max_date FROM daily_summaries`;
  const dateRes = db.exec(dateQuery);
  if (dateRes.length === 0 || !dateRes[0].values[0][0]) return [];
  const maxDate = dateRes[0].values[0][0];

  const query = `
    SELECT summary_date, category, summary_text
    FROM daily_summaries
    WHERE summary_date = ?
  `;
  const stmt = db.prepare(query);
  stmt.bind([maxDate as string]);
  
  const summaries: DailySummary[] = [];
  while (stmt.step()) {
    const row = stmt.getAsObject();
    summaries.push(row as unknown as DailySummary);
  }
  stmt.free();
  return summaries;
};
