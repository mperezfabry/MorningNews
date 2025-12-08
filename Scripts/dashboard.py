from pathlib import Path

import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from io import BytesIO
import re
import hashlib
import json
from typing import Dict, List

# Attempt to import gTTS for the audio feature.
# I wrapped this in a try/except block so the whole app doesn't crash if
# someone forgets to 'pip install gTTS'. It just disables the audio button.
try:
    from gtts import gTTS

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# --- CONFIGURATION (set early per Streamlit guidance) ---
st.set_page_config(
    page_title="MorningNews AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
SENTIMENT_POSITIVE_THRESHOLD = 0.6
SENTIMENT_NEGATIVE_THRESHOLD = 0.4
STOP_WORDS = set(
    [
        'the','and','to','of','in','a','for','on','with','at','by','is','it','from','as','be',
        'that','this','new','after','report','says','will','today','news','update','latest','breaking',
        'about','their','over','under','into','more','could','would','should','while','where','there'
    ]
)

# FIX: Re-implementing the robust path definition.
# We import 'get_db_path' and assign its result to the global DB_PATH variable.
try:
    from storage import get_db_path

    DB_PATH = Path(get_db_path())
    if not DB_PATH.exists():
        st.error(f"Critical Error: Database not found at {DB_PATH}")
        st.stop()
except ImportError:
    st.error("Critical Error: Could not import 'storage.py'. Please ensure it exists in the project root.")
    st.stop()
# END FIX

# Custom CSS
# I added some custom styling here to make the article cards look cleaner and
# to create the visual 'Bias Meter' bar. Streamlit's default look is okay,
# but this makes it pop.
st.markdown("""
<style>
    .article-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-card {
        border: 2px solid #8b5cf6 !important;
        background-color: #f5f3ff !important;
    }
    /* Bias Meter CSS */
    .bias-meter-container {
        height: 10px;
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #e0e0e0 50%, #ef4444 100%);
        border-radius: 5px;
        position: relative;
        margin-top: 5px;
    }
    .bias-marker {
        width: 4px;
        height: 14px;
        background-color: #111827;
        position: absolute;
        top: -2px;
        border-radius: 2px;
        transition: left 0.3s ease;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 5px;
        color: white;
    }
    .tag-clickbait { background-color: #f59e0b; } /* Orange */
    .tag-verified { background-color: #10b981; } /* Green */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #eef2ff;
        color: #312e81;
        font-size: 0.75em;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-provider { background: #f0f9ff; color: #0f172a; }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    """
    Connects to SQLite and pulls recent articles.
    I'm caching this for 5 minutes (ttl=300) so we don't hammer the database
    every time we click a button.
    """
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT 
                id, title, description, author, source, published_at, url, content,
                sentiment_score, bias_score, is_clickbait, ai_summary, topic, provider
            FROM articles
            ORDER BY published_at DESC
            LIMIT 2000
        """
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    try:
        df['published_at'] = pd.to_datetime(df['published_at'], format='mixed', errors='coerce', utc=True)
    except ValueError:
        df['published_at'] = pd.to_datetime(df['published_at'], infer_datetime_format=True, errors='coerce',
                                            utc=True)

    df = df.dropna(subset=['published_at'])
    df['date_display'] = df['published_at'].dt.strftime('%b %d, %H:%M')
    df['source'] = df['source'].fillna('Unknown Source')

    if 'bias_score' not in df.columns:
        df['bias_score'] = 0.0
    else:
        df['bias_score'] = df['bias_score'].fillna(0.0)

    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.5
    else:
        df['sentiment_score'] = df['sentiment_score'].fillna(0.5)

    if 'is_clickbait' not in df.columns:
        df['is_clickbait'] = False
    else:
        df['is_clickbait'] = df['is_clickbait'].fillna(0).astype(bool)

    if 'ai_summary' not in df.columns:
        df['ai_summary'] = df['description']
    else:
        df['ai_summary'] = df['ai_summary'].fillna(df['description'])

    if "topic" not in df.columns:
        df["topic"] = pd.NA
    if "provider" not in df.columns:
        df["provider"] = pd.NA

    def _infer_topic(row):
        if pd.notna(row.get("topic")) and str(row.get("topic")).strip():
            return str(row.get("topic")).strip()
        text = ((row.get("title") or "") + " " + (row.get("description") or "")).lower()
        source = (row.get("source") or "").lower()
        provider = (row.get("provider") or "").lower()

        keyword_map = {
            "business": ["market", "stocks", "finance", "economy", "business"],
            "technology": ["tech", "ai", "software", "hardware"],
            "science": ["science", "research", "space", "biology"],
            "health": ["health", "medicine", "covid"],
            "sports": ["sport", "game", "nfl", "nba", "mlb", "soccer"],
            "politics": ["election", "policy", "government", "congress", "senate"],
            "international": ["world", "international", "global", "geopolitics"],
        }
        for cat, kws in keyword_map.items():
            if any(kw in text for kw in kws):
                return cat.title()
        if provider:
            return provider.title()
        if source:
            return source.title()
        return "Uncategorized"

    df["inferred_topic"] = df.apply(_infer_topic, axis=1)
    df["topic_display"] = (
        df["topic"]
        .fillna(df["inferred_topic"])
        .fillna("Uncategorized")
        .astype(str)
        .str.strip()
        .replace("", "Uncategorized")
        .str.title()
    )

    return df


def toggle_bookmark(article_id: str):
    """
    Adds or removes an article from the bookmarks table.
    I used a simple check-then-insert/delete logic here.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bookmarks (
                    article_id TEXT PRIMARY KEY
                )
                """
            )
            cur = conn.execute("SELECT 1 FROM bookmarks WHERE article_id = ?", (article_id,))
            exists = cur.fetchone()
            if exists:
                conn.execute("DELETE FROM bookmarks WHERE article_id = ?", (article_id,))
            else:
                conn.execute("INSERT INTO bookmarks (article_id) VALUES (?)", (article_id,))
            conn.commit()
    except Exception as e:
        st.error(f"Bookmark error: {e}")


@st.cache_data(ttl=60, show_spinner=False)
def load_bookmarks():
    """Cached bookmark loader to reduce DB hits."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT article_id FROM bookmarks", conn)
            return df['article_id'].tolist()
    except Exception:
        return []


def generate_audio(text: str):
    """
    Uses gTTS to generate an audio file in memory (BytesIO) so we don't have to save mp3s to disk.
    If gTTS isn't installed, it just warns the user.
    """
    if not HAS_AUDIO:
        st.warning("gTTS library not installed. Run `pip install gTTS` to enable audio.")
        return None

    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None


def clean_content_for_display(text: str | None) -> str:
    """Strip common navigation filler lines from scraped summaries."""
    if not text:
        return ""
    lines = []
    skip_prefixes = ("browse", "skip links", "skip to content", "all quotes delayed")
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(lower.startswith(prefix) for prefix in skip_prefixes):
            continue
        lines.append(stripped)
    return " ".join(lines)


def get_filters(df: pd.DataFrame, coverage: Dict[str, float]) -> Dict:
    """Render sidebar controls and return selected filter state."""
    with st.sidebar:
        st.title("🤖 MorningNews AI")
        st.caption("Curate, filter, and brief your feed.")
        st.caption(
            f"AI coverage • summaries: {coverage.get('ai_summary_pct', 0):.0f}% • "
            f"bias: {coverage.get('bias_score_pct', 0):.0f}%"
        )

        with st.expander("Personalize", expanded=True):
            watchlist_input = st.text_input(
                "Watchlist (comma separated)",
                placeholder="Crypto, AI, SpaceX",
            )
            watchlist = [t.strip().lower() for t in watchlist_input.split(",") if t.strip()]

        with st.expander("Filters", expanded=True):
            min_date = df["published_at"].min().date()
            max_date = df["published_at"].max().date()
            selected = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            sources = st.multiselect("Sources", options=sorted(df["source"].unique()))
            categories_default = st.session_state.get("category_filter", [])
            categories = st.multiselect(
                "Categories",
                options=sorted(df["topic_display"].unique()),
                default=categories_default,
            )
            good_news_only = st.checkbox(f"Positive sentiment only (>= {SENTIMENT_POSITIVE_THRESHOLD})")
            show_bookmarks = st.checkbox("Bookmarked only")
            st.caption(
                f"Sentiment tags: Positive ≥ {SENTIMENT_POSITIVE_THRESHOLD}, "
                f"Negative < {SENTIMENT_NEGATIVE_THRESHOLD}; bias meter ranges -1 (Left) to 1 (Right)."
            )

    return {
        "date_selection": selected,
        "sources": sources,
        "categories": categories,
        "good_news_only": good_news_only,
        "show_bookmarks": show_bookmarks,
        "watchlist": watchlist,
    }


def apply_filters(df: pd.DataFrame, filters: Dict, bookmarks: List[str]) -> pd.DataFrame:
    """Apply all filters to the dataframe and return the result."""
    filtered = df.copy()

    selected = filters["date_selection"]
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        start_date, end_date = selected
    else:
        start_date = end_date = selected
    mask = (filtered["published_at"].dt.date >= start_date) & (filtered["published_at"].dt.date <= end_date)
    filtered = filtered.loc[mask]

    if filters["sources"]:
        filtered = filtered[filtered["source"].isin(filters["sources"])]

    if filters["categories"]:
        st.session_state.category_filter = filters["categories"]
        filtered = filtered[filtered["topic_display"].isin(filters["categories"])]
    else:
        st.session_state.category_filter = []

    if filters["good_news_only"]:
        filtered = filtered[filtered["sentiment_score"] >= SENTIMENT_POSITIVE_THRESHOLD]

    if filters["show_bookmarks"]:
        filtered = filtered[filtered["id"].isin(bookmarks)]

    return filtered


def filters_hash(filters: Dict) -> str:
    """Stable hash of filter state for pagination resets."""
    safe = {}
    for k, v in filters.items():
        if k == "watchlist":
            # Changing watchlist shouldn't reset pagination
            continue
        if isinstance(v, (list, tuple, set)):
            safe[k] = list(v)
        else:
            safe[k] = v
    blob = json.dumps(safe, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


# --- MAIN APP ---

def main():
    # Initialize session state for bookmarks so we can track what's saved across reruns
    # This must be inside main() where st.session_state is available
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = load_bookmarks()

    df = load_data()
    if df.empty:
        st.warning("⚠️ No data found. Run ingestion scripts first.")
        return

    bias_data_present = df["bias_score"].abs().max(skipna=True) > 0.01
    total = len(df)
    ai_summary_pct = (df["ai_summary"].notna().sum() / total * 100) if total else 0.0
    bias_score_pct = (df["bias_score"].notna().sum() / total * 100) if total else 0.0
    coverage = {"ai_summary_pct": ai_summary_pct, "bias_score_pct": bias_score_pct}

    filters = get_filters(df, coverage)
    filtered = apply_filters(df, filters, st.session_state.bookmarks)
    watchlist = filters["watchlist"]
    current_hash = filters_hash(filters)
    if "filters_hash" not in st.session_state or st.session_state.filters_hash != current_hash:
        st.session_state.filters_hash = current_hash
        st.session_state.page = 1

    # --- TABS ---
    tab_feed, tab_brief, tab_viz = st.tabs(["📰 Smart Feed", "🎙️ Daily Brief", "📈 Trends"])

    # Precompile watchlist regex once per run
    watch_terms = [t for t in watchlist if t]
    watch_pattern = None
    if watch_terms:
        watch_pattern = re.compile(r"\b(" + "|".join(map(re.escape, watch_terms)) + r")\b", re.IGNORECASE)

    # --- TAB 1: SMART FEED ---
    with tab_feed:
        st.subheader(f"Top Stories ({len(filtered)})")

        if not bias_data_present:
            st.info("Bias meter data missing (bias_score ~ 0). Run `python3 Scripts/ai_analysis.py` with GEMINI_API_KEY to populate AI fields.")

        # Quick category pills
        top_categories = [c for c in sorted(df['topic_display'].unique()) if c and c.lower() != "uncategorized"][:8]
        cols = st.columns(len(top_categories) + 1) if top_categories else st.columns(1)
        with cols[0]:
            if st.button("All", key="cat_all"):
                st.session_state.category_filter = []
                st.rerun()
        for idx, cat in enumerate(top_categories, start=1):
            with cols[idx]:
                active = cat in st.session_state.get("category_filter", [])
                label = f"✅ {cat}" if active else cat
                if st.button(label, key=f"cat_{cat}"):
                    st.session_state.category_filter = [] if active else [cat]
                    st.rerun()

        # Simple Pagination with session state
        page_size = 20
        page_total = max((len(filtered) - 1) // page_size + 1, 1)
        if "page" not in st.session_state:
            st.session_state.page = 1
        if st.session_state.page > page_total:
            st.session_state.page = page_total
        col_prev, col_page, col_next = st.columns([0.2, 0.6, 0.2])
        with col_prev:
            if st.button("⬅️ Prev", disabled=st.session_state.page <= 1):
                st.session_state.page = max(1, st.session_state.page - 1)
                st.rerun()
        with col_page:
            st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} / {page_total}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next ➡️", disabled=st.session_state.page >= page_total):
                st.session_state.page = min(page_total, st.session_state.page + 1)
                st.rerun()

        start = (st.session_state.page - 1) * page_size
        page_data = filtered.iloc[start : start + page_size]

        if page_data.empty:
            st.info("No articles match your filters. Try widening the date range or clearing bookmarks.")

        for _, row in page_data.iterrows():
            # Check Watchlist Match
            haystack = (row['title'] + " " + str(row['description']))
            is_watchlist = bool(watch_pattern.search(haystack)) if watch_pattern else False

            # Dynamic CSS class for highlighting
            card_style = "article-card highlight-card" if is_watchlist else "article-card"

            st.markdown(f'<div class="{card_style}">', unsafe_allow_html=True)
            with st.container():
                if is_watchlist:
                    st.info("👀 Matches your watchlist")

                c1, c2 = st.columns([0.80, 0.20])

                with c1:
                    # Clickbait warning icon next to the title.
                    warning = "⚠️ " if row['is_clickbait'] else ""
                    st.markdown(f"#### [{warning}{row['title']}]({row['url']})")

                    badges = ""
                    topic_badge = row.get("topic_display") or "Uncategorized"
                    provider = row.get("source") or "Unknown"
                    badges += f'<span class="badge">{topic_badge}</span>'
                    badges += f'<span class="badge badge-provider">{provider}</span>'
                    st.markdown(badges, unsafe_allow_html=True)
                    st.caption(f"{row['date_display']}")

                    # BIAS METER VISUALIZATION
                    try:
                        bias_raw = float(row['bias_score'])
                    except Exception:
                        bias_raw = 0.0
                    bias_clamped = max(-1.0, min(1.0, bias_raw))
                    bias_pct = (bias_clamped + 1) / 2 * 100
                    st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:10px; font-size:0.8em; margin-bottom:5px; width: 60%;">
                            <span style="color:blue">Left</span>
                            <div class="bias-meter-container">
                                <div class="bias-marker" style="left: {bias_pct}%;"></div>
                            </div>
                            <span style="color:red">Right</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Prefer the AI summary if we have it, otherwise fallback to description.
                    content_text = row['ai_summary'] if row['ai_summary'] else row['description']
                    content_text = clean_content_for_display(content_text)
                    if content_text:
                        st.write(content_text)

                with c2:
                    # Bookmark Button Logic
                    is_saved = row['id'] in st.session_state.bookmarks
                    btn_label = "★ Saved" if is_saved else "☆ Save"
                    if st.button(btn_label, key=f"btn_{row['id']}"):
                        toggle_bookmark(row['id'])
                        load_bookmarks.clear()  # Avoid stale cached bookmarks
                        st.session_state.bookmarks = load_bookmarks()  # Refresh state immediately
                        st.rerun()

                    if row['is_clickbait']:
                        st.markdown('<span class="tag tag-clickbait">Clickbait</span>', unsafe_allow_html=True)

                    sent = row['sentiment_score']
                    if sent >= SENTIMENT_POSITIVE_THRESHOLD:
                        st.markdown('<span class="tag tag-verified">Positive</span>', unsafe_allow_html=True)
                    elif sent < SENTIMENT_NEGATIVE_THRESHOLD:
                        st.markdown('<span class="tag" style="background-color:#ef4444">Negative</span>',
                                    unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()

    # --- TAB 2: INTERACTIVE BRIEF ---
    with tab_brief:
        c1, c2 = st.columns([2, 1])

        with c1:
            with st.container():
                st.subheader("☀️ The Daily Download")
                top_5 = filtered.head(5)
                if not top_5.empty:
                    bullets = []
                    for _, r in top_5.iterrows():
                        short = (r.get("ai_summary") or r.get("description") or "").strip()
                        short = clean_content_for_display(short)
                        if len(short) > 200:
                            short = short[:200] + "..."
                        bullets.append(f"{r['title']}: {short}")
                    summary_text = "Here is your Morning News AI Briefing.\n\n" + "\n".join(f"- {b}" for b in bullets)
                else:
                    summary_text = "No major stories found matching your current filters."

                st.info(summary_text)

                if st.button("▶️ Play Audio Briefing"):
                    audio_bytes = generate_audio(summary_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')

        with c2:
            st.subheader("Chat with News")
            user_query = st.text_input("Ask about the news:", placeholder="What happened with Apple?")

            if user_query:
                tokens = [t for t in re.split(r"\W+", user_query.lower()) if t]
                if tokens:
                    pattern = "|".join(re.escape(t) for t in tokens)
                    mask = filtered['title'].str.contains(pattern, case=False, na=False)
                    mask |= filtered['description'].str.contains(pattern, case=False, na=False)
                    hits = filtered[mask]
                else:
                    hits = filtered.iloc[0:0]

                if not hits.empty:
                    st.success(f"Found {len(hits)} relevant articles:")
                    for _, hit in hits.head(3).iterrows():
                        summary = clean_content_for_display(hit.get("ai_summary") or hit.get("description"))
                        summary = (summary[:220] + "...") if summary and len(summary) > 220 else summary
                        title = hit.get("title") or "(untitled)"
                        url = hit.get("url")
                        header = f"[{title}]({url})" if url else title
                        st.markdown(
                            f"- {header} ({hit['source']}, {hit['date_display']})\n\n"
                            f"  {summary}"
                        )
                else:
                    st.warning("No relevant articles found in the current feed.")

    # --- TAB 3: TRENDS VISUALIZATION ---
    with tab_viz:
        st.subheader("Market & Topic Trends")

        lookback_days = st.slider(
            "Lookback window (days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="Controls the time window for the news velocity plot.",
        )

        # 1. Timeline Chart
        # I'm grouping by hour here to show the "velocity" of news dropping.
        timeline_df = filtered.copy()
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        timeline_df = timeline_df[timeline_df['published_at'] >= cutoff]
        timeline_df['date_hour'] = timeline_df['published_at'].dt.floor('h')
        volume_data = timeline_df.groupby('date_hour').size().reset_index(name='count')

        title = "News Velocity"
        if not volume_data.empty:
            start_dt = volume_data['date_hour'].min().strftime('%b %d')
            end_dt = volume_data['date_hour'].max().strftime('%b %d')
            title = f"News Velocity ({start_dt} – {end_dt})"

        chart_timeline = alt.Chart(volume_data).mark_line(point=True).encode(
            x=alt.X('date_hour', title='Time'),
            y=alt.Y('count', title='Article Volume'),
            tooltip=['date_hour', 'count']
        ).properties(title=title).interactive()

        st.altair_chart(chart_timeline, use_container_width=True)

        # Topic mix over time (stacked area)
        topic_counts = (
            timeline_df
            .groupby(['date_hour', 'topic_display'])
            .size()
            .reset_index(name='count')
        )
        if not topic_counts.empty:
            chart_topic = (
                alt.Chart(topic_counts)
                .mark_area()
                .encode(
                    x=alt.X('date_hour:T', title='Time'),
                    y=alt.Y('count:Q', stack='normalize', title='Share of Articles'),
                    color=alt.Color('topic_display:N', title='Topic'),
                    tooltip=['date_hour', 'topic_display', 'count'],
                )
                .properties(title="Topic Mix Over Time")
                .interactive()
            )
            st.altair_chart(chart_topic, use_container_width=True)

        # 2. Word Cloud Proxy
        # Streamlit doesn't have a native Word Cloud, so I built a frequency analysis
        # bar chart instead. It's cleaner and interactive.
        all_titles = " ".join(filtered['title'].dropna()).lower()
        words = [re.sub(r"[^a-z']", "", w) for w in all_titles.split()]
        words = [w for w in words if len(w) > 4 and w not in STOP_WORDS]

        if words:
            word_freq = pd.Series(words).value_counts().reset_index()
            word_freq.columns = ['Word', 'Frequency']

            chart_cloud = alt.Chart(word_freq.head(20)).mark_bar().encode(
                x=alt.X('Frequency'),
                y=alt.Y('Word', sort='-x'),
                color=alt.Color('Frequency', scale={'scheme': 'viridis'})
            ).properties(title="Trending Keywords (Top 20)")

            st.altair_chart(chart_cloud, use_container_width=True)
        else:
            st.info("Not enough data for keyword analysis.")

        # 3. Sentiment Distribution
        if not filtered.empty:
            sentiment_chart = (
                alt.Chart(filtered)
                .mark_bar()
                .encode(
                    x=alt.X('sentiment_score:Q', bin=alt.Bin(maxbins=20), title='Sentiment Score'),
                    y=alt.Y('count()', title='Article Count'),
                )
                .properties(title="Sentiment Distribution")
            )
            st.altair_chart(sentiment_chart, use_container_width=True)


if __name__ == "__main__":
    main()
