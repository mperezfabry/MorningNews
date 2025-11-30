import sys
import os
import random
import time
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import altair as alt
from io import BytesIO

# --- PATH SETUP ---
# I needed to do this path hacking because 'storage.py' is in the root folder,
# but this script runs from inside 'Scripts/'. Without adding the parent directory
# to sys.path, Python can't find 'storage' and crashes.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



# import gTTS for audio, handle if missing

# Attempt to import gTTS for the audio feature.
# I wrapped this in a try/except block so the whole app doesn't crash if
# someone forgets to 'pip install gTTS'. It just disables the audio button.
try:
    from gtts import gTTS

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
# import for the database path.
# If this fails, it means the project structure is messed up, so I adjust everything to be inside folder MorningNews\Scripts\...
try:
    from storage import DB_PATH
except ImportError:
    st.error("Critical Error: Could not import 'storage.py'. Please ensure it exists in the project root.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MorningNews AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bias Meter, Cards, and Badges
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
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

# load_data connects to SQLite and pulls recent articles.
# I'm caching this for 5 minutes (ttl=300) so we don't hammer the database every time we click a button.
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Explicitly select the AI columns
            query = """
                SELECT 
                    id, title, description, author, source, published_at, url, content,
                    sentiment_score, bias_score, is_clickbait, ai_summary
                FROM articles
                ORDER BY published_at DESC
                LIMIT 500
            """
# Handle case where columns might not exist yet (backward compatibility)
# Fallback logic: if the new columns don't exist yet, just grab everything.
            try:
                df = pd.read_sql_query(query, conn)
            except Exception:
                # Fallback to standard query if schema update failed
                fallback_query = "SELECT * FROM articles ORDER BY published_at DESC LIMIT 500"
                df = pd.read_sql_query(fallback_query, conn)

            if df.empty:
                return df

# Robust Date Parsing because date parsing can be tricky with different formats in the DB.
# I used format='mixed' and errors='coerce' to be safe—if a date is bad, it just becomes NaT instead of crashing the whole app.
            try:
                df['published_at'] = pd.to_datetime(df['published_at'], format='mixed', errors='coerce', utc=True)
            except ValueError:
                df['published_at'] = pd.to_datetime(df['published_at'], infer_datetime_format=True, errors='coerce',
                                                    utc=True)

            df = df.dropna(subset=['published_at'])
            df['date_display'] = df['published_at'].dt.strftime('%b %d, %H:%M')
            df['source'] = df['source'].fillna('Unknown Source')

# AI DATA HANDLING
# If columns are missing or null, fill with defaults/simulation so UI doesn't break
# Ideally, these columns come populated from the DB. But if the AI script haven't run yet,
# or if we are testing, I'm filling in defaults so the UI doesn't break.

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
                # SQLite stores bools as 0/1 usually
                df['is_clickbait'] = df['is_clickbait'].fillna(0).astype(bool)

            if 'ai_summary' not in df.columns:
                df['ai_summary'] = df['description']
            else:
                df['ai_summary'] = df['ai_summary'].fillna(df['description'])

            return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

# Adds or removes an article from the bookmarks table
def toggle_bookmark(article_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("SELECT 1 FROM bookmarks WHERE article_id = ?", (article_id,))
            exists = cur.fetchone()
            if exists:
                conn.execute("DELETE FROM bookmarks WHERE article_id = ?", (article_id,))
            else:
                conn.execute("INSERT INTO bookmarks (article_id) VALUES (?)", (article_id,))
            conn.commit()
    except Exception as e:
        st.error(f"Bookmark error: {e}")

# returns a simple list of IDs for articles the user has saved.
def get_bookmarks():
    if not DB_PATH.exists(): return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Check if table exists first (just in case update_schema failed silently)
            try:
                df = pd.read_sql_query("SELECT article_id FROM bookmarks", conn)
                return df['article_id'].tolist()
            except:
                return []
    except:
        return []

# Uses gTTS to generate an audio file in memory (BytesIO) from text, so we don't have to save mp3s to disk. If gTTS isn't installed, it just warns the user.
def generate_audio(text):
    if not HAS_AUDIO:
        st.warning("gTTS library not installed. Run `pip install gTTS` to enable audio.")
        return None

    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None


# --- MAIN APP ---

def main():
    # Initialize session state for bookmarks so we can track what's saved across reruns
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = get_bookmarks()

    df = load_data()
    if df.empty:
        st.warning("⚠️ No data found. Run ingestion scripts first.")
        return

    # --- SIDEBAR: CONTROLS & PERSONALIZATION ---
    with st.sidebar:
        st.title("🤖 MorningNews AI")
        st.caption("Intelligent Daily Briefing")

        st.header("Personalization")
        # Watchlist Feature: User can types keywords, we highlight them in the feed. Long phrases do not work with this set up.
        watchlist_input = st.text_input("My Watchlist (comma separated)", placeholder="Crypto, AI, SpaceX")
        watchlist = [t.strip().lower() for t in watchlist_input.split(',')] if watchlist_input else []

        st.header("AI Filters")
        # Sentiment Toggle
        good_news_only = st.checkbox("Show only Positive News (+ Sentiment)")
        show_bookmarks = st.checkbox("Show Bookmarked Only")

        st.divider()

        # Standard Date/Source Filters
        min_date = df['published_at'].min().date()
        max_date = df['published_at'].max().date()
        selected_date = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        sources = st.multiselect("Sources", options=sorted(df['source'].unique()))

    # FILTERING LOGIC
    # I create a copy so I don't mess up the original cached dataframe.
    filtered = df.copy()

    # Date Filter
    if isinstance(selected_date, tuple) and len(selected_date) == 2:
        mask = (filtered['published_at'].dt.date >= selected_date[0]) & (
                    filtered['published_at'].dt.date <= selected_date[1])
        filtered = filtered.loc[mask]

    # Source Filter
    if sources:
        filtered = filtered[filtered['source'].isin(sources)]

    # Good News Filter
    if good_news_only:
        filtered = filtered[filtered['sentiment_score'] > 0.6]  # Strictly positive

    # Bookmark Filter
    if show_bookmarks:
        filtered = filtered[filtered['id'].isin(st.session_state.bookmarks)]

    # ---- TABS --------
    tab_feed, tab_brief, tab_viz = st.tabs(["📰 Smart Feed", "🎙️ Daily Brief", "📈 Trends"])

    # ------ TAB 1: SMART FEED -------
    with tab_feed:
        st.subheader(f"Top Stories ({len(filtered)})")

        # Simple Pagination to prevent lag
        # Shows only top 20 to keep the UI snappy.
        # Can add a 'load more' button later if needed.
        page_size = 20
        page_number = st.number_input("Page", min_value=1, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        page_data = filtered.iloc[start_idx:end_idx]

        for _, row in page_data.iterrows():
            # Check Watchlist Match
            is_watchlist = any(term in (row['title'] + str(row['description'])).lower() for term in watchlist)

            # Use columns layout
            with st.container():
                if is_watchlist:
                    st.info("👀 Matches your watchlist")

                c1, c2 = st.columns([0.80, 0.20])

                with c1:
                    # Clickbait Warning Icon
                    warning = "⚠️ " if row['is_clickbait'] else ""
                    st.markdown(f"#### [{warning}{row['title']}]({row['url']})")
                    st.caption(f"**{row['source']}** • {row['date_display']}")

                    # BIAS METER VISUALIZATION
                    # Convert score (-1 to 1) to percentage (0 to 100) for the CSS marker
                    bias_pct = (row['bias_score'] + 1) / 2 * 100
                    st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:10px; font-size:0.8em; margin-bottom:5px; width: 60%;">
                            <span style="color:blue">Left</span>
                            <div class="bias-meter-container">
                                <div class="bias-marker" style="left: {bias_pct}%;"></div>
                            </div>
                            <span style="color:red">Right</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Use AI Summary if available, else description
                    content_text = row['ai_summary'] if row['ai_summary'] else row['description']
                    if content_text:
                        st.write(content_text)

                with c2:
                    # Bookmark Button Logic
                    is_saved = row['id'] in st.session_state.bookmarks
                    if st.button("★ Saved" if is_saved else "☆ Save", key=f"btn_{row['id']}"):
                        toggle_bookmark(row['id'])
                        st.session_state.bookmarks = get_bookmarks()
                        st.rerun()

                    # Tags
                    if row['is_clickbait']:
                        st.markdown('<span class="tag tag-clickbait">Clickbait</span>', unsafe_allow_html=True)

                    # Sentiment Tag
                    sent = row['sentiment_score']
                    if sent > 0.6:
                        st.markdown('<span class="tag tag-verified">Positive</span>', unsafe_allow_html=True)
                    elif sent < 0.4:
                        st.markdown('<span class="tag" style="background-color:#ef4444">Negative</span>',
                                    unsafe_allow_html=True)

                st.divider()

    # --------- TAB 2: INTERACTIVE BRIEF ------------
    with tab_brief:
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("☀️ The Daily Download")
            # Generate a summary string from the current filtered view
            top_5 = filtered.head(5)['title'].tolist()
            if top_5:
                summary_text = "Here is your Morning News AI Briefing. Top stories include: " + ". ".join(top_5) + "."
            else:
                summary_text = "No major stories found matching your current filters."

            st.info(summary_text)

            # Text-to-Speech Button
            if st.button("▶️ Play Audio Briefing"):
                audio_bytes = generate_audio(summary_text)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')

        with c2:
            st.subheader("Chat with News")
            user_query = st.text_input("Ask about the news:", placeholder="What happened with Apple?")

            if user_query:
                # Simple Keyword Search (Simulating RAG)
                hits = filtered[
                    filtered['title'].str.contains(user_query, case=False) |
                    filtered['description'].str.contains(user_query, case=False)
                    ]

                if not hits.empty:
                    st.success(f"Found {len(hits)} relevant articles:")
                    for _, hit in hits.head(3).iterrows():
                        st.write(f"- **{hit['title']}** ({hit['source']})")
                else:
                    st.warning("No relevant articles found in the current feed.")

    # ------ TAB 3: TRENDS VISUALIZATION -------
    with tab_viz:
        st.subheader("Market & Topic Trends")

        # 1. Timeline Chart (News Velocity)
        timeline_df = filtered.copy()
        # Group by hour to show volume over time
        timeline_df['date_hour'] = timeline_df['published_at'].dt.floor('h')
        volume_data = timeline_df.groupby('date_hour').size().reset_index(name='count')

        chart_timeline = alt.Chart(volume_data).mark_line(point=True).encode(
            x=alt.X('date_hour', title='Time'),
            y=alt.Y('count', title='Article Volume'),
            tooltip=['date_hour', 'count']
        ).properties(title="News Velocity (Last 7 Days)").interactive()

        st.altair_chart(chart_timeline, use_container_width=True)

        # 2. Word Cloud Proxy (Top Keywords)
        # Simple frequency count of words in titles
        all_titles = " ".join(filtered['title'].dropna()).lower()
        # Filter out common stop words
        stop_words = set(
            ['the', 'and', 'to', 'of', 'in', 'a', 'for', 'on', 'with', 'at', 'by', 'is', 'it', 'from', 'as', 'be',
             'that', 'this', 'new', 'after'])
        words = [w for w in all_titles.split() if len(w) > 4 and w not in stop_words]

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
            st.info("Not enough data for word cloud.")


if __name__ == "__main__":
    main()