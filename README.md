# MorningNews
A news gathering tool that grabs articles from all over the web and tags them based on quality, bias, popularity, relevance, and other useful metrics. 

## Intended Functions/Workflow:

Read news from a variety of sourcing using APIs, either one or many, depending on which ones we can find that work well. 

Pass news API JSON to AI Agents to critique, review, summarize, and tag it. 

Post tagged articles to a tabbed dashboard with typical news categories, as well as a daily categorical and total summary, update daily.

Allow users to search articles by passing text input into the system and generate a curated return based on user search input. 

## Quick Start (run the dashboard with a provided DB)
1) Clone the repo  
2) Install deps: `python3 -m pip install -r requirements.txt`  
3) Create `.env` at repo root with your own keys (at minimum `OPENAI_API_KEY`; optionally `NEWS_API_KEY`, `TAVILY_API_KEY`, `AI_AGENT_MODEL`, `AI_ANALYSIS_MODEL`).  
4) Place the SQLite database at `data/morningnews.db` (use the prebuilt DB you received, or create it via the ingestion scripts).  
5) Run:  
   ```
   streamlit run Scripts/dashboard.py
   ```
   Opens at http://localhost:8501

## Building the database yourself
If you don’t have a prebuilt `data/morningnews.db`, run the pipeline:
```
python3 Scripts/tavily_ingest.py
python3 Scripts/json_ingest.py
python3 Scripts/ai_agent.py
python3 Scripts/ai_analysis.py
```
Or run the one-liner: `./Scripts/daily_pipeline.sh` (cron/launchd ready).
