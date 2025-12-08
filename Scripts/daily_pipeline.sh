#!/bin/zsh
# Daily pipeline: ingest + tagging + analysis.
# Run this via cron/launchd to keep the dashboard fresh.

set -euo pipefail

# Move to repo root (this script lives in Scripts/)
cd "$(dirname "$0")/.."

echo "[pipeline] Starting run at $(date)"

# Load environment variables if present
if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
else
  echo "[pipeline] Warning: .env not found; relying on shell env"
fi

# Ingest fresh articles (Tavily + JSON/NewsAPI if configured)
python3 Scripts/tavily_ingest.py
python3 Scripts/json_ingest.py

# Tag articles and write summaries
python3 Scripts/ai_agent.py
python3 Scripts/ai_analysis.py

echo "[pipeline] Completed at $(date)"
