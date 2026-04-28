import json
import os
import subprocess
import sys
from Scripts.s3_storage import download_db_from_s3, upload_db_to_s3
from Scripts.storage import get_db_path

def handler(event, context):
    print("Starting MorningNews Ingestion Pipeline via Lambda...")
    
    # 1. Setup DB path for Lambda /tmp
    os.environ["MORNINGNEWS_DB_PATH"] = "/tmp/morningnews.db"
    db_path = get_db_path()
    
    # 2. Download DB from S3
    try:
        download_db_from_s3()
    except Exception as e:
        print(f"No existing DB found on S3 or download failed: {e}")
        # If it fails, we might want to start fresh or fail. 
        # For now, let's assume we need the DB.
    
    # 3. Run Ingestion Scripts
    scripts = [
        "Scripts/tavily_ingest.py",
        "Scripts/json_ingest.py",
        "Scripts/ai_agent.py",
        "Scripts/ai_analysis.py"
    ]
    
    for script in scripts:
        print(f"Running {script}...")
        try:
            # We use sys.executable to ensure we use the same python environment
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running {script}: {result.stderr}")
            else:
                print(f"Successfully ran {script}")
        except Exception as e:
            print(f"Failed to execute {script}: {e}")

    # 4. Upload updated DB back to S3
    print("Uploading updated database to S3...")
    upload_db_to_s3()
    
    return {
        'statusCode': 200,
        'body': json.dumps('Ingestion completed successfully!')
    }
