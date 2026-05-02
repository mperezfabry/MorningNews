import boto3
import os
from pathlib import Path
from Scripts.storage import get_db_path

S3_BUCKET = "morningnews-db-873980777388"
S3_KEY = "morningnews.db"

def download_db_from_s3():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    s3 = boto3.client('s3')
    print(f"Downloading {S3_KEY} from {S3_BUCKET} to {db_path}...")
    s3.download_file(S3_BUCKET, S3_KEY, str(db_path))

def upload_db_to_s3():
    db_path = get_db_path()
    if not db_path.exists():
        print(f"Error: {db_path} does not exist. Cannot upload.")
        return
    
    s3 = boto3.client('s3')
    print(f"Uploading {db_path} to s3://{S3_BUCKET}/{S3_KEY}...")
    s3.upload_file(str(db_path), S3_BUCKET, S3_KEY)

if __name__ == "__main__":
    # Small CLI for testing
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            download_db_from_s3()
        elif sys.argv[1] == "upload":
            upload_db_to_s3()
