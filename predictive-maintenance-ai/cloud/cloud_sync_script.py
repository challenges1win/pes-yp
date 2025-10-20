"""
ynchronizes local sensor and label data
to a cloud database (Firebase or AWS S3/DynamoDB).
Usage:
    python cloud_sync_script.py --provider firebase
or:
    python cloud_sync_script.py --provider aws
"""


import os
import json
import pandas as pd
import datetime
import argparse

# Optional imports (Firebase / AWS)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    boto3 = None

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
SCHEMA_PATH = os.path.join(BASE_DIR, 'database_schema.json')


# -----------------------------------------------------------
# LOAD DATABASE SCHEMA
# -----------------------------------------------------------
def load_schema():
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r') as f:
            schema = json.load(f)
        return schema
    return {}


# -----------------------------------------------------------
#  FIREBASE SYNC
# -----------------------------------------------------------
def sync_to_firebase(schema, dry_run=False):
    """Sync label files to Firebase. When dry_run=True the function will only print
    what it would do instead of making network calls.
    """
    collection = schema.get("collections", {}).get("labels", "labels")

    if dry_run:
        print("DRY RUN: Firebase sync - no network operations will be performed.")
        for file in os.listdir(DATA_DIR):
            if file.startswith("labels_") and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                sensor_type = file.replace("labels_", "").replace(".csv", "")
                print(f"DRY RUN: Would upload {len(df)} labels to Firebase collection '{collection}/{sensor_type}'")
        print("DRY RUN: Firebase sync complete.")
        return

    if firebase_admin is None:
        raise ImportError("firebase_admin not installed. Run `pip install firebase-admin`")

    cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Missing Firebase credentials: {cred_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Upload labeled data files
    for file in os.listdir(DATA_DIR):
        if file.startswith("labels_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            sensor_type = file.replace("labels_", "").replace(".csv", "")
            ref = db.collection(collection).document(sensor_type)
            ref.set({
                "sensor": sensor_type,
                "uploaded_at": datetime.datetime.utcnow(),
                "records": df.to_dict(orient="records")
            })
            print(f"✅ Uploaded {len(df)} labels to Firebase collection '{collection}/{sensor_type}'")

    print("🌩️ Firebase sync complete.")


# -----------------------------------------------------------
#  AWS SYNC
# -----------------------------------------------------------
def sync_to_aws(schema, dry_run=False):
    """Sync files to AWS S3/DynamoDB. When dry_run=True the function will only print
    what it would do instead of making network calls.
    """
    if boto3 is None and not dry_run:
        raise ImportError("boto3 not installed. Run `pip install boto3`")

    s3_bucket = os.getenv("AWS_S3_BUCKET", "predictive-maintenance-data")
    dynamodb_table = os.getenv("AWS_DYNAMO_TABLE", "PredictiveMaintenanceLabels")

    if dry_run:
        print("DRY RUN: AWS sync - no network operations will be performed.")
        for file in os.listdir(DATA_DIR):
            if file.endswith(".csv"):
                key = f"data/{file}"
                print(f"DRY RUN: Would upload {file} to S3 bucket {s3_bucket}/{key}")

        for file in os.listdir(DATA_DIR):
            if file.startswith("labels_") and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                sensor_type = file.replace("labels_", "").replace(".csv", "")
                print(f"DRY RUN: Would sync {len(df)} labeled rows to DynamoDB table {dynamodb_table} for sensor {sensor_type}")

        print("DRY RUN: AWS sync complete.")
        return

    # S3 client
    s3 = boto3.client("s3")

    # Upload CSVs to S3
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            local_path = os.path.join(DATA_DIR, file)
            key = f"data/{file}"
            try:
                s3.upload_file(local_path, s3_bucket, key)
                print(f"✅ Uploaded {file} to S3 bucket {s3_bucket}/{key}")
            except NoCredentialsError:
                print("❌ AWS credentials not found. Configure with `aws configure`.")

    # DynamoDB: upload labeled data
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(dynamodb_table)

    for file in os.listdir(DATA_DIR):
        if file.startswith("labels_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            sensor_type = file.replace("labels_", "").replace(".csv", "")
            for _, row in df.iterrows():
                item = {
                    "sensor": sensor_type,
                    "time": str(row.get("time")),
                    "label": int(row.get("label", 0)),
                    "iso_score": float(row.get("iso_score", 0)),
                    "comment": str(row.get("comment", "")),
                    "uploaded_at": datetime.datetime.utcnow().isoformat()
                }
                table.put_item(Item=item)
            print(f"✅ Synced {len(df)} labeled rows to DynamoDB table {dynamodb_table}")

    print("🌩️ AWS sync complete.")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, choices=["firebase", "aws"], required=True, help="Cloud provider")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without performing network operations")
    args = parser.parse_args()

    schema = load_schema()

    if args.provider == "firebase":
        sync_to_firebase(schema, dry_run=args.dry_run)
    elif args.provider == "aws":
        sync_to_aws(schema, dry_run=args.dry_run)
