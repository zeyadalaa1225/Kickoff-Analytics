"""
KNN with PySpark — Distributed S3 Read (LocalStack or AWS)
===========================================================
Spark reads CSV partitions DIRECTLY from S3 in parallel.
No data is downloaded to the driver first.

Usage:
  LocalStack:  python knn_s3_spark.py --mode localstack
  Real AWS:    python knn_s3_spark.py --mode aws
  Re-seed:     python knn_s3_spark.py --mode localstack --seed
"""

import argparse
import time
from collections import Counter

import boto3
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BUCKET_NAME = "kickoff-analytics"
TRAIN_KEY   = "Matches_cleaned.csv"
TEST_KEY    = "Matches_test.csv"

LOCALSTACK_URL        = "http://localhost:4566"
LOCALSTACK_ACCESS_KEY = "test"
LOCALSTACK_SECRET_KEY = "test"
LOCALSTACK_REGION     = "us-east-1"

# Spark reads from s3a:// URIs directly (distributed)
TRAIN_S3_PATH = f"s3a://{BUCKET_NAME}/{TRAIN_KEY}"
TEST_S3_PATH  = f"s3a://{BUCKET_NAME}/{TEST_KEY}"

PRE_MATCH_FEATURES = [
    "Division", "MatchDate", "HomeTeam", "AwayTeam",
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home",
    "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway",
]
TARGET       = "FTResult"
K            = 8
N_PARTITIONS = 16


# ─────────────────────────────────────────────────────────────
# SPARK SESSION — configured to talk to S3 / LocalStack
# ─────────────────────────────────────────────────────────────
def build_spark(mode: str) -> SparkSession:
    """
    SparkSession with hadoop-aws so Spark reads s3a:// paths
    directly from S3 in parallel across partitions.
    Jars are auto-downloaded from Maven on first run.
    """
    builder = (
        SparkSession.builder
        .appName("KNN_Distributed_S3")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(N_PARTITIONS))
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    )

    if mode == "localstack":
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.endpoint",              LOCALSTACK_URL)
            .config("spark.hadoop.fs.s3a.access.key",            LOCALSTACK_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key",            LOCALSTACK_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access",     "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled","false")
        )
    else:
        # Real AWS — pull creds from ~/.aws/credentials or environment variables
        session = boto3.Session()
        creds   = session.get_credentials().get_frozen_credentials()
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key",            creds.access_key)
            .config("spark.hadoop.fs.s3a.secret.key",            creds.secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access",     "false")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled","true")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────
# SEED HELPER  (LocalStack only)
# ─────────────────────────────────────────────────────────────
def seed_localstack(
    train_local: str = r"D:\cmp_fourth_year\Big_Data\Kickoff-Analytics\Matches_cleaned.csv",
    test_local:  str = r"D:\cmp_fourth_year\Big_Data\Kickoff-Analytics\Matches_test.csv",
):
    print("[SEED] Uploading local CSVs to LocalStack S3...")
    s3 = boto3.client(
        "s3",
        endpoint_url=LOCALSTACK_URL,
        aws_access_key_id=LOCALSTACK_ACCESS_KEY,
        aws_secret_access_key=LOCALSTACK_SECRET_KEY,
        region_name=LOCALSTACK_REGION,
    )
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET_NAME not in buckets:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"  Bucket '{BUCKET_NAME}' created.")
    for local, key in [(train_local, TRAIN_KEY), (test_local, TEST_KEY)]:
        print(f"  Uploading {local} → s3://{BUCKET_NAME}/{key} ...")
        with open(local, "rb") as f:
            s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=f)
    print("  Seeding complete.\n")


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(train: pd.DataFrame, test: pd.DataFrame):
    for df in [train, test]:
        df["MatchDate"] = pd.to_datetime(df["MatchDate"], dayfirst=True)
        df["DayOfWeek"] = df["MatchDate"].dt.dayofweek
        df["Month"]     = df["MatchDate"].dt.month
        df["Year"]      = df["MatchDate"].dt.year
        df["DayOfYear"] = df["MatchDate"].dt.dayofyear

    DATE_FEATURES = ["DayOfWeek", "Month", "Year", "DayOfYear"]
    CAT_FEATURES  = ["Division", "HomeTeam", "AwayTeam"]
    NUM_FEATURES  = [
        f for f in PRE_MATCH_FEATURES
        if f not in CAT_FEATURES + ["MatchDate"]
    ] + DATE_FEATURES

    for col in CAT_FEATURES:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    return train, test, CAT_FEATURES + NUM_FEATURES


# ─────────────────────────────────────────────────────────────
# KNN MAP-REDUCE  (runs inside Spark partitions — distributed)
# ─────────────────────────────────────────────────────────────
def map_local_topk(partition_iter, test_point, k):
    """MAP: each partition independently finds its local top-k."""
    rows = list(partition_iter)
    if not rows:
        return iter([])
    labels = np.array([r[0] for r in rows], dtype=np.int32)
    X_part = np.array([r[1] for r in rows], dtype=np.float64)
    dists  = np.sqrt(np.sum((X_part - test_point) ** 2, axis=1))
    idx    = np.argsort(dists)[:k]
    yield [(float(dists[i]), int(labels[i])) for i in idx]


def reduce_global_topk(a, b, k):
    """REDUCE: merge two local top-k lists into a global top-k."""
    merged = a + b
    merged.sort(key=lambda x: x[0])
    return merged[:k]


def predict_one(train_rdd, test_point, k):
    local_topk_rdd = train_rdd.mapPartitions(
        lambda part: map_local_topk(part, test_point, k)
    )
    global_topk = local_topk_rdd.reduce(
        lambda a, b: reduce_global_topk(a, b, k)
    )
    votes = Counter(label for _, label in global_topk)
    return max(votes, key=votes.get)


def predict_all(train_rdd, X_test, k):
    preds = []
    n = len(X_test)
    for i, test_point in enumerate(X_test):
        preds.append(predict_one(train_rdd, test_point, k))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  Predicted {i+1}/{n}...")
    return np.array(preds, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(mode: str, seed: bool):
    print(f"\n{'='*60}")
    print(f"  Mode : {mode.upper()}")
    print(f"  Train: {TRAIN_S3_PATH}")
    print(f"  Test : {TEST_S3_PATH}")
    print(f"{'='*60}\n")

    # 0 — seed LocalStack if requested
    if seed:
        seed_localstack()

    # 1 — build Spark with S3A support
    print("[1/5] Starting Spark with S3A connector...")
    spark = build_spark(mode)
    sc    = spark.sparkContext
    print(f"  Spark version : {spark.version}")
    print(f"  Cores in use  : {sc.defaultParallelism}")

    # 2 — read CSVs DIRECTLY from S3 (Spark splits file into partitions)
    print("\n[2/5] Reading CSVs from S3 via Spark (distributed)...")
    train_sdf = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(TRAIN_S3_PATH)
    )
    test_sdf = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(TEST_S3_PATH)
    )
    print(f"  Train S3 partitions : {train_sdf.rdd.getNumPartitions()}")
    print(f"  Test  S3 partitions : {test_sdf.rdd.getNumPartitions()}")
    train_count = train_sdf.count()
    test_count  = test_sdf.count()
    print(f"  Train rows: {train_count:,}  |  Test rows: {test_count:,}")

    # convert to Pandas for sklearn preprocessing
    train = train_sdf.toPandas()
    test  = test_sdf.toPandas()

    # 3 — feature engineering + scaling
    print("\n[3/5] Engineering features & scaling...")
    train, test, ALL_FEATURES = engineer_features(train, test)

    _, test_sample = train_test_split(
        test, test_size=0.01, random_state=42, stratify=test[TARGET]
    )

    X_train = train[ALL_FEATURES].values
    y_train = train[TARGET].values
    X_test  = test_sample[ALL_FEATURES].values
    y_test  = test_sample[TARGET].values

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int32)
    y_test_enc  = le.transform(y_test).astype(np.int32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4 — distribute training data across Spark partitions
    print(f"\n[4/5] Distributing {len(X_train_scaled):,} training rows "
          f"across {N_PARTITIONS} Spark partitions...")
    train_data = list(zip(y_train_enc.tolist(), X_train_scaled.tolist()))
    train_rdd  = sc.parallelize(train_data, numSlices=N_PARTITIONS).cache()

    sizes = train_rdd.glom().map(len).collect()
    for i, s in enumerate(sizes):
        print(f"  Partition {i:02d}: {s:,} rows")

    # 5 — distributed KNN MapReduce
    print(f"\n[5/5] Running Distributed KNN-{K} "
          f"({len(X_test_scaled)} test points × {N_PARTITIONS} partitions)...")
    t0    = time.time()
    preds = predict_all(train_rdd, X_test_scaled, K)
    print(f"\n  Finished in {time.time()-t0:.1f}s")

    # results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy : {accuracy_score(y_test_enc, preds):.4f}")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))

    spark.stop()
    print("Spark stopped.")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["localstack", "aws"], default="localstack",
        help="'localstack' for local dev  |  'aws' for real AWS S3",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Upload local CSVs to LocalStack S3 before running",
    )
    args = parser.parse_args()
    main(mode=args.mode, seed=args.seed)