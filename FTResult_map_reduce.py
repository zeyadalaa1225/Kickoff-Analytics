import time
from collections import Counter

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PRE_MATCH_FEATURES = [
    "Division","MatchDate","HomeTeam","AwayTeam",
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home",
    "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway"
]

TARGET = "FTResult"
train = pd.read_csv("Matches_cleaned.csv")
test  = pd.read_csv("Matches_test.csv")




for df in [train, test]:
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], dayfirst=True)
    df["DayOfWeek"]   = df["MatchDate"].dt.dayofweek          
    df["Month"]       = df["MatchDate"].dt.month
    df["Year"]        = df["MatchDate"].dt.year
    df["DayOfYear"]   = df["MatchDate"].dt.dayofyear

DATE_FEATURES   = ["DayOfWeek", "Month", "Year", "DayOfYear"]
CAT_FEATURES    = ["Division", "HomeTeam", "AwayTeam"]
NUM_FEATURES    = [f for f in PRE_MATCH_FEATURES
                   if f not in CAT_FEATURES + ["MatchDate"]] + DATE_FEATURES

cat_encoders = {}
for col in CAT_FEATURES:
    le_col = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le_col.fit(combined)
    train[col] = le_col.transform(train[col])
    test[col]  = le_col.transform(test[col])
    cat_encoders[col] = le_col
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
_, test_sample = train_test_split(test, test_size=0.1, random_state=42, stratify=test[TARGET])
X_train = train[ALL_FEATURES].values
y_train = train[TARGET].values
X_test  = test_sample[ALL_FEATURES].values
y_test  = test_sample[TARGET].values
K = 8
N_PARTITIONS = 16


def map_local_topk(partition_iter, test_point, k):
    rows = list(partition_iter)
    if not rows:
        return iter([])
    labels  = np.array([r[0] for r in rows], dtype=np.int32)
    X_part  = np.array([r[1] for r in rows], dtype=np.float64)
    dists = np.sqrt(np.sum((X_part - test_point) ** 2, axis=1))
    top_k_idx  = np.argsort(dists)[:k]
    local_topk = [(float(dists[i]), int(labels[i])) for i in top_k_idx]
    yield local_topk


 
 
 
 
 
 

def reduce_global_topk(local_topk_a, local_topk_b, k):
    merged = local_topk_a + local_topk_b
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


 
 
 

def main():

    spark = (
        SparkSession.builder
        .appName("KNN_BigData_MapReduce")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(N_PARTITIONS))
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int32)
    y_test_enc  = le.transform(y_test).astype(np.int32)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

     

    train_data = list(zip(y_train_enc.tolist(), [row for row in X_train_scaled]))
    train_rdd  = sc.parallelize(train_data, numSlices=N_PARTITIONS).cache()

     
    for i, part in enumerate(train_rdd.glom().collect()):
        print(f"  Split {i}: {len(part)} rows")

     
    print(f"\n[4/5] Running KNN-{K} — one test point at a time...")
    t0    = time.time()
    preds = predict_all(train_rdd, X_test_scaled, K)
    print(f"  Done in {time.time() - t0:.1f}s")

     
    print("\n[5/5] Results")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_test_enc, preds))
    print(classification_report(y_test_enc, preds, target_names=le.classes_))

    spark.stop()


if __name__ == "__main__":
    main()