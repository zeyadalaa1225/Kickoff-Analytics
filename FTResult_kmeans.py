import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

X_train = train[ALL_FEATURES].values
y_train = train[TARGET].values
X_test  = test[ALL_FEATURES].values
y_test  = test[TARGET].values
K = 3
N_PARTITIONS = 16
MAX_ITER = 20
TOL = 1e-4


def map_assign(x, means):
    means_arr = np.array(means, dtype=np.float64)
    x_arr     = np.array(x,     dtype=np.float64)
    dists     = np.sum((means_arr - x_arr) ** 2, axis=1)
    cluster_id = int(np.argmin(dists))
    return (cluster_id, x_arr)




def combine_local(partition_iter):
    rows = list(partition_iter)
    if not rows:
        return iter([])

    accum = {}
    for cluster_id, x_arr in rows:
        if cluster_id not in accum:
            accum[cluster_id] = (x_arr.copy(), 1)
        else:
            s, cnt = accum[cluster_id]
            accum[cluster_id] = (s + x_arr, cnt + 1)

    for cluster_id, (s, cnt) in accum.items():
        yield (cluster_id, (s, cnt))




def reduce_global(a, b):
    sum_a, cnt_a = a
    sum_b, cnt_b = b
    return (sum_a + sum_b, cnt_a + cnt_b)




def run_kmeans(data_rdd, k, max_iter, tol):
    sample = data_rdd.takeSample(False, k, seed=42)
    means  = [np.array(s, dtype=np.float64) for s in sample]

    for iteration in range(max_iter):
        means_broadcast = [m.tolist() for m in means]

        mapped_rdd   = data_rdd.map(lambda x: map_assign(x, means_broadcast))
        combined_rdd = mapped_rdd.mapPartitions(combine_local)
        reduced_rdd  = combined_rdd.reduceByKey(reduce_global)

        new_means_list = reduced_rdd.collect()

        new_means = [None] * k
        for cluster_id, (s, cnt) in new_means_list:
            new_means[cluster_id] = s / cnt

        for i in range(k):
            if new_means[i] is None:
                new_means[i] = means[i]

        shift = max(
            np.sqrt(np.sum((new_means[i] - means[i]) ** 2))
            for i in range(k)
        )
        print(f"  Iteration {iteration + 1}: mean shift = {shift:.6f}")
        means = new_means

        if shift < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return means

def main():
    spark = (
        SparkSession.builder
        .appName("KMeans_BigData_MapReduce")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(N_PARTITIONS))
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")


    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train).astype(np.int32)
    y_test_enc  = le.transform(y_test).astype(np.int32)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)


    print("Fitting KMeans...")
    train_rdd = sc.parallelize(
        [row.tolist() for row in X_train_scaled], numSlices=N_PARTITIONS
    ).cache()
    means = run_kmeans(train_rdd, K, MAX_ITER, TOL)


    means_broadcast = [m.tolist() for m in means]

    train_rdd_labeled = sc.parallelize(
        list(zip([row.tolist() for row in X_train_scaled], y_train_enc.tolist())),
        numSlices=N_PARTITIONS
    )


    cluster_true_pairs = train_rdd_labeled \
        .map(lambda x_y: (map_assign(x_y[0], means_broadcast)[0], x_y[1])) \
        .collect()

 
    cluster_to_class = {}
    for cluster_id in range(K):
        labels_in_cluster = [label for cid, label in cluster_true_pairs if cid == cluster_id]
        if labels_in_cluster:
            majority_class = np.bincount(labels_in_cluster, minlength=len(le.classes_)).argmax()
            cluster_to_class[cluster_id] = majority_class

    print("Cluster -> Class mapping (from train):")
    for cid, cls in cluster_to_class.items():
        print(f"  Cluster {cid} -> '{le.classes_[cls]}'")


    print("\nPredicting on test set...")
    test_rdd = sc.parallelize(
        [row.tolist() for row in X_test_scaled], numSlices=N_PARTITIONS
    )

    cluster_labels = np.array(
        test_rdd.map(lambda x, m=means_broadcast: map_assign(x, m)[0])
                .collect(),
        dtype=np.int32
    )

 
    predicted_classes = np.array(
        [cluster_to_class.get(cid, 0) for cid in cluster_labels],
        dtype=np.int32
    )

    
    accuracy = np.mean(predicted_classes == y_test_enc)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    spark.stop()


if __name__ == "__main__":
    main()