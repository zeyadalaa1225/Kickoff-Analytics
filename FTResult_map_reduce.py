import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

PRE_MATCH_FEATURES = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home",
    "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway",
]
TARGET = "FTResult"
K = 8
N_PARTITIONS = 4


def mapper(args):
    partition_X, partition_y, test_point, k, partition_id = args

    diff = partition_X - test_point
    distances = np.sqrt(np.sum(diff ** 2, axis=1))

    if len(distances) <= k:
        local_top_k_idx = np.argsort(distances)
    else:
        local_top_k_idx = np.argpartition(distances, k)[:k]
        local_top_k_idx = local_top_k_idx[np.argsort(distances[local_top_k_idx])]

    emitted_pairs = [(distances[i], int(partition_y[i])) for i in local_top_k_idx]
    return emitted_pairs


def shuffle_and_sort(mapper_outputs):
    all_pairs = []
    for pairs in mapper_outputs:
        all_pairs.extend(pairs)
    all_pairs.sort(key=lambda x: x[0])
    return all_pairs


def reducer(sorted_pairs, k):
    global_top_k = sorted_pairs[:k]

    vote_counts = Counter()
    for dist, label in global_top_k:
        vote_counts[label] += 1

    predicted_label = max(vote_counts, key=vote_counts.get)
    return predicted_label


def predict_one_mapreduce(test_point, partitions_X, partitions_y, k, pool):
    mapper_args = [
        (partitions_X[i], partitions_y[i], test_point, k, i)
        for i in range(len(partitions_X))
    ]

    mapper_outputs = pool.map(mapper, mapper_args)
    sorted_pairs = shuffle_and_sort(mapper_outputs)
    prediction = reducer(sorted_pairs, k)
    return prediction


def predict_all(X_test, partitions_X, partitions_y, k, n_workers):
    predictions = []
    with mp.Pool(processes=n_workers) as pool:
        for i, test_point in enumerate(X_test):
            pred = predict_one_mapreduce(test_point, partitions_X, partitions_y, k, pool)
            predictions.append(pred)
            if (i + 1) % 50 == 0:
                print(f"  Predicted {i+1}/{len(X_test)} test points...")
    return np.array(predictions)


def main():
    print("=" * 60)
    print("  Distributed KNN-8 via MapReduce (Pure Python)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    train = pd.read_csv("Matches_cleaned.csv")
    test  = pd.read_csv("Matches_test.csv")

    X_train_raw = train[PRE_MATCH_FEATURES].values
    y_train_raw = train[TARGET].values
    X_test_raw  = test[PRE_MATCH_FEATURES].values
    y_test_raw  = test[TARGET].values

    print("[2/5] Encoding labels and scaling features...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc  = le.transform(y_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    print(f"[3/5] Partitioning {len(X_train_scaled)} training rows into {N_PARTITIONS} partitions...")
    partitions_X = np.array_split(X_train_scaled, N_PARTITIONS)
    partitions_y = np.array_split(y_train_enc,    N_PARTITIONS)
    for i, p in enumerate(partitions_X):
        print(f"  Partition {i}: {len(p)} rows")

    print(f"\n[4/5] Running MapReduce KNN-{K} on {len(X_test_scaled)} test points ({N_PARTITIONS} parallel workers)...")
    t0 = time.time()
    predictions = predict_all(X_test_scaled, partitions_X, partitions_y, K, N_PARTITIONS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print("\n[5/5] Results")
    print("=" * 60)
    acc = accuracy_score(y_test_enc, predictions)
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test_enc, predictions, target_names=le.classes_))


if __name__ == "__main__":
    mp.freeze_support()
    main()