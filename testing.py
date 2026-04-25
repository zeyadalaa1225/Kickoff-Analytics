from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit

# Import all your existing functions
from data_cleaning import cleaning_pipeline
from FTResult_predictor import (
    create_spark, load_data, base_features,
    compute_team_stats, apply_team_stats,
    build_pipeline, train_model, evaluate,
    NUMERIC_FEATURES, TARGET
)

def run_test_evaluation():
    spark = create_spark()


    test_df = load_data(spark, "Matches_test.csv")
    test_df = cleaning_pipeline(test_df)
    test_df = base_features(test_df)


    train_df = load_data(spark, "Matches_cleaned.csv")
    train_df = base_features(train_df)

    home_stats, away_stats = compute_team_stats(train_df)


    test_df = apply_team_stats(test_df, home_stats, away_stats)


    class_counts = train_df.select(*(NUMERIC_FEATURES + ["Division", TARGET])) \
                           .groupBy(TARGET).count().collect()
    total = sum(r["count"] for r in class_counts)
    class_freq = {r[TARGET]: r["count"] for r in class_counts}
    class_weight = {k: total / (3 * v) for k, v in class_freq.items()}

    df_model_test = test_df.select(*(NUMERIC_FEATURES + ["Division", TARGET]))
    df_model_test = df_model_test.withColumn(
        "classWeight",
        when(col(TARGET) == "H", class_weight["H"])
        .when(col(TARGET) == "D", class_weight["D"])
        .otherwise(class_weight["A"])
    )


    df_model_train = train_df.select(*(NUMERIC_FEATURES + ["Division", TARGET]))
    df_model_train = df_model_train.withColumn(
        "classWeight",
        when(col(TARGET) == "H", class_weight["H"])
        .when(col(TARGET) == "D", class_weight["D"])
        .otherwise(class_weight["A"])
    )

    pipeline = build_pipeline(NUMERIC_FEATURES)
    pipeline_model = pipeline.fit(df_model_train)

    train_prep = pipeline_model.transform(df_model_train).select("features", "label", "classWeight")
    test_prep  = pipeline_model.transform(df_model_test).select("features", "label", "classWeight")


    W, b = train_model(train_prep, n_classes=3)


    test_acc = evaluate(test_prep, W, b)
    print(f"\n{'='*40}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*40}\n")

    spark.stop()
    return test_acc

if __name__ == "__main__":
    run_test_evaluation()