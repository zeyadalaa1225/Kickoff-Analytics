from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, median

def compute_train_stats(train_df):
    stats = {}
    normal_columns = [
        c for c in ["HomeElo", "AwayElo",
                     "Form3Home", "Form5Home",
                     "Form3Away", "Form5Away"]
        if c in train_df.columns
    ]
    numeric_cols = [
        f.name for f in train_df.schema.fields
        if str(f.dataType) in ("DoubleType()", "FloatType()",
                               "IntegerType()", "LongType()")
        and f.name not in normal_columns
    ]
    string_cols = [
        f.name for f in train_df.schema.fields
        if str(f.dataType) == "StringType()"
    ]
   
    mean_row = train_df.select(
            [mean(col(c)).alias(c) for c in normal_columns]
        ).collect()[0].asDict()
    stats["mean_values"] = mean_row

    median_values = {}
    for c in numeric_cols:
        result = train_df.approxQuantile(c, [0.5], 0.001)
        median_values[c] = result[0]

    stats["median_values"] = median_values
    all_numeric = normal_columns + numeric_cols
    iqr_bounds = {}
    for c in all_numeric:
        q1, q3 = train_df.approxQuantile(c, [0.25, 0.75], 0.001)
        iqr = q3 - q1
        iqr_bounds[c] = {
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
        }
    stats["iqr_bounds"] = iqr_bounds
    total = train_df.count()
    null_count_row = train_df.select([
        count(when(isnull(c), c)).alias(c) for c in train_df.columns
    ]).collect()[0].asDict()
 
    null_frac = {c: null_count_row[c] / total for c in train_df.columns}
    stats["string_cols"] = string_cols
    stats["normal_columns"] = normal_columns
    stats["numeric_cols"] = numeric_cols
 
    stats["cols_to_keep"] = [
        c for c in train_df.columns if null_frac.get(c, 0) <= 0.50
    ]
 
    stats["low_null_cols"] = [
        c for c in train_df.columns
        if null_frac.get(c, 0) < 0.05
    ]
 
    return stats

    


def cleaning_pipeline(df, stats, label="train"):
    total_rows = df.count()
    print(total_rows, "rows in original Matches.csv" )
    # ─────────────────────────────────────────────
    # 1. DROP DUPLICATES
    # ─────────────────────────────────────────────
    df     = df.dropDuplicates()
    total_rows_after_removing_dublication = df.count()
    print(f"Count of removed duplicates in Matches.csv: {total_rows - total_rows_after_removing_dublication}")

    # ─────────────────────────────────────────────
    # 2. DROP COLUMNS WITH > 50 % NULLS
    # ─────────────────────────────────────────────
    cols_to_keep = stats["cols_to_keep"]
    df = df.select(cols_to_keep)
    print(f"Count of removed columns with > 50% nulls: {len(df.columns) - len(cols_to_keep)}")
    # ─────────────────────────────────────────────
    # 3. DROP ROWS WHERE < 5 % NULL COLUMNS ARE NULL
    # ─────────────────────────────────────────────
    rows_count = df.count()
    low_null_cols = stats["low_null_cols"]
    df = df.dropna(subset=low_null_cols)
    rows_count_after_dropping = df.count()
    print(f"Count of removed rows with < 5% null columns: {rows_count - rows_count_after_dropping}")
    # ─────────────────────────────────────────────
    # 4. FILL NORMAL-DISTRIBUTION COLS WITH MEAN
    # ─────────────────────────────────────────────
    normal_values = {c: stats["mean_values"][c] for c in stats["normal_columns"] if c in df.columns}# 3a4an feh columns et4alet fel drop nulls 
    print(f"Filling normal-distribution columns with mean: {normal_values}")
    df = df.fillna(normal_values)

    # ─────────────────────────────────────────────
    # 5. FILL REMAINING NUMERIC COLS WITH MEDIAN
    # ─────────────────────────────────────────────
    median_values = {c: stats["median_values"][c] for c in stats["numeric_cols"] if c in df.columns}# 3a4an feh columns et4alet fel drop nulls 
    print(f"Filling remaining numeric columns with median: {median_values}")
    df = df.fillna(median_values)

    # ─────────────────────────────────────────────
    # 6. FILL REMAINING STRING/CATEGORICAL COLS WITH "Unknown"
    # ─────────────────────────────────────────────
    string_cols = stats["string_cols"]
    print(f"Filling remaining string columns with 'Unknown': {string_cols}")
    df = df.fillna({c: "Unknown" for c in string_cols if c in df.columns})# 3a4an feh columns et4alet fel drop nulls

    # ─────────────────────────────────────────────
    # 7. DROP INVALID VALUES IN FTResult
    # ─────────────────────────────────────────────


    valid_results = ["H", "D", "A"]
    invalid_result_count = df.filter(~col("FTResult").isin(valid_results)).count()
    if invalid_result_count > 0:
        print(f"[WARN] {invalid_result_count} rows have unexpected FTResult values — dropping")
        df = df.filter(col("FTResult").isin(valid_results))

    # ─────────────────────────────────────────────
    # 8. Elo columns must be positive 
    # ─────────────────────────────────────────────

    for elo_col in ["HomeElo", "AwayElo"]:
       
        invalid_elo = df.filter(col(elo_col) <= 0).count()
        if invalid_elo > 0:
            print(f"[WARN] {invalid_elo} rows have non-positive {elo_col} — replacing with column mean")
            valid_mean = stats["mean_values"][elo_col]
            df = df.withColumn(
                elo_col,
                when(col(elo_col) > 0, col(elo_col)).otherwise(valid_mean)
            )

    # ─────────────────────────────────────────────
    # 9. Form columns must be non-negative 
    # ─────────────────────────────────────────────

    form_cols = [c for c in ["Form3Home","Form5Home","Form3Away","Form5Away"] if c in df.columns]
    for fc in form_cols:
        negative_count = df.filter(col(fc) < 0).count()
        if negative_count > 0:
            print(f"[WARN] {negative_count} rows have negative {fc} — replacing with 0")
            mean_value = stats["mean_values"][fc]
            print(f"Filling negative {fc} values with 0 (mean was {mean_value})")
            df = df.withColumn(fc, when(col(fc) < 0, mean_value).otherwise(col(fc)))

    # ─────────────────────────────────────────────
    # 10. OUTLIER CHECK 
    # ─────────────────────────────────────────────

    
    # iqr_bounds = stats["iqr_bounds"]
    # all_numeric_in_df = [
    #     c for c in (stats["normal_columns"] + stats["numeric_cols"])
    #     if c in df.columns
    # ]
 
    # for c in all_numeric_in_df:
    #     if c not in iqr_bounds:
    #         continue
    #     lower = iqr_bounds[c]["lower"]
    #     upper = iqr_bounds[c]["upper"]
 
    #     outliers_low  = df.filter(col(c) < lower).count()
    #     outliers_high = df.filter(col(c) > upper).count()
 
    #     if outliers_low + outliers_high > 0:
    #         print(f"  • {c}: {outliers_low} below lower fence ({lower:.4f}), "
    #               f"{outliers_high} above upper fence ({upper:.4f}) — capped")
    #         df = df.withColumn(
    #             c,
    #             when(col(c) < lower, lower)
    #            .when(col(c) > upper, upper)
    #            .otherwise(col(c))
    #         )
    #     else:
    #         print(f"  • {c}: no outliers detected ✓")
    # ─────────────────────────────────────────────
    # 11. FINAL NULL CHECK
    # ─────────────────────────────────────────────
    print("\n=== Final null check ===")
    df.select([
        count(when(isnull(c), c)).alias(c) for c in df.columns
    ]).show(vertical=True)

    return df

if __name__ == "__main__":
    spark = SparkSession.builder \
    .appName("FootballMatchCleaning") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")


    df = spark.read.csv("Matches.csv", header=True, inferSchema=True)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    stats = compute_train_stats(train_df)
    train_cleaned = cleaning_pipeline(train_df, stats, label="train")
    test_cleaned  = cleaning_pipeline(test_df,  stats, label="test")
    test_cleaned.toPandas().to_csv("Matches_test.csv", index=False)
    train_cleaned.toPandas().to_csv("Matches_cleaned.csv", index=False)
    print(train_cleaned.count(), "rows in cleaned Matches_train.csv")
    print(test_cleaned.count(), "rows in cleaned Matches_test.csv")
    print(train_cleaned.columns)
    print("\nDone  →  Matches_train_cleaned.csv  &  Matches_test_cleaned.csv")
    spark.stop()