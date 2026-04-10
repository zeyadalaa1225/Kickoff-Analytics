from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, median

spark = SparkSession.builder \
    .appName("FootballMatchCleaning") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


df = spark.read.csv("Matches.csv", header=True, inferSchema=True)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

test_df.toPandas().to_csv("Matches_test.csv", index=False)

df = train_df

total_rows = df.count()
print(total_rows, "rows in train Matches.csv")

total_rows = df.count()
print(total_rows, "rows in original Matches.csv" )
# ─────────────────────────────────────────────
# 3. DUPLICATE REMOVAL
# ─────────────────────────────────────────────
df     = df.dropDuplicates()
total_rows_after_removing_dublication = df.count()
print(f"Count of removed duplicates in Matches.csv: {total_rows - total_rows_after_removing_dublication}")
# ─────────────────────────────────────────────
# 4. NULL AUDIT  
# ─────────────────────────────────────────────
null_counts = df.select([
    count(when(isnull(c) | isnan(c) if dict(df.dtypes)[c] in ("double","float","int","bigint")
               else isnull(c), c)).alias(c)
    for c in df.columns
])
null_counts.show(vertical=True)

# ─────────────────────────────────────────────
# 5. DROP COLUMNS WITH > 50 % NULLS
# ─────────────────────────────────────────────
columns_count = df.count()
null_count_row = df.select([
    count(when(isnull(c), c)).alias(c) for c in df.columns
]).collect()[0].asDict()

null_frac = {c: null_count_row[c] / total_rows for c in df.columns}

cols_to_keep = [c for c in df.columns
                if (null_frac.get(c, 0)) <= 0.50]
df = df.select(cols_to_keep)
columns_count_after_dropping_nulls = df.count()
print(f"Count of removed columns with > 50% nulls: {len(df.columns) - len(cols_to_keep)}")
# ─────────────────────────────────────────────
# 6. DROP ROWS WHERE < 5 % NULL COLUMNS ARE NULL
# ─────────────────────────────────────────────
rows_count = df.count()
threshold   = 0.05 * total_rows

low_null_cols = [
    c for c in df.columns
    if df.filter(isnull(col(c))).count() < threshold
]

df = df.dropna(subset=low_null_cols)
rows_count_after_dropping = df.count()
print(f"Count of removed rows with < 5% null columns: {rows_count - rows_count_after_dropping}")
# ─────────────────────────────────────────────
# 7. FILL NORMAL-DISTRIBUTION COLS WITH MEAN
# ─────────────────────────────────────────────
normal_columns = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home",
    "Form3Away", "Form5Away"
]


mean_values = df.select(
    [mean(col(c)).alias(c) for c in normal_columns]
).collect()[0].asDict()

df = df.fillna(mean_values)

# ─────────────────────────────────────────────
# 8. FILL REMAINING NUMERIC COLS WITH MEDIAN
# ─────────────────────────────────────────────
numeric_cols = [
    f.name for f in df.schema.fields
    if str(f.dataType) in ("DoubleType()", "FloatType()",
                           "IntegerType()", "LongType()")
    and f.name not in normal_columns
]

for c in numeric_cols:
    median_val = df.approxQuantile(c, [0.5], 0.001)[0]
    df = df.fillna({c: median_val})

# ─────────────────────────────────────────────
# 9. FILL REMAINING STRING/CATEGORICAL COLS WITH "Unknown"
# ─────────────────────────────────────────────
string_cols = [
    f.name for f in df.schema.fields
    if str(f.dataType) == "StringType()"
]

df = df.fillna("Unknown", subset=string_cols)

# ─────────────────────────────────────────────
# 10. DROP INVALID VALUES IN FTResult
# ─────────────────────────────────────────────


valid_results = ["H", "D", "A"]
invalid_result_count = df.filter(~col("FTResult").isin(valid_results)).count()
if invalid_result_count > 0:
    print(f"[WARN] {invalid_result_count} rows have unexpected FTResult values — dropping")
    df = df.filter(col("FTResult").isin(valid_results))

# ─────────────────────────────────────────────
# 11. Elo columns must be positive 
# ─────────────────────────────────────────────

for elo_col in ["HomeElo", "AwayElo"]:
    if elo_col in df.columns:
        invalid_elo = df.filter(col(elo_col) <= 0).count()
        if invalid_elo > 0:
            print(f"[WARN] {invalid_elo} rows have non-positive {elo_col} — replacing with column mean")
            valid_mean = df.filter(col(elo_col) > 0).select(mean(col(elo_col))).collect()[0][0]
            df = df.withColumn(
                elo_col,
                when(col(elo_col) > 0, col(elo_col)).otherwise(valid_mean)
            )

# ─────────────────────────────────────────────
# 12. Form columns must be non-negative 
# ─────────────────────────────────────────────

form_cols = [c for c in ["Form3Home","Form5Home","Form3Away","Form5Away"] if c in df.columns]
for fc in form_cols:
    negative_count = df.filter(col(fc) < 0).count()
    if negative_count > 0:
        print(f"[WARN] {negative_count} rows have negative {fc} — replacing with 0")
        df = df.withColumn(fc, when(col(fc) < 0, 0.0).otherwise(col(fc)))



# ─────────────────────────────────────────────
# 11. FINAL NULL CHECK
# ─────────────────────────────────────────────
print("\n=== Final null check ===")
df.select([
    count(when(isnull(c), c)).alias(c) for c in df.columns
]).show(vertical=True)




# ─────────────────────────────────────────────
# 12. SAVE CLEANED FILES  (via pandas — no winutils needed)
# ─────────────────────────────────────────────
df.toPandas().to_csv("Matches_cleaned.csv", index=False)

print("\nCleaned files written to  →  Matches_cleaned.csv  &  EloRatings_cleaned.csv")

spark.stop()