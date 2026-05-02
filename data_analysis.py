from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, lit ,floor
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ─── Spark Setup ───────────────────────────────────────────────────────────────
Spark = SparkSession.builder \
    .appName("FootballAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()
Spark.sparkContext.setLogLevel("WARN")
df = Spark.read.csv("Matches_cleaned.csv", header=True, inferSchema=True)
print(df.count(), "rows in cleaned Matches.csv")
 
"""
Columns:
['Division', 'MatchDate', 'HomeTeam', 'AwayTeam', 'HomeElo', 'AwayElo', 'Form3Home',
 'Form5Home', 'Form3Away', 'Form5Away', 'FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway',
 'HTResult', 'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed', 'OddHome', 'OddDraw', 'OddAway',
 'MaxHome', 'MaxDraw', 'MaxAway', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25', 'HandiSize',
 'HandiHome', 'HandiAway']
"""

# # ══════════════════════════════════════════════════════════════════════════════
# print("\n" + "="*70)
# print("SECTION 0 – DATASET INTRODUCTION & SURFACELEVEL OVERVIEW")
# print("="*70)
# # ══════════════════════════════════════════════════════════════════════════════
# # Q1 – Matches per Division

matches_per_division = (
    df.groupBy("Division")
    .agg(count("*").alias("match_count"))
    .orderBy(col("match_count").desc())
)

print("\nQ1 – Matches per Division")
matches_per_division.show(truncate=False)

# Convert to Pandas
matches_df = matches_per_division.toPandas()

# Plot
plt.figure(figsize=(12, 6))

bars = plt.bar(
    matches_df["Division"],
    matches_df["match_count"],
    color=sns.light_palette("blue", n_colors=len(matches_df), reverse=True)
)

plt.title(
    "Number of Matches per Division (2000–2025)\n",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel("Division")
plt.ylabel("Number of Matches")

plt.xticks(rotation=45, ha="right")

# Value labels
for i, v in enumerate(matches_df["match_count"]):
    plt.text(i, v + 1, str(v), ha='center')

plt.tight_layout()
plt.show()
# Q2 – Top 10 highestscoring teams
home_goals = df.groupBy("HomeTeam") \
    .agg(F.sum("HTHome").alias("total_goals"))

away_goals = df.groupBy("AwayTeam") \
    .agg(F.sum("HTAway").alias("total_goals")) \
    .withColumnRenamed("AwayTeam", "HomeTeam")

total_goals = home_goals.unionByName(away_goals) \
    .groupBy("HomeTeam") \
    .agg(F.sum("total_goals").alias("total_goals"))

top10_goals = total_goals.orderBy(col("total_goals").desc()) \
    .limit(10)

print("\nQ2 – Top 10 HighestScoring Teams")
top10_goals.show(truncate=False)

top10_pd = top10_goals.toPandas()

plt.figure(figsize=(12, 6))

bars = plt.bar(
    top10_pd["HomeTeam"],
    top10_pd["total_goals"],
    color=sns.light_palette("blue", n_colors=len(top10_pd), reverse=True)
)

plt.title("Top 10 HighestScoring Teams (2000–2025)")
plt.xlabel("Team")
plt.ylabel("Total Goals Scored")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(top10_pd["total_goals"]):
    plt.text(i, v + 1, str(v), ha='center')

plt.tight_layout()
plt.show()


# Q3 – Home vs Away goals for the top 15 teams
home_g = df.groupBy("HomeTeam") \
    .agg(F.sum("HTHome").alias("home_goals"))

away_g = df.groupBy("AwayTeam") \
    .agg(F.sum("HTAway").alias("away_goals")) \
    .withColumnRenamed("AwayTeam", "HomeTeam")

ha_df = home_g.join(away_g, "HomeTeam") \
    .orderBy(col("home_goals").desc()) \
    .limit(15)

print("\nQ3  Home vs Away Goals for Top 15 Teams")
ha_df.show(truncate=False)

ha_pd = ha_df.toPandas()

teams = ha_pd["HomeTeam"]
x = np.arange(len(teams))
width = 0.35

plt.figure(figsize=(14, 6))

plt.bar(
    x - width/2,
    ha_pd["home_goals"],
    width,
    label="Home Goals",
    color=sns.light_palette("blue", n_colors=len(teams), reverse=True)
)

plt.bar(
    x + width/2,
    ha_pd["away_goals"],
    width,
    label="Away Goals",
    color=sns.light_palette("orange", n_colors=len(teams), reverse=True)
)

plt.title("Home vs Away Goals for Top 15 Teams")
plt.xlabel("Team")
plt.ylabel("Goals")

plt.xticks(x, teams, rotation=45, ha="right")
plt.legend()

plt.tight_layout()
plt.show()

# Q4 – Average goals per match by division
avg_goals_per_division = df.groupBy("Division") \
    .agg(mean(col("HTHome") + col("HTAway")).alias("avg_goals_per_match")) \
    .orderBy(col("avg_goals_per_match").desc())

print("\nQ4 – Average Goals per Match by Division")
avg_goals_per_division.show(10, truncate=False)

avg_goals_df = avg_goals_per_division.toPandas()
overall_avg = avg_goals_df["avg_goals_per_match"].mean()

plt.figure(figsize=(16, 7))

plt.bar(
    avg_goals_df["Division"],
    avg_goals_df["avg_goals_per_match"],
    color=sns.light_palette("blue", n_colors=len(avg_goals_df), reverse=True)
)

plt.axhline(
    overall_avg,
    color="red",
    linestyle="",
    linewidth=1.5,
    label=f"Overall Avg: {overall_avg:.2f}"
)

plt.title("Average Goals per Match by Division (2000–2025)")
plt.xlabel("Division")
plt.ylabel("Average Goals per Match")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(avg_goals_df["avg_goals_per_match"]):
    plt.text(i, v + 0.03, f"{v:.2f}", ha="center")

plt.legend()
plt.tight_layout()
plt.show()

# Q5 – Which teams have played the most matches?
home_matches = df.groupBy("HomeTeam") \
    .agg(count("*").alias("match_count"))

away_matches = df.groupBy("AwayTeam") \
    .agg(count("*").alias("match_count")) \
    .select(
        col("AwayTeam").alias("HomeTeam"),
        col("match_count")
    )

total_matches = home_matches.unionByName(away_matches) \
    .groupBy("HomeTeam") \
    .agg(F.sum("match_count").alias("total_matches"))

most_active_teams = total_matches.orderBy(col("total_matches").desc())

print("\nQ5 – Teams with the Most Matches Played")
most_active_teams.show(10, truncate=False)

plot_df = most_active_teams.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["HomeTeam"],
    plot_df["total_matches"],
    color=sns.light_palette("blue", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams With Most Matches Played")
plt.xlabel("Team")
plt.ylabel("Total Matches Played")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["total_matches"]):
    plt.text(i, v + 1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q6 – Top 10 teams with the most wins
home_wins = df.groupBy("HomeTeam") \
    .agg(count(when(col("FTResult") == "H", 1)).alias("wins"))

away_wins = df.groupBy("AwayTeam") \
    .agg(count(when(col("FTResult") == "A", 1)).alias("wins")) \
    .select(
        col("AwayTeam").alias("HomeTeam"),
        col("wins")
    )

total_wins = home_wins.unionByName(away_wins) \
    .groupBy("HomeTeam") \
    .agg(F.sum("wins").alias("total_wins"))

top_winning_teams = total_wins.orderBy(col("total_wins").desc())

print("\nQ6 – Top 10 Teams with the Most Wins")
top_winning_teams.show(10, truncate=False)

plot_df = top_winning_teams.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["HomeTeam"],
    plot_df["total_wins"],
    color=sns.light_palette("green", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams with Most Wins (2000–2025)")
plt.xlabel("Team")
plt.ylabel("Total Wins")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["total_wins"]):
    plt.text(i, v + 1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q7 – Top 10 teams by number of home wins
home_wins_only = df.filter(col("FTResult") == "H") \
    .groupBy("HomeTeam") \
    .agg(count("*").alias("home_wins")) \
    .orderBy(col("home_wins").desc())

print("\nQ7 – Top 10 Teams by Home Wins")
home_wins_only.show(10, truncate=False)

top10_pd = home_wins_only.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    top10_pd["HomeTeam"],
    top10_pd["home_wins"],
    color=sns.light_palette("green", n_colors=len(top10_pd), reverse=True)
)

plt.title("Top 10 Teams by Home Wins")
plt.xlabel("Team")
plt.ylabel("Number of Home Wins")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(top10_pd["home_wins"]):
    plt.text(i, v + 1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q8 – Top 10 teams by average Elo per year
matches_year = df.withColumn(
    "Year",
    F.year(F.to_date("MatchDate"))
)

home_elo = matches_year.select(
    F.col("HomeTeam").alias("Team"),
    F.col("Year"),
    F.col("HomeElo").alias("Elo")
)

away_elo = matches_year.select(
    F.col("AwayTeam").alias("Team"),
    F.col("Year"),
    F.col("AwayElo").alias("Elo")
)

team_elo = home_elo.unionByName(away_elo) \
    .groupBy("Team", "Year") \
    .agg(F.mean("Elo").alias("MeanElo")) \
    .orderBy(col("MeanElo").desc())

print("\nQ – Top 10 Team-Year Elo Peaks")
team_elo.show(10, truncate=False)

plot_df = team_elo.limit(10).toPandas()

labels = plot_df["Team"] + " (" + plot_df["Year"].astype(str) + ")"

plt.figure(figsize=(13, 6))

plt.bar(
    labels,
    plot_df["MeanElo"],
    color=sns.light_palette("purple", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Team–Year Elo Peaks")
plt.xlabel("Team (Year)")
plt.ylabel("Average Elo")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["MeanElo"]):
    plt.text(i, v + 2, f"{v:.1f}", ha="center")

plt.tight_layout()
plt.show()

# Q9 – Teams with the most yellow cards
home_yellow = df.groupBy("HomeTeam") \
    .agg(F.sum("HomeYellow").alias("yellows"))

away_yellow = df.groupBy("AwayTeam") \
    .agg(F.sum("AwayYellow").alias("yellows")) \
    .withColumnRenamed("AwayTeam", "HomeTeam")

total_yellow = home_yellow.unionByName(away_yellow) \
    .groupBy("HomeTeam") \
    .agg(F.sum("yellows").alias("total_yellows")) \
    .orderBy(col("total_yellows").desc())

print("\nQ9  Top 10 Teams by Yellow Cards")
total_yellow.show(10, truncate=False)

plot_df = total_yellow.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["HomeTeam"],
    plot_df["total_yellows"],
    color=sns.light_palette("gold", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams by Yellow Cards (2000–2025)")
plt.xlabel("Team")
plt.ylabel("Total Yellow Cards")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["total_yellows"]):
    plt.text(i, v + 1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q10 – Teams with the most red cards
red_cards = df.select(
    F.col("HomeTeam").alias("Team"),
    F.col("HomeRed").cast("double").alias("Red")
).union(
    df.select(
        F.col("AwayTeam").alias("Team"),
        F.col("AwayRed").cast("double").alias("Red")
    )
)

total_red = red_cards.groupBy("Team") \
    .agg(F.sum("Red").alias("total_reds")) \
    .orderBy(col("total_reds").desc())

print("\nQ10 – Top 10 Teams by Red Cards")
total_red.show(10, truncate=False)

plot_df = total_red.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["Team"],
    plot_df["total_reds"],
    color=sns.light_palette("red", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams by Red Cards (2000–2025)")
plt.xlabel("Team")
plt.ylabel("Total Red Cards")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["total_reds"]):
    plt.text(i, v + 0.1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q11 – Average cards per division
cards_div = df.groupBy("Division").agg(
    mean(col("HomeYellow") + col("AwayYellow")).alias("avg_yellows_per_match"),
    mean(col("HomeRed") + col("AwayRed")).alias("avg_reds_per_match")
).orderBy(col("avg_yellows_per_match").desc())

print("\nQ11  Average Cards per Division")
cards_div.show(truncate=False)

cards_div = cards_div.toPandas()

x = np.arange(len(cards_div))
w = 0.4

plt.figure(figsize=(14, 6))

plt.bar(
    x - w/2,
    cards_div["avg_yellows_per_match"],
    w,
    label="Avg Yellows",
    color="gold"
)

plt.bar(
    x + w/2,
    cards_div["avg_reds_per_match"],
    w,
    label="Avg Reds",
    color="red"
)

plt.xticks(x, cards_div["Division"], rotation=45, ha="right")

plt.title("Average Cards per Match by Division")
plt.xlabel("Division")
plt.ylabel("Average Cards per Match")

plt.legend()
plt.tight_layout()
plt.show()

# Q12 – Most common fulltime scorelines (top 15)
scoreline = df.withColumn(
    "Scoreline",
    F.concat(col("FTHome").cast("string"), lit("-"), col("FTAway").cast("string"))
).groupBy("Scoreline") \
 .agg(count("*").alias("count")) \
 .orderBy(col("count").desc())

print("\nQ12  Top 15 Most Common Scorelines")
scoreline.show(15, truncate=False)

plot_df = scoreline.limit(15).toPandas()

plt.figure(figsize=(14, 6))

plt.bar(
    plot_df["Scoreline"],
    plot_df["count"],
    color=sns.color_palette("husl", len(plot_df))
)

plt.title("Top 15 Most Common Full-Time Scorelines")
plt.xlabel("Scoreline")
plt.ylabel("Frequency")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["count"]):
    plt.text(i, v + 10, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q13 – Top 10 teams with most comebacks (HT Loss → FT Win)
home_cb = df.filter((col("HTResult") == "A") & (col("FTResult") == "H")) \
    .groupBy("HomeTeam") \
    .agg(count("*").alias("comebacks"))

away_cb = df.filter((col("HTResult") == "H") & (col("FTResult") == "A")) \
    .groupBy("AwayTeam") \
    .agg(count("*").alias("comebacks")) \
    .withColumnRenamed("AwayTeam", "HomeTeam")

comebacks = home_cb.unionByName(away_cb) \
    .groupBy("HomeTeam") \
    .agg(F.sum("comebacks").alias("comebacks")) \
    .orderBy(col("comebacks").desc())

print("\nQ13  Top 10 Teams with Most Comebacks")
comebacks.show(10, truncate=False)

plot_df = comebacks.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["HomeTeam"],
    plot_df["comebacks"],
    color=sns.light_palette("coral", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams with Most Comebacks (HT Loss → FT Win)")
plt.xlabel("Team")
plt.ylabel("Comebacks")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["comebacks"]):
    plt.text(i, v + 0.1, str(v), ha="center")

plt.tight_layout()
plt.show()

# Q14 – Top 10 teams with most blown leads (HT Win → FT Loss)
home_bl = df.filter((col("HTResult") == "H") & (col("FTResult") == "A")) \
    .groupBy("HomeTeam") \
    .agg(count("*").alias("blown_leads"))

away_bl = df.filter((col("HTResult") == "A") & (col("FTResult") == "H")) \
    .groupBy("AwayTeam") \
    .agg(count("*").alias("blown_leads")) \
    .withColumnRenamed("AwayTeam", "HomeTeam")

blown = home_bl.unionByName(away_bl) \
    .groupBy("HomeTeam") \
    .agg(F.sum("blown_leads").alias("blown_leads")) \
    .orderBy(col("blown_leads").desc())

print("\nQ14  Top 10 Teams with Most Blown Leads")
blown.show(10, truncate=False)

plot_df = blown.limit(10).toPandas()

plt.figure(figsize=(12, 6))

plt.bar(
    plot_df["HomeTeam"],
    plot_df["blown_leads"],
    color=sns.light_palette("coral", n_colors=len(plot_df), reverse=True)
)

plt.title("Top 10 Teams with Most Blown Leads (HT Win → FT Loss)")
plt.xlabel("Team")
plt.ylabel("Blown Leads")

plt.xticks(rotation=45, ha="right")

for i, v in enumerate(plot_df["blown_leads"]):
    plt.text(i, v + 0.1, str(v), ha="center")

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1  DATASET OVERVIEW & MATCH OUTCOMES")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════
# Q1 – Match outcome distribution + home advantage magnitude


total = df.count()
hw = df.filter(col("FTResult") == "H").count()
dr = df.filter(col("FTResult") == "D").count()
aw = df.filter(col("FTResult") == "A").count()


home_pct = hw / total * 100
draw_pct = dr / total * 100
away_pct = aw / total * 100

print(f"\nQ1 – Match Outcome Distribution")
print(f"Home Win: {home_pct:.1f}%  |  Draw: {draw_pct:.1f}%  |  Away Win: {away_pct:.1f}%")


labels = ["Home Win", "Draw", "Away Win"]
values = [home_pct, draw_pct, away_pct]
colors = ['skyblue', 'lightgreen', 'salmon']

plt.figure(figsize=(8,6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops={'edgecolor':'black'})
plt.title("Match Outcome Distribution (2000–2025)")
plt.axis('equal')
plt.tight_layout()
plt.show()


home_avg = df.filter(col("FTResult") == "H") \
    .select(mean(col("FTHome") + col("FTAway"))).first()[0]

draw_avg = df.filter(col("FTResult") == "D") \
    .select(mean(col("FTHome") + col("FTAway"))).first()[0]

away_avg = df.filter(col("FTResult") == "A") \
    .select(mean(col("FTHome") + col("FTAway"))).first()[0]

print("\nAverage Goals Per Match by Outcome:")
print(f"Home Win: {home_avg:.2f}  |  Draw: {draw_avg:.2f}  |  Away Win: {away_avg:.2f}")


labels = ["Home Win", "Draw", "Away Win"]
values = [home_avg, draw_avg, away_avg]

plt.figure(figsize=(8,6))
bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])

plt.title("Average Goals per Match by Outcome")
plt.xlabel("Match Outcome")
plt.ylabel("Average Goals")

for i, v in enumerate(values):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()

# Q2 – Home vs Away goals per division (with home advantage margin)

div_goals = df.groupBy("Division").agg(
    mean("FTHome").alias("avg_home_goals"),
    mean("FTAway").alias("avg_away_goals"),
    mean(col("FTHome") - col("FTAway")).alias("home_advantage_margin"),
    count("*").alias("n_matches")
).orderBy(col("home_advantage_margin").desc())

print("\nQ2  Home vs Away Goals per Division")
div_goals.show(truncate=False)

div_goals_pd = div_goals.toPandas()


x = np.arange(len(div_goals_pd))
width = 0.35

plt.figure(figsize=(14,6))

bars1 = plt.bar(x - width/2, div_goals_pd["avg_home_goals"], width,
                label="Avg Home Goals", color='skyblue')

bars2 = plt.bar(x + width/2, div_goals_pd["avg_away_goals"], width,
                label="Avg Away Goals", color='salmon')

plt.xticks(x, div_goals_pd["Division"], rotation=45, ha='right')
plt.ylabel("Average Goals per Match")
plt.xlabel("Division")
plt.title("Home vs Away Average Goals per Division\n(+value = home advantage)")
plt.legend()


for i, row in div_goals_pd.iterrows():
    max_val = max(row["avg_home_goals"], row["avg_away_goals"])
    plt.text(i, max_val + 0.05, f"+{row['home_advantage_margin']:.2f}", ha='center')

plt.tight_layout()
plt.show()


# Q3 – Result distribution per division (stacked percentages)
div_results = df.groupBy("Division").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_pct"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("away_win_pct"),
    count("*").alias("n_matches")
).orderBy(col("home_win_pct").desc())

print("\nQ3 – Result Distribution per Division")
div_results.show(truncate=False)

div_results_pd = div_results.toPandas()


x = np.arange(len(div_results_pd))

plt.figure(figsize=(14,6))


bars1 = plt.bar(x, div_results_pd["home_win_pct"], label="Home Win %", color='skyblue')
bars2 = plt.bar(x, div_results_pd["draw_pct"],
                bottom=div_results_pd["home_win_pct"],
                label="Draw %", color='lightgreen')
bars3 = plt.bar(x, div_results_pd["away_win_pct"],
                bottom=div_results_pd["home_win_pct"] + div_results_pd["draw_pct"],
                label="Away Win %", color='salmon')


plt.axhline(home_pct, linestyle='', linewidth=1, label="Overall Home %")
plt.axhline(home_pct + draw_pct, linestyle='', linewidth=1, label="Home + Draw %")


plt.xticks(x, div_results_pd["Division"], rotation=45, ha='right')
plt.ylabel("Match Outcome (%)")
plt.xlabel("Division")
plt.title("Result Distribution per Division (Stacked %)")
plt.legend()

plt.tight_layout()
plt.show()

 
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2 – ELO & TEAM STRENGTH")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════

# Q4 – Elo difference vs match outcome probability (25pt bins)

elo_df = df.withColumn("EloDiff", col("HomeElo") - col("AwayElo"))

bins_expr = F.round(col("EloDiff") / 25) * 25

elo_bins = elo_df.withColumn("EloBin", bins_expr).groupBy("EloBin").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_pct"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("away_win_pct"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 50).orderBy("EloBin")

print("\nQ4 – Elo Difference vs Match Outcome Probability")
elo_bins.show(20, truncate=False)

elo_bins_pd = elo_bins.toPandas()


plt.figure(figsize=(12,6))

plt.plot(elo_bins_pd["EloBin"], elo_bins_pd["home_win_pct"],
         marker='o', label="Home Win %")

plt.plot(elo_bins_pd["EloBin"], elo_bins_pd["draw_pct"],
         marker='o', label="Draw %")

plt.plot(elo_bins_pd["EloBin"], elo_bins_pd["away_win_pct"],
         marker='o', label="Away Win %")

plt.axvline(0, linestyle='', linewidth=1)

plt.xlabel("Elo Difference (Home − Away) [25pt bins]")
plt.ylabel("Outcome Probability (%)")
plt.title("Elo Difference vs Match Outcome Probability")
plt.legend()

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Q5 – Elo vs goals scored & conceded (Home vs Away)

elo_goals = df.withColumn("HomeEloBin", F.round(col("HomeElo") / 100) * 100) \
              .withColumn("AwayEloBin", F.round(col("AwayElo") / 100) * 100)


home_elo_goals = elo_goals.groupBy("HomeEloBin").agg(
    mean("FTHome").alias("avg_goals_scored"),
    mean("FTAway").alias("avg_goals_conceded"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("HomeEloBin")


away_elo_goals = elo_goals.groupBy("AwayEloBin").agg(
    mean("FTAway").alias("avg_goals_scored"),
    mean("FTHome").alias("avg_goals_conceded"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("AwayEloBin")

print("\nQ5 – Elo vs Goals (Home & Away)")
home_elo_goals.show(10, truncate=False)
away_elo_goals.show(10, truncate=False)

home_pd = home_elo_goals.toPandas()
away_pd = away_elo_goals.toPandas()


plt.figure(figsize=(12,6))


plt.plot(home_pd["HomeEloBin"], home_pd["avg_goals_scored"],
         marker='o', label="Home – Scored")

plt.plot(home_pd["HomeEloBin"], home_pd["avg_goals_conceded"],
         marker='s', linestyle='', label="Home – Conceded")


plt.plot(away_pd["AwayEloBin"], away_pd["avg_goals_scored"],
         marker='o', label="Away – Scored")

plt.plot(away_pd["AwayEloBin"], away_pd["avg_goals_conceded"],
         marker='s', linestyle='', label="Away – Conceded")

plt.xlabel("Team Elo Rating [100pt bins]")
plt.ylabel("Average Goals per Match")
plt.title("Elo Rating vs Goals Scored & Conceded (Home vs Away)")
plt.legend(ncol=2)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Q6 – Elo vs Win Rate (Home vs Away)

home_elo_wr = elo_goals.groupBy("HomeEloBin").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("win_rate"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_rate"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("loss_rate"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("HomeEloBin")


away_elo_wr = elo_goals.groupBy("AwayEloBin").agg(
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("win_rate"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_rate"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("loss_rate"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("AwayEloBin")

print("\nQ6 – Elo vs Outcome Rates (Home & Away)")
home_elo_wr.show(10, truncate=False)
away_elo_wr.show(10, truncate=False)

home_pd = home_elo_wr.toPandas()
away_pd = away_elo_wr.toPandas()


fig, axes = plt.subplots(1, 2, figsize=(14,6))


axes[0].plot(home_pd["HomeEloBin"], home_pd["win_rate"], marker='o', label="Win %")
axes[0].plot(home_pd["HomeEloBin"], home_pd["draw_rate"], marker='o', linestyle='', label="Draw %")
axes[0].plot(home_pd["HomeEloBin"], home_pd["loss_rate"], marker='o', linestyle=':', label="Loss %")

axes[0].set_title("Home Team: Elo vs Outcome %")
axes[0].set_xlabel("Home Elo [100pt bins]")
axes[0].set_ylabel("Outcome (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)


axes[1].plot(away_pd["AwayEloBin"], away_pd["win_rate"], marker='o', label="Win %")
axes[1].plot(away_pd["AwayEloBin"], away_pd["draw_rate"], marker='o', linestyle='', label="Draw %")
axes[1].plot(away_pd["AwayEloBin"], away_pd["loss_rate"], marker='o', linestyle=':', label="Loss %")

axes[1].set_title("Away Team: Elo vs Outcome %")
axes[1].set_xlabel("Away Elo [100pt bins]")
axes[1].set_ylabel("Outcome (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Elo Rating vs Match Outcomes (Home vs Away)")
plt.tight_layout()
plt.show()

# Q7 – Elo difference distribution by outcome (box plot + correlation)

elo_band_df2 = elo_df.withColumn("AbsoluteEloDiff", F.abs(col("EloDiff"))) \
    .withColumn("StrongerTeamWon",
        when(((col("EloDiff") > 0) & (col("FTResult") == "H")) |
             ((col("EloDiff") < 0) & (col("FTResult") == "A")), 1)
        .when(col("FTResult") == "D", 0)
        .otherwise(1)
    )

plot_df = elo_band_df2.select("AbsoluteEloDiff", "StrongerTeamWon") \
    .sample(fraction=0.3, seed=42).toPandas()


binary_df = plot_df[plot_df["StrongerTeamWon"] != 0]
corr_pb, pvalue = stats.pointbiserialr(
    binary_df["StrongerTeamWon"],
    binary_df["AbsoluteEloDiff"]
)

print(f"\nQ7 – Elo Gap vs Match Outcome")
print(f"PointBiserial Correlation: {corr_pb:.4f}  (pvalue: {pvalue:.4f})")


stronger_won = plot_df[plot_df["StrongerTeamWon"] ==  1]["AbsoluteEloDiff"]
draw_elo     = plot_df[plot_df["StrongerTeamWon"] ==  0]["AbsoluteEloDiff"]
weaker_won   = plot_df[plot_df["StrongerTeamWon"] == 1]["AbsoluteEloDiff"]


plt.figure(figsize=(8,5))

bp = plt.boxplot(
    [stronger_won, draw_elo, weaker_won],
    labels=["Stronger Won", "Draw", "Weaker Won"],
    patch_artist=True,
    showfliers=False
)


colors = ["#2ecc71", "#f39c12", "#e74c3c"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

plt.title(
    f"Elo Gap Distribution by Match Outcome\n"
    f"Correlation = {corr_pb:.4f} (p = {pvalue:.4f})"
)

plt.xlabel("Match Outcome")
plt.ylabel("Absolute Elo Difference |Home − Away|")

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3 – FORM & MOMENTUM")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════

 # Q8 – Form3 (last 3 matches) vs goals & win rate (4panel)

form_df = df.withColumn("Form3HomeBin", F.round(col("Form3Home") * 10) / 10) \
            .withColumn("Form3AwayBin", F.round(col("Form3Away") * 10) / 10)


home_form = form_df.groupBy("Form3HomeBin").agg(
    mean("FTHome").alias("avg_home_goals"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("Form3HomeBin")


away_form = form_df.groupBy("Form3AwayBin").agg(
    mean("FTAway").alias("avg_away_goals"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("away_win_pct"),
    count("*").alias("n_matches")
).filter(col("n_matches") > 30).orderBy("Form3AwayBin")

print("\nQ8 – Form3 vs Goals & Win Rate")
home_form.show(10, truncate=False)
away_form.show(10, truncate=False)

home_pd = home_form.toPandas()
away_pd = away_form.toPandas()


fig, axes = plt.subplots(2, 2, figsize=(14,8))


axes[0,0].bar(home_pd["Form3HomeBin"], home_pd["avg_home_goals"], width=0.08)
axes[0,0].set_title("Home Form3 → Avg Goals (Home)")
axes[0,0].set_xlabel("Home Form3")
axes[0,0].set_ylabel("Avg Home Goals")


axes[0,1].bar(away_pd["Form3AwayBin"], away_pd["avg_away_goals"], width=0.08)
axes[0,1].set_title("Away Form3 → Avg Goals (Away)")
axes[0,1].set_xlabel("Away Form3")
axes[0,1].set_ylabel("Avg Away Goals")


axes[1,0].plot(home_pd["Form3HomeBin"], home_pd["home_win_pct"], marker='o')
axes[1,0].set_title("Home Form3 → Home Win %")
axes[1,0].set_xlabel("Home Form3")
axes[1,0].set_ylabel("Home Win %")
axes[1,0].grid(alpha=0.3)


axes[1,1].plot(away_pd["Form3AwayBin"], away_pd["away_win_pct"], marker='o')
axes[1,1].set_title("Away Form3 → Away Win %")
axes[1,1].set_xlabel("Away Form3")
axes[1,1].set_ylabel("Away Win %")
axes[1,1].grid(alpha=0.3)

plt.suptitle("Form (Last 3 Matches) vs Goals & Match Outcomes")
plt.tight_layout()
plt.show()

# Q9 – Form3 vs Form5: which is the better predictor?

def compute_form_signal(df, home_col, away_col):
    tmp = df.withColumn("FormAdv", col(home_col) - col(away_col)) \
            .withColumn("FormAdvBin", F.round(col("FormAdv") * 5) / 5)

    tmp = tmp.groupBy("FormAdvBin").agg(
        (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
        count("*").alias("n_matches")
    ).filter(col("n_matches") > 40).orderBy("FormAdvBin")

    return tmp


f3 = compute_form_signal(df, "Form3Home", "Form3Away")
f5 = compute_form_signal(df, "Form5Home", "Form5Away")


f3_pd = f3.toPandas()
f5_pd = f5.toPandas()


corr3, _ = stats.pearsonr(f3_pd["FormAdvBin"], f3_pd["home_win_pct"])
corr5, _ = stats.pearsonr(f5_pd["FormAdvBin"], f5_pd["home_win_pct"])

print("\nQ9 – Form3 vs Form5 Predictive Power")
print(f"Form3 Pearson r: {corr3:.4f}")
print(f"Form5 Pearson r: {corr5:.4f}")


plt.figure(figsize=(12,5))

plt.plot(f3_pd["FormAdvBin"], f3_pd["home_win_pct"],
         marker='o', label=f"Form3 (r={corr3:.3f})")

plt.plot(f5_pd["FormAdvBin"], f5_pd["home_win_pct"],
         marker='s', label=f"Form5 (r={corr5:.3f})")

plt.axvline(0, linestyle='', linewidth=1)

plt.xlabel("Form Advantage (Home − Away)")
plt.ylabel("Home Win %")
plt.title("Form3 vs Form5: Predictive Strength Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Q10 – Does having both Form AND Elo advantage guarantee a win?

combined_df = df.withColumn("EloDiff", col("HomeElo") - col("AwayElo")) \
    .withColumn("FormDiff", col("Form3Home") - col("Form3Away")) \
    .withColumn(
        "AdvantageType",
        when((col("EloDiff") > 0) & (col("FormDiff") > 0), "Both")
        .when((col("EloDiff") > 0) & (col("FormDiff") <= 0), "Elo Only")
        .when((col("EloDiff") <= 0) & (col("FormDiff") > 0), "Form Only")
        .otherwise("None")
    )

advantage_stats = combined_df.groupBy("AdvantageType").agg(
    count("*").alias("total"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_pct"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("away_win_pct")
).toPandas()

order = ["Both", "Elo Only", "Form Only", "None"]
advantage_stats = advantage_stats.set_index("AdvantageType").reindex(order).reset_index()

print("Combined Elo + Form Advantage vs Outcomes")
print(advantage_stats.to_string(index=False))


x = np.arange(len(advantage_stats))
w = 0.25

plt.figure(figsize=(12,6))

bars1 = plt.bar(x - w, advantage_stats["home_win_pct"], w, label="Home Win %", color="skyblue")
bars2 = plt.bar(x,     advantage_stats["draw_pct"],     w, label="Draw %",     color="orange")
bars3 = plt.bar(x + w, advantage_stats["away_win_pct"], w, label="Away Win %", color="seagreen")

plt.xticks(x, advantage_stats["AdvantageType"])
plt.ylabel("Percentage (%)")
plt.title("Match Outcomes by Combined Elo and Form Advantage")

plt.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 6 – BETTING ODDS & MARKET EFFICIENCY (COMPRESSED)")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════


# ── Q11 – Bookmaker calibration: implied vs actual home win probability ───────
odds_df = df.withColumn("ImpliedHome", lit(1.0) / col("OddHome")) \
            .withColumn("ImpliedHomeBin", F.round(col("ImpliedHome") * 10) / 10)

calib = odds_df.groupBy("ImpliedHomeBin").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("actual_pct"),
    count("*").alias("n"),
    mean("ImpliedHome").alias("implied_mean")
).filter(col("n") > 50).orderBy("ImpliedHomeBin").toPandas()

calib["implied_pct"] = calib["implied_mean"] * 100

plt.figure(figsize=(10,6))
plt.plot([0,100],[0,100], linestyle="", alpha=0.5, label="Perfect calibration")
plt.scatter(
    calib["implied_pct"],
    calib["actual_pct"],
    s=calib["n"]/calib["n"].max()*300,
    alpha=0.7
)

plt.title("Bookmaker Calibration: Implied vs Actual Home Win %")
plt.xlabel("Implied Probability (%)")
plt.ylabel("Actual Home Win %")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ── Q12 – Market accuracy by favourite type ───────────────────────────────────
seg_df = df.withColumn(
    "FavType",
    when(col("OddHome") < col("OddAway"), "Home Favourite")
    .when(col("OddAway") < col("OddHome"), "Away Favourite")
    .otherwise("Even Odds")
)

seg_stats = seg_df.groupBy("FavType").agg(
    (count(when(
        ((col("FavType")=="Home Favourite") & (col("FTResult")=="H")) |
        ((col("FavType")=="Away Favourite") & (col("FTResult")=="A")),
        1
    )) / count("*") * 100).alias("accuracy"),
    count("*").alias("n")
).toPandas()

print("\nMarket Favourite Accuracy")
print(seg_stats.to_string(index=False))

plt.figure(figsize=(8,5))
plt.bar(seg_stats["FavType"], seg_stats["accuracy"])
plt.title("Bookmaker Favourite Accuracy")
plt.ylabel("Correct Prediction %")
plt.grid(axis="y", alpha=0.3)
plt.show()


# ── Q13 – Underdog win rate (value betting signal) ───────────────────────────
underdog_home = df.filter(col("OddHome") > col("OddAway"))
underdog_home_wins = underdog_home.filter(col("FTResult") == "H").count()
underdog_home_total = underdog_home.count()

print(f"\nUnderdog Home Win Rate: {underdog_home_wins / underdog_home_total * 100:.1f}% "
      f"({underdog_home_wins}/{underdog_home_total})")


# ── Q14 – Over 2.5 goals vs Elo strength ──────────────────────────────────────
over_df = df.withColumn("Over25", when(col("FTHome")+col("FTAway") > 2, 1).otherwise(0)) \
            .withColumn("AvgElo", (col("HomeElo")+col("AwayElo"))/2) \
            .withColumn("EloBin", F.round(col("AvgElo")/100)*100)

over_elo = over_df.groupBy("EloBin").agg(
    (count(when(col("Over25")==1, 1)) / count("*") * 100).alias("over_pct"),
    count("*").alias("n")
).filter(col("n") > 30).orderBy("EloBin").toPandas()

plt.figure(figsize=(12,5))
plt.plot(over_elo["EloBin"], over_elo["over_pct"], marker="o")
plt.fill_between(over_elo["EloBin"], over_elo["over_pct"], alpha=0.2)

plt.axhline(50, linestyle="", alpha=0.5)
plt.title("Over 2.5 Goals vs Average Elo Strength")
plt.xlabel("Average Elo (binned)")
plt.ylabel("Over 2.5 Rate %")
plt.grid(alpha=0.3)
plt.show()


# ── Q15 – Elo vs bookmaker odds consistency ──────────────────────────────────
scatter = df.select(
    (col("HomeElo") - col("AwayElo")).alias("EloDiff"),
    "OddHome", "FTResult"
).dropna().sample(0.05, seed=42).toPandas()

plt.figure(figsize=(10,6))
plt.scatter(scatter["EloDiff"], scatter["OddHome"], alpha=0.3)

plt.title("Elo Difference vs Home Odds")
plt.xlabel("Elo Difference (Home - Away)")
plt.ylabel("Home Odds")
plt.grid(alpha=0.3)
plt.show()

# ───────────────────────────────────────────────────────

outcomes = ["H", "D", "A"]
out_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
out_colors = {"H": "skyblue", "D": "lightgreen", "A": "salmon"}

def avg_by_outcome(spark_df, numeric_col):
    rows = spark_df.groupBy("FTResult").agg(
        mean(col(numeric_col)).alias("avg_val"),
        count("*").alias("n")
    ).toPandas()
    rows = rows.set_index("FTResult").reindex(outcomes)
    return rows

def box_data(spark_df, numeric_col, frac=0.3, seed=42):
    sdf = spark_df.select("FTResult", col(numeric_col).cast(DoubleType()).alias("val")).dropna()
    pdf = sdf.sample(fraction=min(frac, 1.0), seed=seed).toPandas()
    groups = [pdf[pdf["FTResult"] == o]["val"].dropna().values for o in outcomes]
    return groups

def outcome_pct_by_bin(spark_df, bin_col):
    binned = spark_df.groupBy(bin_col).agg(
        (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("H_pct"),
        (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("D_pct"),
        (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("A_pct"),
        count("*").alias("n")
    ).filter(col("n") > 30).orderBy(bin_col).toPandas()
    return binned

print("\nF1 – Division vs FTResult")
div_res = outcome_pct_by_bin(df, "Division")
 
x = np.arange(len(div_res))
plt.figure(figsize=(16, 6))
plt.bar(x, div_res["H_pct"], label="Home Win %", color="skyblue")
plt.bar(x, div_res["D_pct"], bottom=div_res["H_pct"], label="Draw %", color="lightgreen")
plt.bar(x, div_res["A_pct"], bottom=div_res["H_pct"] + div_res["D_pct"], label="Away Win %", color="salmon")
plt.xticks(x, div_res["Division"], rotation=45, ha="right")
plt.ylabel("Match Outcome (%)")
plt.xlabel("Division")
plt.title("F1 – Division vs FTResult (Stacked %)", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()

print("\nF2 – MatchDate (Year) vs FTResult")
year_df = df.withColumn("Year", F.year(F.to_date("MatchDate")))
year_res = outcome_pct_by_bin(year_df, "Year")
 
plt.figure(figsize=(14, 6))
plt.plot(year_res["Year"], year_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(year_res["Year"], year_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(year_res["Year"], year_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Year")
plt.ylabel("Outcome (%)")
plt.title("F2 – Match Year vs FTResult", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("\nF5 – HomeElo vs FTResult (Probability per 100 Elo bin)")
bin_size = 100
df_plot = df.withColumn(
    "HomeEloBin",
    (floor(col("HomeElo") / bin_size) * bin_size)
)

binned = df_plot.groupBy("HomeEloBin").agg(
    count("*").alias("n"),

    (count(when(col("FTResult") == "H", True)) / count("*")).alias("P_H"),
    (count(when(col("FTResult") == "D", True)) / count("*")).alias("P_D"),
    (count(when(col("FTResult") == "A", True)) / count("*")).alias("P_A")
).filter(col("n") > 30).orderBy("HomeEloBin").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(
    binned["HomeEloBin"], binned["P_H"],
    label="P(Home Win)",
    color=out_colors["H"],
    marker="o"
)
plt.plot(
    binned["HomeEloBin"], binned["P_D"],
    label="P(Draw)",
    color=out_colors["D"],
    marker="o"
)
plt.plot(
    binned["HomeEloBin"], binned["P_A"],
    label="P(Away Win)",
    color=out_colors["A"],
    marker="o"
)
plt.title("F5 – HomeElo vs FTResult ", fontweight="bold")
plt.xlabel("Home Elo (binned by 100)")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# de el probability bta3et el H gaya men esmet 3add el H fel interval da 3ala 3add 
# el matches fel interval da 


print("\nF5 – AwayElo vs FTResult (Probability per 100 Elo bin)")
bin_size = 100
df_plot = df.withColumn(
    "AwayEloBin",
    (floor(col("AwayElo") / bin_size) * bin_size)
)

binned = df_plot.groupBy("AwayEloBin").agg(
    count("*").alias("n"),

    (count(when(col("FTResult") == "H", True)) / count("*")).alias("P_H"),
    (count(when(col("FTResult") == "D", True)) / count("*")).alias("P_D"),
    (count(when(col("FTResult") == "A", True)) / count("*")).alias("P_A")
).filter(col("n") > 30).orderBy("AwayEloBin").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(
    binned["AwayEloBin"], binned["P_H"],
    label="P(Home Win)",
    color=out_colors["H"],
    marker="o"
)
plt.plot(
    binned["AwayEloBin"], binned["P_D"],
    label="P(Draw)",
    color=out_colors["D"],
    marker="o"
)
plt.plot(
    binned["AwayEloBin"], binned["P_A"],
    label="P(Away Win)",
    color=out_colors["A"],
    marker="o"
)
plt.title("F5 – AwayElo vs FTResult ", fontweight="bold")
plt.xlabel("Away Elo (binned by 100)")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



print("\nF5 – HomeAwayDiffElo vs FTResult (Probability per 100 Elo bin)")
bin_size = 100
df_plot = df.withColumn(
    "HomeAwayDiffElo",
    col("HomeElo") - col("AwayElo")
).withColumn(
    "HomeAwayDiffEloBin",
    floor(col("HomeAwayDiffElo") / bin_size) * bin_size
)
binned = df_plot.groupBy("HomeAwayDiffEloBin").agg(
    count("*").alias("n"),

    (count(when(col("FTResult") == "H", True)) / count("*")).alias("P_H"),
    (count(when(col("FTResult") == "D", True)) / count("*")).alias("P_D"),
    (count(when(col("FTResult") == "A", True)) / count("*")).alias("P_A")
).filter(col("n") > 30).orderBy("HomeAwayDiffEloBin").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(
    binned["HomeAwayDiffEloBin"], binned["P_H"],
    label="P(Home Win)",
    color=out_colors["H"],
    marker="o"
)
plt.plot(
    binned["HomeAwayDiffEloBin"], binned["P_D"],
    label="P(Draw)",
    color=out_colors["D"],
    marker="o"
)
plt.plot(
    binned["HomeAwayDiffEloBin"], binned["P_A"],
    label="P(Away Win)",
    color=out_colors["A"],
    marker="o"
)
plt.title("F5 – HomeAwayDiffElo vs FTResult ", fontweight="bold")
plt.xlabel("Home-Away Elo Difference (binned by 100)")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nF7 – Form3Home vs FTResult")
f3h = df.withColumn("Form3HomeBin", F.round(col("Form3Home") * 10) / 10)
f3h_res = outcome_pct_by_bin(f3h, "Form3HomeBin")
 
plt.figure(figsize=(10, 6))
plt.plot(f3h_res["Form3HomeBin"], f3h_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(f3h_res["Form3HomeBin"], f3h_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(f3h_res["Form3HomeBin"], f3h_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Form3Home (binned)")
plt.ylabel("Outcome (%)")
plt.title("F7 – Form3Home vs FTResult\n[Feature 7 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nF8 – Form5Home vs FTResult")
f5h = df.withColumn("Form5HomeBin", F.round(col("Form5Home") * 10) / 10)
f5h_res = outcome_pct_by_bin(f5h, "Form5HomeBin")
 
plt.figure(figsize=(10, 6))
plt.plot(f5h_res["Form5HomeBin"], f5h_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(f5h_res["Form5HomeBin"], f5h_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(f5h_res["Form5HomeBin"], f5h_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Form5Home (binned)")
plt.ylabel("Outcome (%)")
plt.title("F8 – Form5Home vs FTResult\n[Feature 8 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nF9 – Form3Away vs FTResult")
f3a = df.withColumn("Form3AwayBin", F.round(col("Form3Away") * 10) / 10)
f3a_res = outcome_pct_by_bin(f3a, "Form3AwayBin")
 
plt.figure(figsize=(10, 6))
plt.plot(f3a_res["Form3AwayBin"], f3a_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(f3a_res["Form3AwayBin"], f3a_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(f3a_res["Form3AwayBin"], f3a_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Form3Away (binned)")
plt.ylabel("Outcome (%)")
plt.title("F9 – Form3Away vs FTResult\n[Feature 9 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 

print("\nF10 – Form5Away vs FTResult")
f5a = df.withColumn("Form5AwayBin", F.round(col("Form5Away") * 10) / 10)
f5a_res = outcome_pct_by_bin(f5a, "Form5AwayBin")
 
plt.figure(figsize=(10, 6))
plt.plot(f5a_res["Form5AwayBin"], f5a_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(f5a_res["Form5AwayBin"], f5a_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(f5a_res["Form5AwayBin"], f5a_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Form5Away (binned)")
plt.ylabel("Outcome (%)")
plt.title("F10 – Form5Away vs FTResult\n[Feature 10 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 
print("\nF11 – FTHome (goals) vs FTResult")
rows = avg_by_outcome(df, "FTHome")
plt.figure(figsize=(8, 6))
bars = plt.bar([out_labels[o] for o in outcomes],
               [rows.loc[o, "avg_val"] for o in outcomes],
               color=[out_colors[o] for o in outcomes])
plt.title("F11 – FTHome Goals vs FTResult\n[Feature 11 of 28]", fontweight="bold")
plt.xlabel("Match Outcome")
plt.ylabel("Avg Full-Time Home Goals")
for i, o in enumerate(outcomes):
    plt.text(i, rows.loc[o, "avg_val"] + 0.01, f"{rows.loc[o, 'avg_val']:.2f}", ha="center")
plt.tight_layout()
plt.show()

print("\nF12 – FTAway (goals) vs FTResult")
rows = avg_by_outcome(df, "FTAway")
plt.figure(figsize=(8, 6))
bars = plt.bar([out_labels[o] for o in outcomes],
               [rows.loc[o, "avg_val"] for o in outcomes],
               color=[out_colors[o] for o in outcomes])
plt.title("F12 – FTAway Goals vs FTResult\n[Feature 12 of 28]", fontweight="bold")
plt.xlabel("Match Outcome")
plt.ylabel("Avg Full-Time Away Goals")
for i, o in enumerate(outcomes):
    plt.text(i, rows.loc[o, "avg_val"] + 0.01, f"{rows.loc[o, 'avg_val']:.2f}", ha="center")
plt.tight_layout()
plt.show()

print("\nF20 – OddHome vs FTResult")
odd_h = df.withColumn("OddHomeBin", F.round(col("OddHome") * 2) / 2) \
          .filter((col("OddHomeBin") >= 1.0) & (col("OddHomeBin") <= 10.0))
odd_h_res = outcome_pct_by_bin(odd_h, "OddHomeBin")
 
plt.figure(figsize=(12, 6))
plt.plot(odd_h_res["OddHomeBin"], odd_h_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(odd_h_res["OddHomeBin"], odd_h_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(odd_h_res["OddHomeBin"], odd_h_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Home Win Odds (binned 0.5 step)")
plt.ylabel("Outcome (%)")
plt.title("F20 – OddHome vs FTResult\n[Feature 20 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 

print("\nF21 – OddDraw vs FTResult")
odd_d = df.withColumn("OddDrawBin", F.round(col("OddDraw") * 2) / 2) \
          .filter((col("OddDrawBin") >= 2.0) & (col("OddDrawBin") <= 8.0))
odd_d_res = outcome_pct_by_bin(odd_d, "OddDrawBin")
 
plt.figure(figsize=(12, 6))
plt.plot(odd_d_res["OddDrawBin"], odd_d_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(odd_d_res["OddDrawBin"], odd_d_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(odd_d_res["OddDrawBin"], odd_d_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Draw Odds (binned 0.5 step)")
plt.ylabel("Outcome (%)")
plt.title("F21 – OddDraw vs FTResult\n[Feature 21 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nF22 – OddAway vs FTResult")
odd_a = df.withColumn("OddAwayBin", F.round(col("OddAway") * 2) / 2) \
          .filter((col("OddAwayBin") >= 1.0) & (col("OddAwayBin") <= 10.0))
odd_a_res = outcome_pct_by_bin(odd_a, "OddAwayBin")
 
plt.figure(figsize=(12, 6))
plt.plot(odd_a_res["OddAwayBin"], odd_a_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(odd_a_res["OddAwayBin"], odd_a_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(odd_a_res["OddAwayBin"], odd_a_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.xlabel("Away Win Odds (binned 0.5 step)")
plt.ylabel("Outcome (%)")
plt.title("F22 – OddAway vs FTResult\n[Feature 22 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 

print("\nF23 – MaxHome vs FTResult")
groups = box_data(df, "MaxHome")
plt.figure(figsize=(8, 6))
bp = plt.boxplot(groups, labels=[out_labels[o] for o in outcomes],
                 patch_artist=True, showfliers=False)
for patch, o in zip(bp["boxes"], outcomes):
    patch.set_facecolor(out_colors[o])
plt.title("F23 – MaxHome Odds vs FTResult\n[Feature 23 of 28]", fontweight="bold")
plt.xlabel("Match Outcome")
plt.ylabel("Max Home Odds")
plt.tight_layout()
plt.show()

print("\nF24 – MaxDraw vs FTResult")
groups = box_data(df, "MaxDraw")
plt.figure(figsize=(8, 6))
bp = plt.boxplot(groups, labels=[out_labels[o] for o in outcomes],
                 patch_artist=True, showfliers=False)
for patch, o in zip(bp["boxes"], outcomes):
    patch.set_facecolor(out_colors[o])
plt.title("F24 – MaxDraw Odds vs FTResult\n[Feature 24 of 28]", fontweight="bold")
plt.xlabel("Match Outcome")
plt.ylabel("Max Draw Odds")
plt.tight_layout()
plt.show()

print("\nF25 – MaxAway vs FTResult")
groups = box_data(df, "MaxAway")
plt.figure(figsize=(8, 6))
bp = plt.boxplot(groups, labels=[out_labels[o] for o in outcomes],
                 patch_artist=True, showfliers=False)
for patch, o in zip(bp["boxes"], outcomes):
    patch.set_facecolor(out_colors[o])
plt.title("F25 – MaxAway Odds vs FTResult\n[Feature 25 of 28]", fontweight="bold")
plt.xlabel("Match Outcome")
plt.ylabel("Max Away Odds")
plt.tight_layout()
plt.show()


print("\nF26 – Over25 flag vs FTResult")
# Recompute Over25 from actual goals if column is odds-based
over_flag = df.withColumn(
    "Over25Flag",
    when(col("FTHome") + col("FTAway") > 2, lit("Over 2.5")).otherwise(lit("Under 2.5"))
)
over_res = over_flag.groupBy("Over25Flag").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("H_pct"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("D_pct"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("A_pct"),
    count("*").alias("n")
).orderBy("Over25Flag").toPandas()
 
x = np.arange(len(over_res))
plt.figure(figsize=(8, 6))
plt.bar(x, over_res["H_pct"], label="Home Win %", color="skyblue")
plt.bar(x, over_res["D_pct"], bottom=over_res["H_pct"], label="Draw %", color="lightgreen")
plt.bar(x, over_res["A_pct"], bottom=over_res["H_pct"] + over_res["D_pct"], label="Away Win %", color="salmon")
plt.xticks(x, over_res["Over25Flag"])
plt.ylabel("Outcome (%)")
plt.xlabel("Total Goals Group")
plt.title("F26 – Over/Under 2.5 Goals vs FTResult\n[Feature 26 of 28]", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()
 

print("\nF27 – HandiSize vs FTResult")
handi = df.withColumn("HandiBin", F.round(col("HandiSize") * 4) / 4) \
          .filter(col("HandiBin").between(-3.0, 3.0))
handi_res = outcome_pct_by_bin(handi, "HandiBin")
 
plt.figure(figsize=(12, 6))
plt.plot(handi_res["HandiBin"], handi_res["H_pct"], marker="o", label="Home Win %", color="skyblue")
plt.plot(handi_res["HandiBin"], handi_res["D_pct"], marker="o", label="Draw %", color="lightgreen")
plt.plot(handi_res["HandiBin"], handi_res["A_pct"], marker="o", label="Away Win %", color="salmon")
plt.axvline(0, linestyle="--", linewidth=1, alpha=0.5)
plt.xlabel("Handicap Size (binned 0.25 step)")
plt.ylabel("Outcome (%)")
plt.title("F27 – HandiSize vs FTResult\n[Feature 27 of 28]", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 

print("\nF28 – HandiHome & HandiAway vs FTResult")
hh_groups = box_data(df, "HandiHome")
ha_groups = box_data(df, "HandiAway")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
bp1 = axes[0].boxplot(hh_groups, labels=[out_labels[o] for o in outcomes],
                      patch_artist=True, showfliers=False)
for patch, o in zip(bp1["boxes"], outcomes):
    patch.set_facecolor(out_colors[o])
axes[0].set_title("HandiHome vs FTResult")
axes[0].set_xlabel("Match Outcome")
axes[0].set_ylabel("Home Handicap Odds")
 
bp2 = axes[1].boxplot(ha_groups, labels=[out_labels[o] for o in outcomes],
                      patch_artist=True, showfliers=False)
for patch, o in zip(bp2["boxes"], outcomes):
    patch.set_facecolor(out_colors[o])
axes[1].set_title("HandiAway vs FTResult")
axes[1].set_xlabel("Match Outcome")
axes[1].set_ylabel("Away Handicap Odds")
 
plt.suptitle("F28 – HandiHome & HandiAway vs FTResult\n[Feature 28 of 28]", fontweight="bold")
plt.tight_layout()
plt.show()
df_raw = df.withColumn("MatchYear",  F.year("MatchDate")) \
           .withColumn("MatchMonth", F.month("MatchDate")) \
           .withColumn("DateInt",    F.unix_timestamp("MatchDate"))

df_home_perspective = df_raw.select(
    col("HomeTeam").alias("Team"),
    col("DateInt"),
    col("FTHome").alias("GoalsFor"),
    col("FTAway").alias("GoalsAgainst")
)

df_away_perspective = df_raw.select(
    col("AwayTeam").alias("Team"),
    col("DateInt"),
    col("FTAway").alias("GoalsFor"),
    col("FTHome").alias("GoalsAgainst")
)

df_team = df_home_perspective.unionByName(df_away_perspective)

w_home = Window.partitionBy("HomeTeam").orderBy("DateInt").rowsBetween(-5, -1)
w_away = Window.partitionBy("AwayTeam").orderBy("DateInt").rowsBetween(-5, -1)

df_features = df_raw \
    .withColumn("home_goals_last3_sum",    F.coalesce(F.sum("FTHome").over(w_home),     F.lit(0))) \
    .withColumn("home_conceded_last3_sum", F.coalesce(F.sum("FTAway").over(w_home),     F.lit(0))) \
    .withColumn("home_reds_last3_sum",     F.coalesce(F.sum("HomeRed").over(w_home),    F.lit(0))) \
    .withColumn("home_yellows_last3_sum",  F.coalesce(F.sum("HomeYellow").over(w_home), F.lit(0))) \
    .withColumn("away_goals_last3_sum",    F.coalesce(F.sum("FTAway").over(w_away),     F.lit(0))) \
    .withColumn("away_conceded_last3_sum", F.coalesce(F.sum("FTHome").over(w_away),     F.lit(0))) \
    .withColumn("away_reds_last3_sum",     F.coalesce(F.sum("AwayRed").over(w_away),    F.lit(0))) \
    .withColumn("away_yellows_last3_sum",  F.coalesce(F.sum("AwayYellow").over(w_away), F.lit(0)))\
    .withColumn("EloDiff",     col("HomeElo")   - col("AwayElo")) \
    .withColumn("Form3Diff",   col("Form3Home") - col("Form3Away")) \
    .withColumn("Form5Diff",   col("Form5Home") - col("Form5Away")) \
    .withColumn("OddDiff",     col("OddHome")   - col("OddAway")) \
    .withColumn("HandiOddDiff",col("HandiHome") - col("HandiAway")) \
    .withColumn("goals_last3_diff", col("home_goals_last3_sum") - col("away_goals_last3_sum")) \
    .withColumn("cards_last3_sum",  col("home_yellows_last3_sum") + col("away_yellows_last3_sum") \
                                  + col("home_reds_last3_sum")    + col("away_reds_last3_sum"))

features_info = [
    ("home_goals_last3_sum",          "Home Goals Scored (Last 3 Home Matches)",      "skyblue"),
    ("home_conceded_last3_sum",       "Home Goals Conceded (Last 3 Home Matches)",    "salmon"),
    ("home_reds_last3_sum",           "Home Red Cards (Last 3 Home Matches)",         "firebrick"),
    ("home_yellows_last3_sum",        "Home Yellow Cards (Last 3 Home Matches)",      "gold"),
    ("away_goals_last3_sum",          "Away Goals Scored (Last 3 Away Matches)",      "skyblue"),
    ("away_conceded_last3_sum",       "Away Goals Conceded (Last 3 Away Matches)",    "salmon"),
    ("away_reds_last3_sum",           "Away Red Cards (Last 3 Away Matches)",         "firebrick"),
    ("away_yellows_last3_sum",        "Away Yellow Cards (Last 3 Away Matches)",      "gold"),
    ("HomeElo",                        "Home Elo Rating",                             "skyblue"),
    ("AwayElo",                        "Away Elo Rating",                             "salmon"),
    ("Form3Home",                      "Home Form (Last 3 Matches)",                 "skyblue"),
    ("Form5Home",                      "Home Form (Last 5 Matches)",                 "skyblue"),
    ("Form3Away",                      "Away Form (Last 3 Matches)",                 "salmon"),
    ("Form5Away",                      "Away Form (Last 5 Matches)",                 "salmon"),
    ("OddHome",                        "Home Win Odds",                             "skyblue"),
    ("OddDraw",                        "Draw Odds",                                 "lightgreen"),
    ("OddAway",                        "Away Win Odds",                             "salmon"),
    ("MaxHome",                        "Max Home Win Odds",                         "skyblue"),
    ("MaxDraw",                        "Max Draw Odds",                             "lightgreen"),
    ("MaxAway",                        "Max Away Win Odds",                         "salmon"),
    ("HandiSize",                      "Handicap Size",                             "steelblue"),
    ("HandiHome",                      "Home Handicap Odds",                       "skyblue"),
    ("HandiAway",                      "Away Handicap Odds",                       "salmon"),
    ("EloDiff",                        "Home-Away Elo Difference",                 "steelblue"),
    ("Form3Diff",                      "Home-Away Form (Last 3) Difference",       "steelblue"),
    ("Form5Diff",                      "Home-Away Form (Last 5) Difference",       "steelblue"),
    ("OddDiff",                        "Home-Away Odds Difference",               "steelblue"),
    ("HandiOddDiff",                   "Home-Away Handicap Odds Difference",      "steelblue"),
    ("goals_last3_diff",               "Home-Away Goals Scored (Last 3) Diff.",   "steelblue"),
    ("cards_last3_sum",                "Total Cards (Yellow+Red) Last 3 Matches", "firebrick"),
]

out_colors = {"H": "skyblue", "D": "lightgreen", "A": "salmon"}

all_feats = [f for f, _, _ in features_info]

df_corr = df_features.select(
    *[col(f).cast("double") for f in all_feats],
    (col("FTResult") == "H").cast("double").alias("is_H"),
    (col("FTResult") == "D").cast("double").alias("is_D"),
    (col("FTResult") == "A").cast("double").alias("is_A"),
).dropna().toPandas()

correlations = {
    f: {
        "H": df_corr[f].corr(df_corr["is_H"], method="spearman"),
        "D": df_corr[f].corr(df_corr["is_D"], method="spearman"),
        "A": df_corr[f].corr(df_corr["is_A"], method="spearman"),
    }
    for f in all_feats
}

for i, (feat, xlabel, _) in enumerate(features_info, start=1):
    feat_num = 29 + i
    binned = (
        df_features
        .withColumn(f"{feat}_bin", F.round(col(feat) * 50) / 50 )
        .groupBy(f"{feat}_bin")
        .agg(
            count("*").alias("n"),
            (count(when(col("FTResult") == "H", True)) / count("*")).alias("P_H"),
            (count(when(col("FTResult") == "D", True)) / count("*")).alias("P_D"),
            (count(when(col("FTResult") == "A", True)) / count("*")).alias("P_A"),
        )
        .orderBy(f"{feat}_bin")
        .filter(col("n") > 30)
        .toPandas()
    )

    rH = correlations[feat]["H"]
    rD = correlations[feat]["D"]
    rA = correlations[feat]["A"]
    print(f"\nF{feat_num} – {feat} vs FTResult\nCorrelation: r(H)={rH:.3f}, r(D)={rD:.3f}, r(A)={rA:.3f}")
    plt.figure(figsize=(10, 6))
    plt.plot(binned[f"{feat}_bin"], binned["P_H"], marker="o", label="P(Home Win)", color=out_colors["H"])
    plt.plot(binned[f"{feat}_bin"], binned["P_D"], marker="o", label="P(Draw)",     color=out_colors["D"])
    plt.plot(binned[f"{feat}_bin"], binned["P_A"], marker="o", label="P(Away Win)", color=out_colors["A"])

    plt.title(
        f"F{feat_num} – {feat} vs FTResult\n"
        f"r(H)={rH:.3f}, r(D)={rD:.3f}, r(A)={rA:.3f}",
        fontweight="bold"
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Mutual Information + Random Forest Feature Importance (ANOVA F-score also included)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

le = LabelEncoder()
df_ml = df_corr.copy()
df_ml["FTResult"] = df_features.select("FTResult").dropna().toPandas()["FTResult"]
df_ml = df_ml.dropna(subset=["FTResult"])
df_ml["FTResult_enc"] = le.fit_transform(df_ml["FTResult"])

X = df_ml[all_feats]
y = df_ml["FTResult_enc"]

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

mi   = mutual_info_classif(X, y, random_state=42)
F_scores, p_vals = f_classif(X, y)

results_df = pd.DataFrame({
    "Feature":              all_feats,
    "RF_Importance":        rf.feature_importances_,
    "Mutual_Info":          mi,
    "ANOVA_F":              F_scores,
    "ANOVA_p":              p_vals,
}).sort_values("RF_Importance", ascending=False)

print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, col_name, title in zip(
    axes,
    ["RF_Importance", "Mutual_Info", "ANOVA_F"],
    ["Random Forest Importance", "Mutual Information", "ANOVA F-Score"]
):
    sub = results_df.sort_values(col_name)
    ax.barh(sub["Feature"], sub[col_name], color="steelblue")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(title)
plt.suptitle("Feature → FTResult Importance (3 Methods)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()



numeric_features = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "FTHome", "FTAway", "HTHome", "HTAway",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway"
]

pdf = df.select(numeric_features).sample(fraction=1.0, seed=42).toPandas()

n_cols = 4
n_rows = (len(numeric_features) + n_cols - 1) // n_cols


fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, n_rows * 4), constrained_layout=True)
axes = axes.flatten()

for i, feat in enumerate(numeric_features):
    ax = axes[i]
    data = pdf[feat]

    ax.hist(data, bins=40, color="steelblue", edgecolor="white", alpha=0.7, density=True)

    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 300)
    ax.plot(x_range, kde(x_range), color="navy", linewidth=2)

    ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean={data.mean():.2f}")
    ax.axvline(data.median(), color="orange", linestyle=":", linewidth=1.2,
               label=f"Median={data.median():.2f}")

    ax.set_title(feat, fontweight="bold", fontsize=11)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


fig.suptitle("Feature Distributions — Numeric Columns",
             fontsize=18, fontweight="bold")


plt.subplots_adjust(top=0.95)

plt.show()