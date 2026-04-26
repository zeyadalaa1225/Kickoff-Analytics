from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, lit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
 
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

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 0 – DATASET INTRODUCTION & SURFACELEVEL OVERVIEW")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════
# Q1 – Matches per Division

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




# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION – CORRELATION MATRIX, PAIR PLOT & FINAL EDA DASHBOARD")
print("="*70)
# ══════════════════════════════════════════════════════════════════════════════


# ─── PLOT 1 – Correlation Heatmap ────────────────────────────────────────────

numeric_cols = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "FTHome", "FTAway",
    "HTHome", "HTAway",
    "HomeYellow", "AwayYellow",
    "HomeRed", "AwayRed",
    "OddHome", "OddDraw", "OddAway"
]

corr_sample = df.select([col(c).cast(DoubleType()) for c in numeric_cols]) \
                .dropna() \
                .sample(fraction=0.3, seed=42) \
                .toPandas()

corr_matrix = corr_sample.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(16, 13))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    annot_kws={"size": 8},
    ax=ax
)

ax.set_title(
    "Correlation Matrix – Key Numerical Features (2000–2025)\n",
    fontsize=13,
    fontweight="bold"
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.show()


# ─── PLOT 2 – Pair Plot ───────────────────────────────────────────────────────

pair_cols = ["HomeElo", "AwayElo", "Form3Home", "Form3Away", "FTHome", "FTAway", "OddHome"]

pair_sample = df.select(
    [col(c).cast(DoubleType()) for c in pair_cols] + ["FTResult"]
).dropna() \
 .sample(fraction=0.05, seed=42) \
 .toPandas()

pair_sample["FTResult"] = pair_sample["FTResult"].map({
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
})

palette = {
    "Home Win": "steelblue",
    "Draw":     "goldenrod",
    "Away Win": "tomato"
}

g = sns.PairGrid(
    pair_sample,
    vars=pair_cols,
    hue="FTResult",
    palette=palette,
    corner=True,
    diag_sharey=False
)

g.map_lower(
    sns.scatterplot,
    alpha=0.25,
    s=10,
    edgecolor="none"
)

g.map_diag(
    sns.kdeplot,
    fill=True,
    alpha=0.4
)

g.add_legend(
    title="Match Result",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    frameon=True
)

g.figure.suptitle(
    "Pair Plot – Distribution & Relationships by Match Outcome",
    y=1.01,
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ─── PLOT 3 – Final EDA Dashboard ────────────────────────────────────────────

total    = df.count()
hw       = df.filter(col("FTResult") == "H").count()
dr       = df.filter(col("FTResult") == "D").count()
aw       = df.filter(col("FTResult") == "A").count()
home_pct = hw / total * 100
draw_pct = dr / total * 100
away_pct = aw / total * 100

# --- data for subplots ---

# 1. Outcome distribution (pie)
outcome_labels  = ["Home Win", "Draw", "Away Win"]
outcome_values  = [home_pct, draw_pct, away_pct]
outcome_colors  = ["steelblue", "goldenrod", "tomato"]

# 2. Avg goals by outcome (bar)
home_avg_g = df.filter(col("FTResult") == "H").select(mean(col("FTHome") + col("FTAway"))).first()[0]
draw_avg_g = df.filter(col("FTResult") == "D").select(mean(col("FTHome") + col("FTAway"))).first()[0]
away_avg_g = df.filter(col("FTResult") == "A").select(mean(col("FTHome") + col("FTAway"))).first()[0]

# 3. Elo diff vs Home Win % (line)
elo_df  = df.withColumn("EloDiff", col("HomeElo") - col("AwayElo"))
elo_bins_pd = elo_df.withColumn("EloBin", F.round(col("EloDiff") / 25) * 25) \
    .groupBy("EloBin").agg(
        (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
        count("*").alias("n_matches")
    ).filter(col("n_matches") > 50).orderBy("EloBin").toPandas()

# 4. Form3 advantage vs Home Win % (line)
form_bins_pd = df.withColumn("FormAdv", F.round((col("Form3Home") - col("Form3Away")) * 5) / 5) \
    .groupBy("FormAdv").agg(
        (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct"),
        count("*").alias("n_matches")
    ).filter(col("n_matches") > 40).orderBy("FormAdv").toPandas()

# 5. Avg goals per division (bar)
avg_goals_pd = df.groupBy("Division") \
    .agg(mean(col("FTHome") + col("FTAway")).alias("avg_goals")) \
    .orderBy(col("avg_goals").desc()).toPandas()

# 6. Result distribution per division (stacked bar)
div_res_pd = df.groupBy("Division").agg(
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_pct"),
    (count(when(col("FTResult") == "D", 1)) / count("*") * 100).alias("draw_pct"),
    (count(when(col("FTResult") == "A", 1)) / count("*") * 100).alias("away_pct")
).orderBy(col("home_pct").desc()).toPandas()

# 7. Top 10 teams by total wins (bar)
home_wins_pd = df.groupBy("HomeTeam").agg(count(when(col("FTResult") == "H", 1)).alias("wins"))
away_wins_pd = df.groupBy("AwayTeam") \
    .agg(count(when(col("FTResult") == "A", 1)).alias("wins")) \
    .select(col("AwayTeam").alias("HomeTeam"), col("wins"))
top_wins_pd  = home_wins_pd.unionByName(away_wins_pd) \
    .groupBy("HomeTeam").agg(F.sum("wins").alias("total_wins")) \
    .orderBy(col("total_wins").desc()).limit(10).toPandas()

# 8. Over 2.5 rate vs Avg Elo (line)
over_elo_pd = df.withColumn("EloBin", F.round((col("HomeElo") + col("AwayElo")) / 2 / 100) * 100) \
    .groupBy("EloBin").agg(
        (count(when((col("FTHome") + col("FTAway")) > 2, 1)) / count("*") * 100).alias("over_pct"),
        count("*").alias("n")
    ).filter(col("n") > 30).orderBy("EloBin").toPandas()

# 9. Cards per match by division (grouped bar)
cards_div_pd = df.groupBy("Division").agg(
    mean(col("HomeYellow") + col("AwayYellow")).alias("avg_yellows"),
    mean(col("HomeRed")    + col("AwayRed")).alias("avg_reds")
).orderBy(col("avg_yellows").desc()).toPandas()

# ── Build Dashboard ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(24, 22))
fig.patch.set_facecolor("#f4f4f4")

gs = GridSpec(
    3, 3,
    figure=fig,
    hspace=0.45,
    wspace=0.38,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.06
)

panel_style = dict(facecolor="white", edgecolor="#cccccc", linewidth=0.8)

# ── Panel 0,0 – Outcome Pie ───────────────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax00.set_facecolor("white")

wedges, texts, autotexts = ax00.pie(
    outcome_values,
    labels=outcome_labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=outcome_colors,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")

ax00.set_title("Match Outcome Distribution", fontsize=11, fontweight="bold", pad=10)

# ── Panel 0,1 – Avg Goals by Outcome ─────────────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])
ax01.set_facecolor("white")

labels_g = ["Home Win", "Draw", "Away Win"]
values_g = [home_avg_g, draw_avg_g, away_avg_g]

bars = ax01.bar(
    labels_g,
    values_g,
    color=outcome_colors,
    edgecolor="white",
    linewidth=1.2
)

ax01.set_title("Avg Goals per Match by Outcome", fontsize=11, fontweight="bold")
ax01.set_ylabel("Average Goals")
ax01.set_ylim(0, max(values_g) * 1.2)
ax01.grid(axis="y", alpha=0.3)

for i, v in enumerate(values_g):
    ax01.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

# ── Panel 0,2 – Elo Diff vs Home Win % ───────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
ax02.set_facecolor("white")

ax02.plot(
    elo_bins_pd["EloBin"],
    elo_bins_pd["home_win_pct"],
    marker="o",
    color="steelblue",
    linewidth=2,
    markersize=5
)
ax02.fill_between(elo_bins_pd["EloBin"], elo_bins_pd["home_win_pct"], alpha=0.15, color="steelblue")
ax02.axvline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)
ax02.axhline(home_pct, linestyle="--", linewidth=1, color="tomato", alpha=0.6,
             label=f"Overall Home Win: {home_pct:.1f}%")

ax02.set_title("Elo Difference vs Home Win %", fontsize=11, fontweight="bold")
ax02.set_xlabel("Elo Difference (Home − Away) [25pt bins]")
ax02.set_ylabel("Home Win %")
ax02.legend(fontsize=8)
ax02.grid(alpha=0.3)

# ── Panel 1,0 – Form3 Advantage vs Home Win % ────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.set_facecolor("white")

ax10.plot(
    form_bins_pd["FormAdv"],
    form_bins_pd["home_win_pct"],
    marker="o",
    color="goldenrod",
    linewidth=2,
    markersize=5
)
ax10.fill_between(form_bins_pd["FormAdv"], form_bins_pd["home_win_pct"], alpha=0.15, color="goldenrod")
ax10.axvline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)

ax10.set_title("Form3 Advantage vs Home Win %", fontsize=11, fontweight="bold")
ax10.set_xlabel("Form3 Advantage (Home − Away)")
ax10.set_ylabel("Home Win %")
ax10.grid(alpha=0.3)

# ── Panel 1,1 – Avg Goals per Division ───────────────────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
ax11.set_facecolor("white")

overall_avg = avg_goals_pd["avg_goals"].mean()

ax11.bar(
    avg_goals_pd["Division"],
    avg_goals_pd["avg_goals"],
    color=sns.light_palette("steelblue", n_colors=len(avg_goals_pd), reverse=True)
)
ax11.axhline(overall_avg, color="red", linestyle="--", linewidth=1.3,
             label=f"Overall Avg: {overall_avg:.2f}")

ax11.set_title("Avg Goals per Match by Division", fontsize=11, fontweight="bold")
ax11.set_xlabel("Division")
ax11.set_ylabel("Average Goals per Match")
ax11.tick_params(axis="x", rotation=45)
ax11.legend(fontsize=8)

for i, v in enumerate(avg_goals_pd["avg_goals"]):
    ax11.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

# ── Panel 1,2 – Result Distribution per Division (stacked) ───────────────────
ax12 = fig.add_subplot(gs[1, 2])
ax12.set_facecolor("white")

x = np.arange(len(div_res_pd))

ax12.bar(x, div_res_pd["home_pct"], label="Home Win %", color="steelblue")
ax12.bar(x, div_res_pd["draw_pct"], bottom=div_res_pd["home_pct"],
         label="Draw %", color="goldenrod")
ax12.bar(x, div_res_pd["away_pct"],
         bottom=div_res_pd["home_pct"] + div_res_pd["draw_pct"],
         label="Away Win %", color="tomato")

ax12.set_title("Result Distribution per Division (Stacked %)", fontsize=11, fontweight="bold")
ax12.set_xlabel("Division")
ax12.set_ylabel("Match Outcome (%)")
ax12.set_xticks(x)
ax12.set_xticklabels(div_res_pd["Division"], rotation=45, ha="right", fontsize=7)
ax12.legend(fontsize=8, loc="upper right")

# ── Panel 2,0 – Top 10 Teams by Total Wins ───────────────────────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax20.set_facecolor("white")

ax20.bar(
    top_wins_pd["HomeTeam"],
    top_wins_pd["total_wins"],
    color=sns.light_palette("green", n_colors=len(top_wins_pd), reverse=True)
)

ax20.set_title("Top 10 Teams by Total Wins", fontsize=11, fontweight="bold")
ax20.set_xlabel("Team")
ax20.set_ylabel("Total Wins")
ax20.tick_params(axis="x", rotation=45)

for i, v in enumerate(top_wins_pd["total_wins"]):
    ax20.text(i, v + 1, str(v), ha="center", fontsize=8)

# ── Panel 2,1 – Over 2.5 Goals vs Avg Elo ────────────────────────────────────
ax21 = fig.add_subplot(gs[2, 1])
ax21.set_facecolor("white")

ax21.plot(
    over_elo_pd["EloBin"],
    over_elo_pd["over_pct"],
    marker="o",
    color="mediumpurple",
    linewidth=2,
    markersize=5
)
ax21.fill_between(over_elo_pd["EloBin"], over_elo_pd["over_pct"], alpha=0.15, color="mediumpurple")
ax21.axhline(50, linestyle="--", linewidth=1, color="gray", alpha=0.6)

ax21.set_title("Over 2.5 Goals Rate vs Average Elo", fontsize=11, fontweight="bold")
ax21.set_xlabel("Average Elo (binned)")
ax21.set_ylabel("Over 2.5 Rate %")
ax21.grid(alpha=0.3)

# ── Panel 2,2 – Avg Cards per Division ───────────────────────────────────────
ax22 = fig.add_subplot(gs[2, 2])
ax22.set_facecolor("white")

x_c = np.arange(len(cards_div_pd))
w   = 0.4

ax22.bar(x_c - w/2, cards_div_pd["avg_yellows"], w, label="Avg Yellows", color="gold")
ax22.bar(x_c + w/2, cards_div_pd["avg_reds"],    w, label="Avg Reds",    color="red")

ax22.set_title("Avg Cards per Match by Division", fontsize=11, fontweight="bold")
ax22.set_xlabel("Division")
ax22.set_ylabel("Average Cards per Match")
ax22.set_xticks(x_c)
ax22.set_xticklabels(cards_div_pd["Division"], rotation=45, ha="right", fontsize=7)
ax22.legend(fontsize=8)

# ── Master Title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Football Matches EDA Dashboard  –  2000 to 2025",
    fontsize=17,
    fontweight="bold",
    y=0.97
)

plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()