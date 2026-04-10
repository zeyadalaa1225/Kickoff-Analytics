from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, median,lit
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
Spark=SparkSession.builder.appName("FootballAnalysis").config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate()
Spark.sparkContext.setLogLevel("WARN")
df = Spark.read.csv("Matches_cleaned.csv", header=True, inferSchema=True)
print(df.count(), "rows in cleaned Matches.csv" )


# ─────────────────────────────────────────────
# Descriptive / Overview Questions
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1  DESCRIPTIVE / OVERVIEW ANALYSIS")
print("="*60)
# Q1 – Top 10 highest-scoring teams (all-time)
home_goals = df.groupBy("HomeTeam").agg(F.sum("HTHome").alias("total_goals"))
away_goals = df.groupBy("AwayTeam").agg(F.sum("HTAway").alias("total_goals")) \
               .withColumnRenamed("AwayTeam", "HomeTeam")
total_goals = home_goals.unionByName(away_goals) \
                        .groupBy("HomeTeam").agg(F.sum("total_goals").alias("total_goals"))
top10_goals = total_goals.orderBy(col("total_goals").desc()).limit(10).toPandas()
print("\nQ1 – Top 10 Highest-Scoring Teams"); print(top10_goals.to_string(index=False))
fig = plt.figure(figsize=(10,6))
plt.bar(top10_goals["HomeTeam"], top10_goals["total_goals"], color=sns.light_palette("blue", n_colors=len(top10_goals), reverse=True))
plt.title("Top 10 Highest-Scoring Teams (2000–2025)")
plt.xlabel("Team")
plt.ylabel("Total Goals Scored")
plt.xticks(rotation=45)
for i, v in enumerate(top10_goals["total_goals"]):
    plt.text(i, v + 2, str(v), ha='center')
plt.show()


# Q2 – Home Advantage: home goals vs away goals per team 
home_g = df.groupBy("HomeTeam").agg(F.sum("HTHome").alias("home_goals"))
away_g = df.groupBy("AwayTeam").agg(F.sum("HTAway").alias("away_goals")) \
           .withColumnRenamed("AwayTeam", "HomeTeam")
ha_df = home_g.join(away_g, "HomeTeam") \
              .orderBy(col("home_goals").desc()) \
              .limit(15) \
              .toPandas()
teams = ha_df["HomeTeam"]
home_goals = ha_df["home_goals"]
away_goals = ha_df["away_goals"]
x = np.arange(len(teams)) 
width = 0.35  
fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, home_goals, width, label='Home Goals', color=sns.light_palette("blue", n_colors=len(teams), reverse=True))
rects2 = ax.bar(x + width/2, away_goals, width, label='Away Goals', color=sns.light_palette("orange", n_colors=len(teams), reverse=True))

ax.set_ylabel('Goals')
ax.set_title('Home vs Away Goals for Top 15 Teams')
ax.set_xticks(x)
ax.set_xticklabels(teams, rotation=45, ha='right')
ax.legend()
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',                     
                    xy=(rect.get_x() + rect.get_width()/2, height),  
                    xytext=(0, 3),                         
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Q3 – Matches per division
matches_per_division =df.groupBy("Division").agg(count("*").alias("match_count")).orderBy(col("match_count").desc())
print("Matches per division:")
matches_per_division.show(truncate=False)
matches_df = matches_per_division.toPandas()

plt.figure(figsize=(10,6))
plt.bar(matches_df["Division"], matches_df["match_count"], color='skyblue')
plt.title("Number of Matches per Division")
plt.xlabel("Division")
plt.ylabel("Number of Matches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Q4 – Average goals per match by division
avg_goals_per_division = df.groupBy("Division").agg(
    mean(col("HTHome") + col("HTAway")).alias("avg_goals_per_match")
).orderBy(col("avg_goals_per_match").desc())
print("Average goals per match by division")
avg_goals_per_division.show(10, truncate=False)
avg_goals_df = avg_goals_per_division.toPandas()
overall_avg = avg_goals_df["avg_goals_per_match"].mean()
plt.figure(figsize=(16,8))
plt.bar(avg_goals_df["Division"], avg_goals_df["avg_goals_per_match"], color='salmon')
plt.axhline(overall_avg, color='blue', linestyle='--', label=f"Overall Avg: {overall_avg:.2f}")
plt.title("Average Goals per Match by Division")
plt.xlabel("Division")
plt.ylabel("Average Goals per Match")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(avg_goals_df["avg_goals_per_match"]):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
plt.legend()
plt.tight_layout()
plt.show()
 
# Q5 – Match outcome distribution
total = df.count()
hw = df.filter(col("FTResult") == "H").count()
dr = df.filter(col("FTResult") == "D").count()
aw = df.filter(col("FTResult") == "A").count()
labels = ["Home Win", "Draw", "Away Win"]
vals   = [hw/total*100, dr/total*100, aw/total*100]
print(f"\nMatch Outcomes: Home {vals[0]:.1f}%  Draw {vals[1]:.1f}%  Away {vals[2]:.1f}%")
colors = ['skyblue', 'lightgreen', 'salmon']

plt.figure(figsize=(8,6))
plt.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor':'black'})
plt.title("Match Outcome Distribution")
plt.axis('equal')
plt.show()

# Q6 – Which teams have played the most matches?
home_matches = df.groupBy("HomeTeam").agg(count("*").alias("match_count"))
away_matches = df.groupBy("AwayTeam").agg(count("*").alias("match_count")) \
    .select(col("AwayTeam").alias("HomeTeam"), col("match_count"))
total_matches = home_matches.unionByName(away_matches) \
    .groupBy("HomeTeam") \
    .agg(F.sum("match_count").alias("total_matches"))
most_active_teams = total_matches.orderBy(col("total_matches").desc())
print("Teams with the most matches played")
most_active_teams.show(10, truncate=False)
plot_df = most_active_teams.limit(10).toPandas()
plt.figure(figsize=(12,6))
plt.bar(plot_df["HomeTeam"], plot_df["total_matches"], color='steelblue')
plt.title("Top 10 Teams With Most Matches Played")
plt.xlabel("Team")
plt.ylabel("Total Matches Played")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(plot_df["total_matches"]):
    plt.text(i, v + 1, str(v), ha='center')
plt.tight_layout()
plt.show()

# Q7 – Top 10 teams with the most wins
home_wins = df.groupBy("HomeTeam").agg(count(when(col("FTResult") == "H", 1)).alias("wins"))
away_wins = df.groupBy("AwayTeam").agg(count(when(col("FTResult") == "A", 1)).alias("wins")) \
    .select(col("AwayTeam").alias("HomeTeam"), col("wins"))
total_wins = home_wins.unionByName(away_wins) \
    .groupBy("HomeTeam") \
    .agg(F.sum("wins").alias("total_wins"))
top_winning_teams = total_wins.orderBy(col("total_wins").desc())
print("Top 10 teams with the most wins")
top_winning_teams.show(10, truncate=False)
plot_df = top_winning_teams.limit(10).toPandas()
plt.figure(figsize=(12,6))
plt.bar(plot_df["HomeTeam"], plot_df["total_wins"], color=sns.light_palette("seagreen", n_colors=len(plot_df), reverse=True))
plt.title("Top 10 Teams with Most Wins")
plt.xlabel("Team")
plt.ylabel("Total Wins")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(plot_df["total_wins"]):
    plt.text(i, v + 1, str(v), ha='center')
plt.tight_layout()
plt.show()

# Q8 – Top 10 teams by number of home wins
home_wins = df.filter(col("FTResult") == "H") \
    .groupBy("HomeTeam") \
    .agg(count("*").alias("home_wins")) \
    .orderBy(col("home_wins").desc())
top10_pd = home_wins.limit(10).toPandas()
plt.figure(figsize=(12,6))
colors = sns.light_palette("green", n_colors=len(top10_pd), reverse=True)
bars = plt.bar(top10_pd["HomeTeam"], top10_pd["home_wins"], color=colors)
plt.title("Top 10 Teams by Home Wins")
plt.xlabel("Team")
plt.ylabel("Number of Home Wins")
plt.xticks(rotation=45, ha='right')
for bar, v in zip(bars, top10_pd["home_wins"]):
    plt.text(bar.get_x() + bar.get_width()/2, v + 1, str(v), ha='center')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – ELO & TEAM STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SECTION 2 – ELO & TEAM STRENGTH ANALYSIS")
print("="*60)
 
# Q9 – Top 10 teams by average Elo per year
matches_year = df.withColumn("Year", F.year(F.to_date("MatchDate", "yyyy-MM-dd")))
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
team_elo = home_elo.unionByName(away_elo).groupBy("Team", "Year").agg(
    F.mean("Elo").alias("MeanElo")
).orderBy(col("MeanElo").desc()).limit(10)
print("Top 10 teams by average Elo (year-wise)")
team_elo.show(truncate=False)
plot_df = team_elo.toPandas()
plt.figure(figsize=(12,6))
labels = plot_df["Team"] + " (" + plot_df["Year"].astype(str) + ")"
plt.bar(labels, plot_df["MeanElo"], color=sns.light_palette("purple", n_colors=len(plot_df), reverse=True))
plt.title("Top 10 Teams by Average Elo per Year")
plt.xlabel("Team (Year)")
plt.ylabel("Average Elo")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(plot_df["MeanElo"]):
    plt.text(i, v + 2, f"{v:.1f}", ha='center')
plt.tight_layout()
plt.show()

# Q10 – Do higher‐Elo teams win more frequently than lower‐Elo teams?
count_higher_elo_wins = df.filter(col("FTResult") == "H").filter(col("HomeElo") > col("AwayElo")).count() + \
                         df.filter(col("FTResult") == "A").filter(col("AwayElo") > col("HomeElo")).count()
count_lower_elo_wins = df.filter(col("FTResult") == "H").filter(col("HomeElo") < col("AwayElo")).count() + \
                        df.filter(col("FTResult") == "A").filter(col("AwayElo") < col("HomeElo")).count()
draws_count = df.filter(col("FTResult") == "D").count()
print(f"Higher-Elo wins: {count_higher_elo_wins}, Lower-Elo wins: {count_lower_elo_wins}, Draws: {draws_count}")
labels = ["Higher Elo Wins", "Lower Elo Wins", "Draws"]
values = [count_higher_elo_wins, count_lower_elo_wins, draws_count]
plt.figure(figsize=(8,6))
colors = ['seagreen', 'salmon', 'skyblue']
plt.bar(labels, values, color=colors)
plt.title("Match Outcomes vs Elo Strength")
plt.xlabel("Outcome Type")
plt.ylabel("Number of Matches")
for i, v in enumerate(values):
    plt.text(i, v + max(values)*0.01, str(v), ha='center')
plt.tight_layout()
plt.show()


# Q11 – Percentage of matches where the lower-Elo team wins (home vs away)
home_wins_lower_elo = df.filter((col("FTResult") == "H") & (col("HomeElo") < col("AwayElo"))).count()
away_wins_lower_elo = df.filter((col("FTResult") == "A") & (col("AwayElo") < col("HomeElo"))).count()
home_win_lower_elo_percentage = (home_wins_lower_elo / df.count()) * 100
away_win_lower_elo_percentage = (away_wins_lower_elo / df.count()) * 100
labels = ["Home (Lower Elo)", "Away (Lower Elo)"]
values = [home_win_lower_elo_percentage, away_win_lower_elo_percentage]
plt.figure(figsize=(10,6))
colors = sns.color_palette("Set2", len(values))
bars = plt.bar(labels, values, color=colors)
plt.title("Lower-Elo Team Wins: Home vs Away (%)")
plt.xlabel("Match Outcome Type")
plt.ylabel("Percentage of Total Matches (%)")
plt.ylim(0, max(values)*1.2)
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.2f}%", ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# Q12 - Elo difference distribution by outcome (are bigger Elo gaps more likely to lead to wins for the stronger team?)
elo_df = df.withColumn("EloDiff", (col("HomeElo") - col("AwayElo")))
elo_df = elo_df.withColumn(
    "StrongerTeamWon",
    when(
        ((col("EloDiff") > 0) & (col("FTResult") == "H")) |
        ((col("EloDiff") < 0) & (col("FTResult") == "A")), 1
    )
    .when(col("FTResult") == "D", 0)
    .otherwise(-1)
)
elo_df = elo_df.withColumn("AbsoluteEloDiff", F.abs(col("EloDiff")))
plot_df = elo_df.select("AbsoluteEloDiff", "StrongerTeamWon").sample(fraction=0.3, seed=42).toPandas()
binary_df = plot_df[plot_df["StrongerTeamWon"] != 0]
corr_pb, pvalue = stats.pointbiserialr(
    binary_df["StrongerTeamWon"],  
    binary_df["AbsoluteEloDiff"]
)
print(f"Point-Biserial Correlation: {corr_pb:.4f}  (p-value: {pvalue:.4f})")
stronger_won  = plot_df[plot_df["StrongerTeamWon"] ==  1]["AbsoluteEloDiff"]
draw          = plot_df[plot_df["StrongerTeamWon"] ==  0]["AbsoluteEloDiff"]
weaker_won    = plot_df[plot_df["StrongerTeamWon"] == -1]["AbsoluteEloDiff"]

plt.figure(figsize=(8, 5))
bp = plt.boxplot(
    [stronger_won, draw, weaker_won],
    tick_labels=["Stronger Won", "Draw", "Weaker Won"],
    patch_artist=True,
    showfliers=False
)
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Elo Strength  \nCorrelation: {corr_pb:.4f}")
plt.xlabel("Match Outcome")
plt.ylabel("Absolute Elo Difference |Home − Away|")
plt.tight_layout()
plt.show()

#Q13 – Win rate by Elo band
elo_band_df = elo_df.withColumn("EloBand",
    when(col("EloDiff") >= 200,  "Much Stronger (+200)")
    .when(col("EloDiff") >= 100,  "Stronger (+100 to 200)")
    .when(col("EloDiff") >= 0,    "Slightly Stronger (0–100)")
    .when(col("EloDiff") >= -100, "Slightly Weaker (-100–0)")
    .when(col("EloDiff") >= -200, "Weaker (-200 to -100)")
    .otherwise("Much Weaker (<-200)")
)
band_stats = elo_band_df.groupBy("EloBand").agg(
    count("*").alias("total"),
    count(when(col("FTResult")=="H", 1)).alias("home_wins"),
    count(when(col("FTResult")=="D", 1)).alias("draws"),
    count(when(col("FTResult")=="A", 1)).alias("away_wins")
).toPandas()
band_stats["home_win_pct"] = band_stats["home_wins"] / band_stats["total"] * 100
band_stats["draw_pct"]     = band_stats["draws"]     / band_stats["total"] * 100
band_stats["away_win_pct"] = band_stats["away_wins"] / band_stats["total"] * 100
order = ["Much Stronger (+200)","Stronger (+100 to 200)","Slightly Stronger (0–100)",
         "Slightly Weaker (-100–0)","Weaker (-200 to -100)","Much Weaker (<-200)"]
band_stats = band_stats.set_index("EloBand").reindex(order).reset_index()
print("\nQ9 – Win rates by Elo band")
print(band_stats[["EloBand","total","home_win_pct","draw_pct","away_win_pct"]].to_string(index=False))
x = np.arange(len(band_stats))
w = 0.28
plt.figure(figsize=(14,6))
bars1 = plt.bar(x - w, band_stats["home_win_pct"], w, label="Home Win %", color="skyblue")
bars2 = plt.bar(x,     band_stats["draw_pct"],     w, label="Draw %",     color="orange")
bars3 = plt.bar(x + w, band_stats["away_win_pct"], w, label="Away Win %", color="seagreen")
plt.xticks(x, band_stats["EloBand"], rotation=30, ha="right")
plt.ylabel("Percentage (%)")
plt.title("Match Outcomes by Elo Difference Bands")
plt.legend()
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha='center')

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – FORM & MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3 – FORM & MOMENTUM ANALYSIS")
print("="*60)
 
# Q14 – Relationship between Form3 difference and Home win distribution (scatter)
form_df = df.withColumn("HomeFormAdv", col("Form3Home") - col("Form3Away"))

scatter_df = form_df.withColumn("Form3Adv", col("Form3Home") - col("Form3Away")).groupBy(
    F.round("Form3Adv", 0).alias("Form3Adv")
).agg(
    count("*").alias("total"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct")
).orderBy("Form3Adv").toPandas()

scatter_df2 = form_df.withColumn("Form5Adv", col("Form5Home") - col("Form5Away")).groupBy(
    F.round("Form5Adv", 0).alias("Form5Adv")
).agg(
    count("*").alias("total"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct")
).orderBy("Form5Adv").toPandas()

plt.figure(figsize=(12,6))

plt.plot(scatter_df["Form3Adv"], scatter_df["home_win_pct"], marker="o", color="darkgreen", label="Form3")
plt.plot(scatter_df2["Form5Adv"], scatter_df2["home_win_pct"], marker="o", color="blue", label="Form5")

plt.title("Home Win % vs Form Advantage")
plt.xlabel("Form Advantage")
plt.ylabel("Home Win %")
plt.grid(alpha=0.3)
plt.legend()

plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 –  CARDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4 - CARDS ANALYSIS")
print("="*60)

# Q15 – Teams with the most yellow cards

home_yellow = df.groupBy("HomeTeam").agg(F.sum("HomeYellow").alias("yellows"))
away_yellow = df.groupBy("AwayTeam").agg(F.sum("AwayYellow").alias("yellows")) \
    .withColumnRenamed("AwayTeam","HomeTeam")

total_yellow = home_yellow.unionByName(away_yellow) \
    .groupBy("HomeTeam") \
    .agg(F.sum("yellows").alias("total_yellows")) \
    .orderBy(col("total_yellows").desc()) \
    .limit(15) \
    .toPandas()

plt.figure(figsize=(12,6))
colors = sns.light_palette("gold", n_colors=len(total_yellow), reverse=True)

plt.bar(total_yellow["HomeTeam"], total_yellow["total_yellows"], color=colors)
plt.title("Top 15 Teams by Yellow Cards")
plt.xlabel("Team")
plt.ylabel("Total Yellow Cards")
plt.xticks(rotation=45, ha='right')

for i, v in enumerate(total_yellow["total_yellows"]):
    plt.text(i, v + 1, str(v), ha='center')

plt.tight_layout()
plt.show()


# Q16 – Teams with the most red cards

home_red = df.groupBy("HomeTeam").agg(F.sum("HomeRed").alias("reds"))
away_red = df.groupBy("AwayTeam").agg(F.sum("AwayRed").alias("reds")) \
    .withColumnRenamed("AwayTeam","HomeTeam")

total_red = home_red.unionByName(away_red) \
    .groupBy("HomeTeam") \
    .agg(F.sum("reds").alias("total_reds")) \
    .orderBy(col("total_reds").desc()) \
    .limit(15) \
    .toPandas()

plt.figure(figsize=(12,6))
colors = sns.light_palette("red", n_colors=len(total_red), reverse=True)

plt.bar(total_red["HomeTeam"], total_red["total_reds"], color=colors)
plt.title("Top 15 Teams by Red Cards")
plt.xlabel("Team")
plt.ylabel("Total Red Cards")
plt.xticks(rotation=45, ha='right')

for i, v in enumerate(total_red["total_reds"]):
    plt.text(i, v + 0.1, str(v), ha='center')

plt.tight_layout()
plt.show()


# Q17 – Home Win % vs Red Card Advantage (with Yellow comparison)

card_df = df.withColumn("HomeRedAdv", col("HomeRed") - col("AwayRed")) \
            .withColumn("HomeYellowAdv", col("HomeYellow") - col("AwayYellow"))

red_scatter = card_df.groupBy(F.round("HomeRedAdv", 0).alias("RedAdv")).agg(
    count("*").alias("total"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct")
).orderBy("RedAdv").toPandas()

yellow_scatter = card_df.groupBy(F.round("HomeYellowAdv", 0).alias("YellowAdv")).agg(
    count("*").alias("total"),
    (count(when(col("FTResult") == "H", 1)) / count("*") * 100).alias("home_win_pct")
).orderBy("YellowAdv").toPandas()

plt.figure(figsize=(12,6))

plt.plot(red_scatter["RedAdv"], red_scatter["home_win_pct"],
         marker="o", color="red", label="Red Card Advantage")

plt.plot(yellow_scatter["YellowAdv"], yellow_scatter["home_win_pct"],
         marker="o", color="gold", label="Yellow Card Advantage")

plt.title("Home Win % vs Card Advantage")
plt.xlabel("Card Advantage (Home - Away)")
plt.ylabel("Home Win %")
plt.grid(alpha=0.3)
plt.legend()

plt.show()

# Q18 Average cards per division
cards_div = df.groupBy("Division").agg(
    mean(col("HomeYellow") + col("AwayYellow")).alias("avg_yellows_per_match"),
    mean(col("HomeRed") + col("AwayRed")).alias("avg_reds_per_match")
).orderBy(col("avg_yellows_per_match").desc()).toPandas()

print(" Average Cards per Division")
print(cards_div.to_string(index=False))

plt.figure(figsize=(14,6))

x = np.arange(len(cards_div))
w = 0.4

plt.bar(x - w/2, cards_div["avg_yellows_per_match"], w, label="Avg Yellows", color="gold")
plt.bar(x + w/2, cards_div["avg_reds_per_match"], w, label="Avg Reds", color="red")

plt.xticks(x, cards_div["Division"], rotation=45, ha="right")
plt.title("Average Cards per Match by Division")
plt.xlabel("Division")
plt.ylabel("Average Cards per Match")
plt.legend()

plt.tight_layout()
plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – HALF-TIME vs FULL-TIME PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5 – HALF-TIME vs FULL-TIME PATTERNS")
print("="*60)

# Q17 – Half-time result to full-time result transition matrix

ht_ft = df.filter(col("HTResult").isin(["H","D","A"])) \
          .groupBy("HTResult","FTResult") \
          .agg(count("*").alias("count")) \
          .toPandas()
pivot = ht_ft.pivot(index="HTResult", columns="FTResult", values="count").fillna(0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
print(" HT→FT Transition Matrix (%)")
print(pivot_pct.round(1).to_string())
plt.figure(figsize=(8,5))
im = plt.imshow(pivot_pct.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
plt.xticks(range(len(pivot_pct.columns)), pivot_pct.columns)
plt.yticks(range(len(pivot_pct.index)), pivot_pct.index)

plt.xlabel("Full-Time Result")
plt.ylabel("Half-Time Result")
plt.title("Half-Time → Full-Time Result Transition (%)")

for i in range(len(pivot_pct.index)):
    for j in range(len(pivot_pct.columns)):
        plt.text(j, i, f"{pivot_pct.values[i,j]:.0f}%", ha="center", va="center", color="black")

plt.colorbar(im, label="% of matches")

plt.tight_layout()
plt.show()

# Q18 – Teams that come back most from HT deficit (HT losing, FT win)
home_cb = df.filter((col("HTResult") == "A") & (col("FTResult") == "H")).groupBy("HomeTeam") .agg(count("*").alias("comebacks"))
away_cb = df.filter((col("HTResult") == "H") & (col("FTResult") == "A")).groupBy("AwayTeam").agg(count("*").alias("comebacks")).withColumnRenamed("AwayTeam", "HomeTeam")
comebacks = home_cb.unionByName(away_cb).groupBy("HomeTeam").agg(F.sum("comebacks").alias("comebacks")).orderBy(F.col("comebacks").desc()).limit(10).toPandas()
plt.figure(figsize=(12,6))
colors = sns.light_palette("coral", n_colors=len(comebacks), reverse=True)
plt.bar(comebacks["HomeTeam"], comebacks["comebacks"], color=colors)
plt.title("Top 10 Teams with Most Comebacks (HT Loss → FT Win)")
plt.xlabel("Team")
plt.ylabel("Number of Comebacks")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(comebacks["comebacks"]):
    plt.text(i, v + 0.1, str(v), ha='center')
plt.tight_layout()
plt.show()

# Q19 - Teams that blow leads most from HT lead (HT win, FT loss)
home_cb = df.filter((col("HTResult") == "H") & (col("FTResult") == "A")).groupBy("HomeTeam") .agg(count("*").alias("comebacks"))
away_cb = df.filter((col("HTResult") == "A") & (col("FTResult") == "H")).groupBy("AwayTeam").agg(count("*").alias("comebacks")).withColumnRenamed("AwayTeam", "HomeTeam")
comebacks = home_cb.unionByName(away_cb).groupBy("HomeTeam").agg(F.sum("comebacks").alias("comebacks")).orderBy(F.col("comebacks").desc()).limit(10).toPandas()
plt.figure(figsize=(12,6))
colors = sns.light_palette("coral", n_colors=len(comebacks), reverse=True)
plt.bar(comebacks["HomeTeam"], comebacks["comebacks"], color=colors)
plt.title("Top 10 Teams with Most Blown Leads (HT Win → FT Loss)")
plt.xlabel("Team")
plt.ylabel("Number of Blown Leads")
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(comebacks["comebacks"]):
    plt.text(i, v + 0.1, str(v), ha='center')
plt.tight_layout()
plt.show()

# Q20 – Average FT goals when team is winning vs losing at HT

ht_goals = df.filter(col("HTResult").isin(["H","D","A"])) \
    .groupBy("HTResult").agg(
        mean(col("FTHome")+col("FTAway")).alias("avg_ft_goals"),
        mean(col("FTHome")).alias("avg_ft_home"),
        mean(col("FTAway")).alias("avg_ft_away"),
        count("*").alias("matches")
    ).orderBy("HTResult").toPandas()
print("FT Goals by HT Result")
print(ht_goals.to_string(index=False))
x = np.arange(len(ht_goals))
w = 0.25
plt.figure(figsize=(10,6))
plt.bar(x - w, ht_goals["avg_ft_goals"], w, label="Total Goals")
plt.bar(x,     ht_goals["avg_ft_home"],  w, label="Home Goals")
plt.bar(x + w, ht_goals["avg_ft_away"],  w, label="Away Goals")
plt.xticks(x, ht_goals["HTResult"])
plt.xlabel("Half-Time Result")
plt.ylabel("Average Goals")
plt.title("Average Full-Time Goals by Half-Time Result")
plt.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – BETTING ODDS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6 – BETTING ODDS ANALYSIS")
print("="*60)

# Q21 – Value betting: how often does the underdog (higher odds) win?
underdog_home = df.filter(col("OddHome") > col("OddAway"))
underdog_home_wins = underdog_home.filter(col("FTResult") == "H").count()
underdog_home_total = underdog_home.count()

print(f"Underdog (home) win rate: {underdog_home_wins / underdog_home_total * 100:.1f}% "
      f"({underdog_home_wins:,}/{underdog_home_total:,})")

# Q22 – Accuracy of bookmaker favorites (lowest odds) in predicting match winners

predicted_home = df.filter((col("OddHome") < col("OddAway")) & (col("FTResult") == "H")).count()
predicted_away = df.filter((col("OddAway") < col("OddHome")) & (col("FTResult") == "A")).count()
predicted_total = predicted_home + predicted_away
total_matches = df.count()
print(f"Favorite win accuracy: {predicted_total / total_matches * 100:.1f}% "
      f"({predicted_total:,}/{total_matches:,})")

# Q23 – Does having both Form AND Elo advantage guarantee a win?
combined_df = df.withColumn("EloDiff", col("HomeElo") - col("AwayElo")) \
    .withColumn("FormDiff", col("Form3Home") - col("Form3Away")) \
    .withColumn("AdvantageType",
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
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha='center')

plt.tight_layout()
plt.show()

# Q24 – Most common full-time scorelines (top 15)

scoreline = df.withColumn(
    "Scoreline",
    F.concat(col("FTHome").cast("string"), lit("-"), col("FTAway").cast("string"))
).groupBy("Scoreline").agg(
    count("*").alias("count")
).orderBy(col("count").desc()).limit(15).toPandas()

print("Top 15 Most Common Scorelines")
print(scoreline.to_string(index=False))

plt.figure(figsize=(14,6))
colors = sns.color_palette("husl", len(scoreline))

bars = plt.bar(scoreline["Scoreline"], scoreline["count"], color=colors)

plt.xlabel("Scoreline")
plt.ylabel("Frequency")
plt.title("Top 15 Most Common Full-Time Scorelines")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 10, f"{int(h)}", ha='center')

plt.tight_layout()
plt.show()

# Q25 – Correlation matrix
corr_cols = ["HomeElo","AwayElo","Form3Home","Form3Away","Form5Home","Form5Away",
             "FTHome","FTAway","HomeYellow","AwayYellow","HomeRed","AwayRed"]

corr_pd = df.select(corr_cols).dropna().sample(fraction=0.3, seed=42).toPandas()
corr_matrix = corr_pd.corr()

print("Feature Correlation Matrix")

plt.figure(figsize=(12,10))

im = plt.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
plt.yticks(range(len(corr_cols)), corr_cols)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        plt.text(j, i, f"{corr_matrix.values[i,j]:.2f}", ha="center", va="center",
                 color="black" if abs(corr_matrix.values[i,j]) > 0.5 else "white", fontsize=7)

plt.colorbar(im, label="Correlation")

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()