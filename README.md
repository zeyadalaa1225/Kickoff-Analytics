# Kickoff-Analytics
⚽ Football Match Analysis & Outcome Prediction

📌 Overview

This project aims to analyze historical football match data to better understand the key factors that influence match outcomes. By exploring patterns across different leagues, seasons, and match conditions, the goal is to uncover meaningful insights that explain why teams win, lose, or draw.


## Architecture

```
Raw CSV Dataset
      │
      ▼
┌─────────────────────────────────┐
│   PySpark Ingestion Layer       │
│   (SparkSession — pseudo-dist.) │
└────────────┬────────────────────┘
             │
      80/20 train-test split
      (all stats fit on TRAIN only)
             │
      ┌──────┴──────┐
      ▼             ▼
   TRAIN SET     TEST SET
      │             │
      ▼             │
┌─────────────┐    │
│  Cleaning   │────┘  (apply same transforms)
│  Pipeline   │
│  (10 steps) │
└──────┬──────┘
       │
       ├──► EDA + Visualization (Pyplot / Matplotlib)
       │
       ├──► Feature Engineering
       │
       ├──► MapReduce KMeans Clustering (Descriptive)
       │
       └──► Classification Models (Predictive)
                 ├── Random Forest
                 ├── Weighted Random Forest
                 ├── KNN (MapReduce)
                 ├── XGBoost
                 ├── Weighted XGBoost
                 ├── Neural Network
                 └── Weighted Neural Network
```
