import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv("Matches_cleaned.csv")
test  = pd.read_csv("Matches_test.csv")

PRE_MATCH_FEATURES = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home",
    "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway"
]

TARGET = "FTResult"

X_train = train[PRE_MATCH_FEATURES]
y_train = train[TARGET]
X_test  = test[PRE_MATCH_FEATURES]
y_test  = test[TARGET]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


print("=" * 55)
print("MODEL 1 — Random Forest")
print("=" * 55)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_enc)
rf_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test_enc, rf_pred):.4f}\n")
print(classification_report(y_test_enc, rf_pred, target_names=le.classes_))

print("=" * 55)
print("MODEL 2 — KNN")
print("=" * 55)

knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn.fit(X_train_scaled, y_train_enc)
knn_pred = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test_enc, knn_pred):.4f}\n")
print(classification_report(y_test_enc, knn_pred, target_names=le.classes_))

print("=" * 55)
print("MODEL 3 — XGBoost")
print("=" * 55)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train_enc)
xgb_pred = xgb.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test_enc, xgb_pred):.4f}\n")
print(classification_report(y_test_enc, xgb_pred, target_names=le.classes_))


print("=" * 55)
print("MODEL 4 — Neural Network")
print("=" * 55)
y_train_cat = to_categorical(y_train_enc, num_classes=3)
y_test_cat  = to_categorical(y_test_enc,  num_classes=3)

nn = Sequential([
    Dense(256, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(3, activation="softmax")
])

nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

nn.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.1,
    epochs=100,
    batch_size=256,
    callbacks=[es],
    verbose=1
)

nn_pred = np.argmax(nn.predict(X_test_scaled), axis=1)
print(f"\nAccuracy: {accuracy_score(y_test_enc, nn_pred):.4f}\n")
print(classification_report(y_test_enc, nn_pred, target_names=le.classes_))


