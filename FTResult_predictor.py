import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping,LambdaCallback
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
train = pd.read_csv("Matches_cleaned.csv")
test  = pd.read_csv("Matches_test.csv")

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

X_train = train[ALL_FEATURES]
y_train = train[TARGET]

X_test  = test[ALL_FEATURES]
y_test  = test[TARGET]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
classes = np.unique(y_train_enc)
class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train_enc)
class_weight_dict   = dict(enumerate(class_weights_array))
sample_weights = compute_sample_weight("balanced", y=y_train_enc)

print("=" * 55)
print("MODEL 1 — Random Forest")
print("=" * 55)
rf=RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train_enc)
rf_train_pred = rf.predict(X_train)
rf_pred       = rf.predict(X_test)

rf_train_acc  = accuracy_score(y_train_enc, rf_train_pred)
rf_test_acc   = accuracy_score(y_test_enc,  rf_pred)

print(f"Train Accuracy: {rf_train_acc:.4f}")
print(f"Test  Accuracy: {rf_test_acc:.4f}")
print(f"Gap           : {rf_train_acc - rf_test_acc:.4f}  {'⚠ Overfitting' if rf_train_acc - rf_test_acc > 0.1 else '✓ OK'}\n")
print(classification_report(y_test_enc, rf_pred, target_names=le.classes_))

print("=" * 55)
print("MODEL 2 — KNN")
print("=" * 55)

knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1,weights="distance")
knn.fit(X_train_scaled, y_train_enc)
knn_train_pred = knn.predict(X_train_scaled)
knn_pred       = knn.predict(X_test_scaled)

knn_train_acc  = accuracy_score(y_train_enc, knn_train_pred)
knn_test_acc   = accuracy_score(y_test_enc,  knn_pred)

print(f"Train Accuracy: {knn_train_acc:.4f}")
print(f"Test  Accuracy: {knn_test_acc:.4f}")
print(f"Gap           : {knn_train_acc - knn_test_acc:.4f}  {'⚠ Overfitting' if knn_train_acc - knn_test_acc > 0.1 else '✓ OK'}\n")
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
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(
    X_train_scaled, y_train_enc,
    eval_set=[ (X_test_scaled, y_test_enc)],
    sample_weight=sample_weights,
)
xgb_train_pred = xgb.predict(X_train_scaled)
xgb_pred       = xgb.predict(X_test_scaled)

xgb_train_acc  = accuracy_score(y_train_enc, xgb_train_pred)
xgb_test_acc   = accuracy_score(y_test_enc,  xgb_pred)

print(f"\nTrain Accuracy: {xgb_train_acc:.4f}")
print(f"Test  Accuracy: {xgb_test_acc:.4f}")
print(f"Gap           : {xgb_train_acc - xgb_test_acc:.4f}  {'⚠ Overfitting' if xgb_train_acc - xgb_test_acc > 0.1 else '✓ OK'}\n")
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
print_cb = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f} | "
        f"Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}"
    )
)

history =nn.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.1,
    epochs=100,
    batch_size=256,
    callbacks=[es, print_cb],
    verbose=0,
    class_weight=class_weight_dict
)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],     label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss"); plt.legend()

plt.tight_layout()
plt.show()
nn_pred = np.argmax(nn.predict(X_test_scaled), axis=1)
print(f"\nAccuracy: {accuracy_score(y_test_enc, nn_pred):.4f}\n")
print(classification_report(y_test_enc, nn_pred, target_names=le.classes_))


