from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, recall_score, precision_score, roc_auc_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["HighBP","HighChol","BMI","Smoker","PhysActivity","Fruits","Veggies","GenHlth","DiffWalk","Sex","Age","Education","Income"]
TARGET = "Diabetes_binary"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}\nDownload it as explained in README.md.")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing expected columns: " + ", ".join(missing) + "\nUse the file named diabetes_binary_health_indicators_BRFSS2015.csv.")
    data = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    print("\n--- Held-out test results (NOT clinical validation) ---")
    print(f"Rows used: {len(data):,}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.3f}")
    print(f"Positive-class recall/sensitivity: {recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"Positive-class precision: {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"Specificity: {tn/(tn+fp):.3f}" if tn+fp else "Specificity: undefined")
    print(f"ROC AUC: {roc_auc_score(y_test, prob):.3f}")
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_test, pred, labels=[0,1]))
    print("These are internal test-set metrics on US survey data, not proof of clinical performance.")
    joblib.dump(model, MODEL_DIR / "diabetes_risk_model.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")
    print(f"Saved model: {MODEL_DIR / 'diabetes_risk_model.joblib'}")

if __name__ == "__main__":
    main()
