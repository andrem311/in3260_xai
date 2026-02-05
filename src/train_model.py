import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"
FEATURES = ["latency_ms", "throughputh_mbps","packet_loss_pct","jitter_ms","cpu_pct", "mem_pct", "io_ms"]


def main():
    #reads the data into a dataframe
    df = pd.read_csv(DATA_PATH)
    #
    X = df[FEATURES].values
    
    y = df[LABEL_COL].values.astype(int)
    #puts the training set into smaller random chunks 
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)

    # 1) Logistic Regression (scaled)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    lr.fit(Xtr,ytr)
    p_lr = lr.predict_proba(Xte)[:,1]
    print("\n[LR] AUC:",roc_auc_score(yte,p_lr))
    print(classification_report(yte,(p_lr>= 0.5).astype(int)))

    #2
    rf = RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced")
    rf.fit(Xtr,ytr)
    p_rf = rf.predict(Xte)[:,1]
    print("\n[RF] AUC:", roc_auc_score(yte,p_rf))
    print(classification_report(yte, (p_rf>=0.5).astype(int)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(lr, "models/lr.joblib")
    joblib.dump(rf, "models/rf.joblib")
    joblib.dump(FEATURES, "models/features.joblib")
    print("\n[OK] Saved models in models...")

    if __name__ == "__main__":
        main()