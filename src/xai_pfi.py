#permutation
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"

def main():
    rf = joblib.load("models/rf.joblib")
    features = joblib.load("models/features.joblib")

    df = pd.read_csv(DATA_PATH)
    X = df[features].values
    y = df[LABEL_COL].values.astype(int)


    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    r = permutation_importance(rf,Xte, yte, n_repeats=10, random_state=7, scoring="fi")