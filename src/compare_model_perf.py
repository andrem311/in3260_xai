import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = ["data/synthetic_network_system_short.csv","data/synthetic_network_system_hard.csv","data/synthetic_network_system.csv"]
REAL_DATA = ["data/real_network_system.csv"]
LABEL_COL = "is_event"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]


def main():
    TESTING = 0
    REAL_DATA_TEST = 1
    mlp_sci = joblib.load("models/mlp_s.joblib")
    # mlp_own = joblib.load("models/mlp.joblib")
    rf = joblib.load("models/rf.joblib")
    lr = joblib.load("models/lr.joblib")

    if(TESTING == 1):
        df = pd.read_csv(DATA_PATH[0])

        X = df[FEATURES].values
        y = df[LABEL_COL].values.astype(int)

        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)
        #Logistic regression
        p_lr = lr.predict_proba(Xte)[:,1]
        print("\n[LR] AUC:",roc_auc_score(yte,p_lr))
        # set the zero_division to 0.0 so that there are no error messages shown.
        print(classification_report(yte,(p_lr>= 0.5).astype(int),zero_division=0.0))

        #Random forrest
        p_rf = rf.predict_proba(Xte)[:,1]

        print("\n[RF] AUC:", roc_auc_score(yte,p_rf))
        print(classification_report(yte, (p_rf>=0.5).astype(int)))

        #MLP
        p_mlp = mlp_sci.predict_proba(Xte)[:,1]
        print("AUC p_mlp ", roc_auc_score(yte,p_mlp))
        print(classification_report(yte, (p_mlp>=0.5).astype(int),zero_division=0.0))
    if(REAL_DATA_TEST == 1):
        #Testing real data
        df_real = pd.read_csv(REAL_DATA[0])
        X = df_real[FEATURES].values
        
        #Logistic regression
        p_r_lr = lr.predict_proba(X)[:,1]
        print(p_r_lr)
        #Random forrest
        p_r_rf = rf.predict_proba(X)[:,1]
        print(p_r_rf)
        #MLP
        p_r_mlp = mlp_sci.predict_proba(X)[:,1]
        print(p_r_mlp)



if __name__ == "__main__":
    main()