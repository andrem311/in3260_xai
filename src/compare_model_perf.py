import os, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train_model import MLP

DATA_PATH = ["data/synthetic_network_system_short.csv","data/synthetic_network_system_hard.csv","data/synthetic_network_system.csv"]
REAL_DATA = ["data/real_data_windows.csv"]
LABEL_COL = "is_event"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]

"""
This program is for testing the performance of the AI models made with sci-kit learn.
 The main purpose was simply to test precision, recall and auc.
But we also needed a place to test the real data that have been collected, to see what the models think of them.  
"""

def main():
    #turn to 1 for testing of synthetic dataset 
    TESTING = 1
    #turn to 1 for testing of real dataset 
    REAL_DATA_TEST = 1
    rf = joblib.load("models/rf.joblib")
    print("[Loaded RF] ")
    lr = joblib.load("models/lr.joblib")
    print("[Loaded LR] ")
    mlp = joblib.load("models/mlp.joblib")
    scaler = joblib.load("models/scaler.joblib")
    print("[Loaded MLP]")




    if(TESTING == 1):
        df = pd.read_csv(DATA_PATH[0])

        X = df[FEATURES].values
        y = df[LABEL_COL].values.astype(int)
        #really no need to split up the data as we are not training, but done for efficeny in testing
        #when there are large data sets
        #nice to only test a small part
        #if the wish is to test
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)
        #set test to something else for to test it
        test = Xte
        #Logistic regression
        p_lr = lr.predict_proba(test)[:,1]
        print("\n[LR] AUC:",roc_auc_score(yte,p_lr))
        # set the zero_division to 0.0 so that there are no error messages shown.
        print(classification_report(yte,(p_lr>= 0.5).astype(int),zero_division=0.0))

        #Random forrest
        p_rf = rf.predict_proba(test)[:,1]

        print("\n[RF] AUC:", roc_auc_score(yte,p_rf))
        print(classification_report(yte, (p_rf>=0.5).astype(int)))

        #MLP
        
        
        test = scaler.transform(test).astype(np.float32)
        outputMLP = mlp(torch.tensor(test))
        outputMLP = outputMLP.detach().numpy()
        print("\n[MLP] AUC:", roc_auc_score(yte,outputMLP))
        print(classification_report(yte,(outputMLP>=0.5).astype(int),zero_division=0.0))
        
    #prints all the probability for 
    if(REAL_DATA_TEST == 1):
        #Testing real data
        print()
        print("Testing real data")
        df_real = pd.read_csv(REAL_DATA[0])
        X = df_real[FEATURES].values
        
        #Logistic regression
        print("logistic regression:")
        p_r_lr = lr.predict_proba(X)[:,1]
        print(p_r_lr)
        #Random forrest
        print("Random forrest:")
        p_r_rf = rf.predict_proba(X)[:,1]
        print(p_r_rf)

        
        X = scaler.transform(X).astype(np.float32)
        outputMLP = mlp(torch.tensor(X))
        outputMLP = outputMLP.detach().numpy()
        print("MLP:")
        print(outputMLP) 









if __name__ == "__main__":
    main()