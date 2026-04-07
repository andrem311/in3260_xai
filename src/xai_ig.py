import os, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients
from train_model import MLP
# from sklearn
DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]



#works
def main():
    #the model training is moved to 
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=7,stratify=y)

    device = "cpu"
    idx = int(np.where(yte == 1)[0][0])


    #our own made by pythourch
    model = joblib.load("models/mlp.joblib")
    model.eval()
    for i in range(40,50):
        x = torch.tensor(Xte[i:i+1]).to(device)
        print(torch.sigmoid(model(x)))
        
    
    x = torch.tensor(Xte[idx:idx+1]).to(device)
    baseline = torch.zeros_like(x)
    ig = IntegratedGradients(lambda inp : torch.sigmoid(model(inp)))
    attr = ig.attribute(x,baselines=baseline,target=None).detach().cpu().numpy().reshape(-1)
    order = np.argsort(-np.abs(attr))
    print(order)
    print(f"\n[IG] Local explanation (abs attribution) for one anomaly test sample idx={idx}:")
    for j in order[:5]:
        print(f"{FEATURES[j]}: IG={attr[j]:.4f}")

    # model.eval()
    #The one made by scikit learn
    model = joblib.load("models/mlp_s.joblib")
    for i in range(40, 300):
        x = Xte[i:i+1]
        print((model.predict_proba(x)[:,1]))
    
    x = torch.tensor(Xte[idx:idx+1]).to(device)
    # ig = IntegratedGradients(lambda inp: (model.predict(inp).detach().numpy()))
    ig = IntegratedGradients(lambda inp: torch.sigmoid((torch.from_numpy(model.predict_proba(inp.detach().numpy())))))
    # attr = ig.attribute(x,baselines=baseline, target=None).detach().cpu().numpy().reshape(-1)
    attr = ig.attribute(x,baselines=baseline, target=None)

    order = np.argsort(-np.abs(attr))
    print(order)
    print(f"\n[IG] Local explanation (abs attribution) for one anomaly test sample idx={idx}:")
    for j in order[:5]:
        print(f"{FEATURES[j]}: IG={attr[j]:.4f}")







if __name__ == "__main__":
    main()



