import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients

DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]


class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.net(x).squeeze(-1)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=7,stratify=y)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    device = "cpu"
    model = MLP(d=Xtr.shape[1].to(device))
    opt = torch.optim.Adam(model.parameters(),lr = 1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),batch_size=128,shuffle=True)

    model.train()
    for epoch in range(10):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits,yb)
            loss.backward()
            opt.step()


    idx = int(np.where(yte == 1)[0][0])
    x = torch.tensor(Xte[idx:idx+1]).to(device)

    baseline = torch.zeros_like(x)


    model.eval()
    ig = IntegratedGradients(lambda inp: torch.sigmoid(model(inp)))
    attr = ig.attribute(x,baselines=baseline, target=None).detach().cpu().numpy().reshape(-1)

    order = np.argsort(-np.abs(attr))
    print(f"\n[IG] Local explanation (abs attribution) for one anomaly test sample idx={idx}:")
    for j in order[:5]:
        print(f"{FEATURES[j]}: IG={attr[j]:.4f}")

if __name__ == "__main__":
    main()



