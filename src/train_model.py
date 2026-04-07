import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import torchmetrics.classification
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import tensor
from sklearn.metrics import roc_curve

# from torchmetrics.classification import BinaryAUROC
 


DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]

class MLP(nn.Module):
    def __init__(self,d):
        super(MLP,self).__init__()
        self.Linear1 = nn.Linear(d,32)
        self.Linear2 = nn.Linear(32,16)
        self.Linear3 = nn.Linear(16,1)
        self.activation1 = nn.Sigmoid()
        # self.Softmax = nn.Softmax()

        # self.net = nn.Sequential(
        #     nn.Linear(d,32),nn.ReLU(), #defines the layers
        #     nn.Linear(32,16),nn.ReLU(),
        #     nn.Linear(16,1)
        # )

    def forward(self,x):
        x = self.Linear1(x)
        x = self.activation1(x)
        x = self.Linear2(x)
        x = self.activation1(x)
        x = self.Linear3(x)
        x = self.activation1(x)
        # x = self.Linear4(x)
        # x = self.Softmax(x)
        x = x.squeeze(-1) #changes the dimensions to be correct.
        return x


def main():
    #reads the data into a dataframe
    print("hello")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[LABEL_COL].values.astype(int)
    #puts the training set into smaller(75%) random chunks 
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)
    # 1) Logistic Regression (scaled)
    #logistic regression is unlike linear regression since linear can have a lot of values on a linear gradiant
    #logistic usally takes only two(0-1), but for this it is more than that 

    #Pipleine automates data transformation
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    #fit modifies the parameters of the model based on the data provided, corrects errors and optimizes
    lr.fit(Xtr,ytr)
    p_lr = lr.predict_proba(Xte)[:,1]
    # print(p_lr)
    print("\n[LR] AUC:",roc_auc_score(yte,p_lr))
    # print(classification_report(yte,p_lr))
    # set the zero_division to 0.0 so that there are no error messages shown.
    print(classification_report(yte,(p_lr>= 0.5).astype(int),zero_division=0.0))

    #2 Forest Classifier diveds up in small groups and 
    rf = RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced")
    rf.fit(Xtr,ytr)
    
    
    p_rf = rf.predict_proba(Xte)[:,1]
    # print("[RF] p_rf[0] ", p_rf[40:50])
    print("\n[RF] AUC:", roc_auc_score(yte,p_rf))
    print(classification_report(yte, (p_rf>=0.5).astype(int)))
    # print(classification_report(yte, (p_rf)))

    os.makedirs("models",exist_ok=True)
    joblib.dump(lr, "models/lr.joblib")
    joblib.dump(rf, "models/rf.joblib")
    joblib.dump(FEATURES, "models/features.joblib")
    print("\n[OK] Saved models in models. LR and RF")

    # y_pred = rf.predict(Xte)

    # print(classification_report(y_pred,yte))

    #MLP (our own made method that does not work aswell as the scikit-learn ones)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=7,stratify=y)

    scaler = StandardScaler() #make it to standard scores by making into standarad scores(hoover around zero+-3)
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    # use the cpu to train
    device = "cpu"
    model = MLP(d=Xtr.shape[1]).to(device)
    # optimal
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

    # from sklearn.metrics import classification_report, roc_auc_score
    # outputMLP = model
    # print(model.state_dict)
    # p_mlp = model.
    # print(classification_report(yte,(p_mlp>=0.5).astype(int)))
    joblib.dump(model, "models/mlp.joblib")
    print("\n[OK] Saved models in models. MLP")
    # auc = AUROC()
    # prob = nnf.softmax()
    model.eval()
    outputMLP = model(torch.tensor(Xte))
    print("output of model:")
    # print(outputMLP)
    probs_mlp = nnf.softmax(outputMLP,dim=0)
    probs_mlp1 = nnf.softmax(outputMLP,dim=-1)
    # print(probs_mlp)

    # r_mlp = torch.sigmoid(outputMLP)
    # r_mlp1 = r_mlp >= 0.5
    # print(r_mlp)
    # print(r_mlp1)
    # top_p, top_class = probs_mlp.topk(1,dim= -1)
    # print(top_class)
    # print()
    # print(top_p)
    probs_mlp = probs_mlp.detach().numpy()
    # # print(probs_mlp)
    # r_mlp = r_mlp.detach().numpy()
    outputMLP = outputMLP.detach().numpy()
    print("\n[MLP] AUC:", roc_auc_score(yte,probs_mlp))
    # print("\n[MLP] AUC:", roc_auc_score(yte,r_mlp))

    # fpr,tpr, thresholds = roc_curve(yte,r_mlp)
    # print("mlp")
    # print(fpr)
    # print(tpr)
    # print(thresholds)

    # fpr,tpr, thresholds = roc_curve(yte,p_lr)

    # print("lr")
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    #######################################
    #MLP SCIKIT
    from sklearn.neural_network import MLPClassifier
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)

    clf = MLPClassifier(solver='adam',alpha=1e-5,activation='relu',hidden_layer_sizes=(7,2), random_state= 7)

    clf.fit(Xtr,ytr)
    #we use predict_proba to get the chances, but to use the model we just use predict
    # print(lr.predict_proba(Xte)[:,1])
    # print()
    # print(rf.predict(Xte))
    # print()
    u1 = 700
    u2 = 800
    print(lr.predict(X[u1:u2]))
    print((y[u1:u2]))
    # print(lr.predict(Xte))
    # print()
    # print(yte)
    # print(lr.score(X,y))
    # print(rf.score(X,y))
    # print(clf.score(X,y))
    p_mlp = clf.predict_proba(Xte)[:,1]
    print("AUC p_mlp ", roc_auc_score(yte,p_mlp))
    print(classification_report(yte, (p_mlp>=0.5).astype(int),zero_division=0.0))
    
    joblib.dump(clf,"models/mlp_s.joblib")
    

if __name__ == "__main__":
    main()