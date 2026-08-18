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
 


DATA_PATH = "data/synthetic_network_system.csv"
LABEL_COL = "is_event"
FEATURES = ["latency_ms","throughput_mbps","packet_loss_pct","jitter_ms","cpu_pct","mem_pct","io_ms"]

#Possibly the simplest model, found on the pytorch website
class MLP(nn.Module):
    #very simple with one input layer, one hidden layer and one output layer
    def __init__(self,d):
        super(MLP,self).__init__()
        self.Linear1 = nn.Linear(d,32)
        self.Linear2 = nn.Linear(32,16)
        self.Linear3 = nn.Linear(16,1)
        self.activation1 = nn.Sigmoid()


    def forward(self,x):
        x = self.Linear1(x)
        x = self.activation1(x)
        x = self.Linear2(x)
        x = self.activation1(x)
        x = self.Linear3(x)
        x = self.activation1(x)

        x = x.squeeze(-1) #changes the dimensions to be correct.
        return x


def main():
    #reads the data into a dataframe
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[LABEL_COL].values.astype(int)
    #puts the training set into smaller(75%) random chunks 
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)
    # 1) Logistic Regression (scaled)
    #logistic regression is unlike linear regression since linear can have a lot of values on a linear gradiant
    #logistic takes only between 0-1, which is what we want since we are working with probability

    #Pipleine automates data transformation
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    #fit modifies the parameters of the model based on the data provided, corrects errors and optimizes
    lr.fit(Xtr,ytr)
    #test on the testing set and extract the changes for it being an event or not
    p_lr = lr.predict_proba(Xte)[:,1]
    #Check the AUC score with the yte as the correct options
    print("\n[LR] AUC:",roc_auc_score(yte,p_lr))
    
    # Give a report 
    # set the zero_division to 0.0 so that there are no error messages shown.
    print(classification_report(yte,(p_lr>= 0.5).astype(int),zero_division=0.0))

    #2 Forest Classifier diveds up in small groups and  //random state
    rf = RandomForestClassifier(n_estimators=300, random_state=90, class_weight="balanced")
    rf.fit(Xtr,ytr)
    
    
    p_rf = rf.predict_proba(Xte)[:,1]
    # print("[RF] p_rf[0] ", p_rf[40:50])
    print("\n[RF] AUC:", roc_auc_score(yte,p_rf))
    print(classification_report(yte, (p_rf>=0.5).astype(int)))
    # print(classification_report(yte, (p_rf)))
    
    #save the models for later use in XAI
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

    mlp = model
    joblib.dump(mlp, "models/mlp.joblib")
    joblib.dump(scaler,"models/scaler.joblib")
    model.eval()
    outputMLP = model(torch.tensor(Xte))

    outputMLP = outputMLP.detach().numpy()
    print("\n[MLP] AUC:", roc_auc_score(yte,outputMLP))
    print(classification_report(yte,(outputMLP>=0.5).astype(int),zero_division=0.0))
    
    #######################################
    #MLP SCIKIT
    # from sklearn.neural_network import MLPClassifier
    # Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=7,stratify=y)

    # clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(7,4),random_state=5)

    # clf.fit(Xtr,ytr)
    # p_mlp = clf.predict_proba(Xte)[:,1]
    # print("AUC p_mlp ", roc_auc_score(yte,p_mlp))
    # print(classification_report(yte, (p_mlp>=0.5).astype(int),zero_division=0.0))
    
    # joblib.dump(clf,"models/mlp_s.joblib")
    

if __name__ == "__main__":
    main()