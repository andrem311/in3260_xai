import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from shap.plots import beeswarm
import sklearn

# dataset with noise 
DATA_PATH = "data/synthetic_network_system_hard.csv"
#the dataset that it trained on
DATA_PATH_TR = "data/synthetic_network_system_short.csv"
#The dataset collected from
DATA_PATH_REAL = "data/real_data_windows.csv"
LABEL_COL = "is_event"
#A zero, zero would be running random forrest with synthethic dataset.
LR = 0
REAL = 1

def main():
    shap.initjs()
    rf = joblib.load("models/rf.joblib")
    lr = joblib.load("models/lr.joblib")
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATA_PATH)
    y = df[LABEL_COL].values
    if(REAL == 1):
        df = pd.read_csv(DATA_PATH_REAL)
        y = []
    df_tr = pd.read_csv(DATA_PATH_TR)
    # print(df.head())
    X_TR = df_tr[features].values
    y_TR = df_tr[LABEL_COL].values
    X = df[features].values
    
    X1 = df[features]

    if LR == 0:
        #SHap for rf
        print("Running SHAP for rf")
        explainer = shap.TreeExplainer(rf,data=X_TR,feature_names=features)
        sv = explainer(X)
        exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=X,feature_names=features)
        # exp = shap.Explanation(sv.values, sv.base_values,data=X,feature_names=features)

        np.save("outputs/shap_scores.npy",exp.values)
        print(exp.values)
        #comes out as an explanation object
        mean_from_exp = exp.mean(0).abs.values.tolist()
        #not a np array and need to do some list manipulation
        order_pre = reversed(np.argsort(mean_from_exp))

        print("\n[SHAP] Global importance (mean |SHAP|):")
        for j in order_pre:
            print(f"{features[j]}: {mean_from_exp[j]:.4f}")

        # Local explanation: pick one anomaly row
        idx = 5

        if len(y) != 0:
            idx = int(np.where(y == 1)[0][0])
        shap.waterfall_plot(exp[idx])

        local = exp[idx].values.tolist()
        order_local = np.argsort(-np.abs(local))
        print(f"\n[SHAP] Local explanation for sample idx={idx}:")
        for j in order_local[:7]:
            print(f"{features[j]}: shap={local[j]:.4f}, value={X1.iloc[idx, j]}")


    elif LR== 1:
        print("Running SHAP for lr")
        
        scaler = sklearn.preprocessing.StandardScaler().set_output(transform="pandas")
        X_std = scaler.fit_transform(X)
        # explainer = shap.TreeExplainer(rf,data=background,feature_names=features)
        #use the data the model was trained on to be the mask
        mask = shap.maskers.Independent(data=X_TR)
        # print("mask:")
        # print(maks.data)
        explainer = shap.LinearExplainer(lr.named_steps["clf"],masker=mask)
        # explainer = shap.Explainer(lr)
        # sv = explainer(Xtr)
        sv = explainer(X)
        print("SV:")
        print(sv.values)
        print("base values")
        print(sv.base_values)
        # print("second")
        # print(sv.base_values)
        # shap.plots.beeswarm(sv)
        # exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=Xtr,feature_names=features)
        # exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=X,feature_names=features)
        exp = shap.Explanation(sv.values,sv.base_values,feature_names=features)
        
        print(exp[0])
        np.save("outputs/shap_scores.npy",exp.values)
        # shap_values = explainer.shap_values(Xtr)
        # print("explainer:")
        # print(exp[0,0])
        # print(type(exp))
        # print(len(exp))
        # print(exp[0])
        # print("EXP mean:")
        # print(sv.mean(0)) # global vaules (or just some arbitrary value)
        # sv_mean = sv.mean(0)
        
        # np.shape(shap_values)
        # beeswarm(shap_values)
        

        # For binary classification, shap_values can be a list [class0, class1]
        # if isinstance(shap_values, list):
        #     sv = shap_values[1]   # explain class=1 (anomaly)
        # else:
        #     sv = shap_values

        # shap.summary_plot(exp)
        # print("EXP MEAN: abs")
        # print(exp.mean(0).abs)
        # beeswarm(exp)
        mean_from_exp = exp.mean(0).abs.values.tolist()
        #not a np array and need to do some list manipulation
        order_pre = reversed(np.argsort(mean_from_exp))
        # order_pre = reversed(order_pre)

        # shap.waterfall_plot(exp[40])
        print("\n[SHAP] Global importance (mean |SHAP|):")
        for j in order_pre:
            print(f"{features[j]}: {mean_from_exp[j]:.4f}")

        # Local explanation: pick one anomaly row
        # idx = int(np.where(y == 1)[0][0])
        idx = 5

        if len(y) != 0:
            idx = int(np.where(y == 1)[0][0])
            
        shap.waterfall_plot(exp[idx])
        print(X[idx])
        print(lr.predict_proba(X)[idx])
        local = exp[idx].values.tolist()
        order_local = np.argsort(-np.abs(local))
        print(f"\n[SHAP] Local explanation for sample idx={idx}:")
        for j in order_local[:7]:
            print(f"{features[j]}: shap={local[j]:.4f}, value={X1.iloc[idx, j]}")

if __name__ == "__main__":
    main()