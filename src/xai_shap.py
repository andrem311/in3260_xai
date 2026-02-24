import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from shap.plots import beeswarm


DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"

def main():
    shap.initjs()
    rf = joblib.load("models/rf.joblib")
    lr = joblib.load("models/lr.joblib")
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    X = df[features].values
    y = df[LABEL_COL].values.astype(int)
    print(features)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    print(rf)

    # pick a small background set for speed
    background = shap.sample(Xtr, 200, random_state=7)

    # explainer = shap.TreeExplainer(rf)

    explainer = shap.TreeExplainer(rf,data=background,feature_names=features)
    # explainer = shap.Explainer(lr,feature_names=features)
    sv = explainer(Xtr)
    exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=Xtr,feature_names=features)
    # shap_values = explainer.shap_values(Xtr)
    print("explainer:")
    print(exp[0,0])
    print(type(exp))
    print(exp[0])
    # np.shape(shap_values)
    # beeswarm(shap_values)
    

    # For binary classification, shap_values can be a list [class0, class1]
    # if isinstance(shap_values, list):
    #     sv = shap_values[1]   # explain class=1 (anomaly)
    # else:
    #     sv = shap_values

    # print("Xtr again")
    # print(Xtr)
    # print("Shap values")
    # is a tripple dime
    # print(shap_values)
    shap.summary_plot(exp)
    shap.waterfall_plot(exp[0])
    # global_imp = np.mean(np.abs(sv), axis=0)
    # order = np.argsort(-global_imp)
    # print(order)
    # print(sv)
    # print("\n[SHAP] Global importance (mean |SHAP|):")



    # for j in order:
        # print(f"{features[j]}: {global_imp[j]:.4f}")

    # Local explanation: pick one anomaly row
    # idx = int(np.where(y == 1)[0][0])
    # local = sv[idx]
    # order_local = np.argsort(-np.abs(local))
    # print(f"\n[SHAP] Local explanation for sample idx={idx}:")
    # for j in order_local[:5]:
        # print(f"{features[j]}: shap={local[j]:.4f}, value={X.iloc[idx, j]}")

if __name__ == "__main__":
    main()