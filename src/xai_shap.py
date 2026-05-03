import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from shap.plots import beeswarm


DATA_PATH = "data/synthetic_with_detections_short.csv"
LABEL_COL = "is_anom"

def main():
    shap.initjs()
    rf = joblib.load("models/rf.joblib")
    lr = joblib.load("models/lr.joblib")
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    X = df[features].values
    # to use iloc further down
    X1 = df[features]
    y = df[LABEL_COL].values.astype(int)
    print(features)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    print(X1)

    # pick a small background set for speed
    # background = shap.sample(Xtr, 200, random_state=7)
    background = shap.sample(X, 200, random_state=7)

    # explainer = shap.TreeExplainer(rf)

    # explainer = shap.TreeExplainer(rf,data=background,feature_names=features)
    explainer = shap.TreeExplainer(rf,data=background,feature_names=features)
    # explainer = shap.Explainer(lr,feature_names=features)
    # sv = explainer(Xtr)
    sv = explainer(X)
    print("SV:")
    print(sv.display_data)

    # exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=Xtr,feature_names=features)
    exp = shap.Explanation(sv.values[:,:,1], sv.base_values[:,1],data=X,feature_names=features)
    
    print(exp[0])
    np.save("outputs/shap_scores.npy",exp.values)
    # shap_values = explainer.shap_values(Xtr)
    # print("explainer:")
    # print(exp[0,0])
    # print(type(exp))
    # print(len(exp))
    # print(exp[0])
    print("EXP mean:")
    print(sv.mean(0)) # global vaules (or just some arbitrary value)
    sv_mean = sv.mean(0)
    
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
    
    mean_from_exp = exp.mean(0).abs.values.tolist()
    #not a np array and need to do some list manipulation
    order_pre = reversed(np.argsort(mean_from_exp))
    # order_pre = reversed(order_pre)

    # shap.waterfall_plot(exp[40])
    # global_imp = np.mean(np.abs(sv.data), axis=0)
    # order = np.argsort(-global_imp)
    
    print("\n[SHAP] Global importance (mean |SHAP|):")
    for j in order_pre:
        print(f"{features[j]}: {mean_from_exp[j]:.4f}")

    # Local explanation: pick one anomaly row
    idx = int(np.where(y == 1)[0][0])
    local = exp[idx].values.tolist()
    order_local = np.argsort(-np.abs(local))
    print(f"\n[SHAP] Local explanation for sample idx={idx}:")
    for j in order_local[:7]:
        print(f"{features[j]}: shap={local[j]:.4f}, value={X1.iloc[idx, j]}")

if __name__ == "__main__":
    main()