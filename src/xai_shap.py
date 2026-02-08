import joblib
import numpy as np
import pandas as pd
import shap


DATA_PATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"



def main():
    rf = joblib.load("models/rf.joblib")
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATA_PATH)
    
    X = df[features]
    y = df[LABEL_COL].astype(int).values

    background = shap.sample(X,200,random_state=7)

    explainer = shap.TreeExplainer(rf,data=background,feature_names=features)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    
    global_imp = np.mean(np.abs(sv),axis=0)
    order = np.argsort(-global_imp)
    print("\n[SHAP] Global importance (mean |SHAP|):")

    for j in order:
        print(f"{features[j]}: {global_imp[j]:.4f}")
    #huh?
    idx = int(np.where(y == 1)[0][0])
    local = sv[idx]

    order_local = np.argsort(-np.abs(local))
    print(f"\n[SHAP] Local explanation for sample idx={idx}:")
    for j in order_local[:5]:
        print(f"{features[j]}: shap={local[j]:.4f}, value={X.iloc[idx, j]}")


if __name__ == "__main__":
    main()