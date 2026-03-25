import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

DATAPATH = "data/synthetic_with_detections.csv"
LABEL_COL = "is_anom"
#works
def main():
    # rf = joblib.load("models/rf.joblib")
    # we look at the logistic regression instead since this xai does better with differentiable models
    # rather than non-differentiable ones. 
    lr = joblib.load("models/lr.joblib") 
    features = joblib.load("models/features.joblib")
    df = pd.read_csv(DATAPATH)

    X = df[features].values
    y = df[LABEL_COL].values.astype(int)

    explainer = LimeTabularExplainer(training_data=X,
                                     feature_names=features,class_names=["normal","anomaly"],
                                     mode="classification",discretize_continuous=True)
    
    idx = int(np.where(y==1)[0][0])
    # exp = explainer.explain_instance(data_row=X[idx], predict_fn=rf.predict_proba, num_features=7)
    exp = explainer.explain_instance(data_row=X[idx], predict_fn=lr.predict_proba, num_features=7)
    print(exp.as_list())
    print(f"\n[LIME] Explanation for sample idx={idx}:")
    for feat, w in exp.as_list():
        print(f"{feat}: weight={w:.4f}")


if __name__ == "__main__":
    main()



