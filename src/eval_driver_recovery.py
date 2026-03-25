import numpy as np
import pandas as pd
import joblib

EXPECTED_DRIVERS = {
    "congestion": {"latency", "packet_loss", "throughput"},
    "cpu": {"cpu", "latency"},
    "io": {"io", "latency", "throughput"}
}

def topk_features(scores, feature_names, k=5):
    idx = np.argsort(-np.abs(scores))[:k]
    return {feature_names[j] for j in idx}

def driver_recovery_at_k(topk_set, expected_set):
    return len(topk_set.intersection(expected_set))/ max(1,len(expected_set))


def evaluate_driver_recovery(df, feature_names, method_to_scores, k=5):
    results = []
    # print(type(method_to_scores.items()))
    # print(method_to_scores.items())
    for method, scores_mat in method_to_scores.items():
        print("Scores_mat: ")
        # print(type(scores_mat))
        # print(scores_mat.shape[0])
        print(scores_mat.shape)
        # print("method: ")
        # print(type(method))
        # print(method)
        assert scores_mat.shape[0] == len(df), f"{method}: N mismatch"
        assert scores_mat.shape[1] == len(feature_names), f"{method}: d mismatch"

        for i in range(len(df)):
            e = df.loc[i, "event_type"]
            if e not in EXPECTED_DRIVERS:
                continue

            excepted = EXPECTED_DRIVERS[e]
            topk = topk_features(scores_mat[i],feature_names=feature_names, k=k)
            dr = driver_recovery_at_k(topk, excepted)

            results.append({"method": method, "event_type": e, f"DR@{k}":dr})

    res_df = pd.DataFrame(results)
    table = (
        res_df.groupby(["event_type","method"])[f"DR@{k}"].mean().unstack("method")

    )
    return res_df, table


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_with_detections.csv")
    # feature_names = list(np.load("models/features.npy",allow_pickle=True))
    feature_names = joblib.load("models/features.joblib")

    shap_scores = np.load("outputs/shap_scores.npy")
    # lime_scores = np.load("outputs/lime_scores.npy")
    pfi_scores = np.load("outputs/pfi_scores.npy")


    methods_to_score = {
        "SHAP": shap_scores,
        # "PFI": pfi_scores
    }

    _,table = evaluate_driver_recovery(df,feature_names=feature_names,method_to_scores=methods_to_score,k=5)

    print("\nDriver Recovery Table (mean DR@5): \n ")
    print(table.round(3))
    table.to_csv("outputs/driver_recovery_table.csv")
    print("\nSaved: outputs/driver_recovery_table.csv")

