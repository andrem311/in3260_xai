import numpy as np
import pandas as pd
import joblib

EXPECTED_DRIVERS = {
    "congestion": {"latency_ms", "packet_loss_pct", "throughput_mbps"},
    "cpu": {"cpu_pct", "latency_ms"},
    "io": {"io_ms", "latency_ms", "throughput_mbps"}
}

#does a sort of the best features
def topk_features(scores, feature_names, k=5):
    idx = np.argsort(-np.abs(scores))[:k]

    return {feature_names[j] for j in idx}

# a simple intersection chekc to see what is in the topk variables
def driver_recovery_at_k(topk_set, expected_set):
    return len(topk_set.intersection(expected_set))/ max(1,len(expected_set))


def evaluate_driver_recovery(df, feature_names, method_to_scores, k=5):
    results = []
    # 
    for method, scores_mat in method_to_scores.items():
        print("Scores_mat: ")
        print(type(scores_mat))
        # print(scores_mat.shape[0])
        print(scores_mat.shape)
        
        # print("method: ")
        print(type(method))
        print(method)
        print()
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
    df = pd.read_csv("data/synthetic_network_system_short.csv")
    feature_names = joblib.load("models/features.joblib")
    print(feature_names)

    shap_scores = np.load("outputs/shap_scores.npy")
    lime_scores = np.load("outputs/lime_scores.npy")
    ig_scores = np.load("outputs/ig_scores.npy")
    # pfi_scores = np.load("outputs/pfi_scores.npy")


    methods_to_score = {
        "LIME": lime_scores,
        "SHAP": shap_scores,
        "IG":ig_scores,
        # "PFI": pfi_scores
    }

    res,table = evaluate_driver_recovery(df,feature_names=feature_names,method_to_scores=methods_to_score,k=5)
    
    print("\nDriver Recovery Table (mean DR@5): \n ")
    print(table.round(3))
    table.to_csv("outputs/driver_recovery_table.csv")
    print("\nSaved: outputs/driver_recovery_table.csv")

