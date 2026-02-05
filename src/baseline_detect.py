import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    "latency_ms", "throughput_mbps", "packet_loss_pct", "jitter_ms", "cpu_pct", "mem_pct", "io_ms"
]

# rolling z score, 
def rolling_zscore(x: pd.Series, win: int = 60):
    #Rolling takes window(win) and does an aggregate function over it
    mu = x.rolling(win,min_periods = max(5, win//4)).mean()
    #standard devietion looks at the difference from the mean, if number is bigger
    #then the values are a lot pretty diffrent from the mean
    sd = x.rolling(win, min_periods = max(5,win//4)).std()
    z = (x - mu) / (sd+ 1e-8)
    return z

def detect_events(df: pd.DataFrame, win: int = 60, z_thr: float = 3.0, min_metrics: int = 2):
    """
    Rolling z-score detector: 
    - compute z-score per metric
    - mark time t as anomalous 
    """
    out = df.copy()
    Z = {}
    for m in METRICS:
        z = rolling_zscore(out[m],win = win)

        Z[m] = z;

    Zdf = pd.DataFrame(Z)
    out = pd.concat([out, Zdf.add_prefix("z_")], axis=1)

    # abnormality incdicator per metric
    flags = (Zdf.abs() >= z_thr).astype(int)
    out["num_flagged_metrics"] = flags.sum(axis=1)

    out["score_sum"] = Zdf.abs().sum(axis=1)

    out["is_anom"] = (out["num_flagged_metrics"] >= min_metrics).astype(int)

    return out


def group_intervals(mask: np.ndarray):

    intervals = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j+=1
            intervals.append((i,j))
            i = j
        else:
            i += 1
    return intervals

def explain_interval(df: pd.DataFrame, start: int, end: int, topk: int = 2):
    scores = {}
    for m in METRICS:
        zcol = "z_" + m
        scores[m] = float(df.loc[start:end-1,zcol].abs().mean())
    ranked = sorted(scores.items(), key = lambda kv: kv[1], reverse = True)
    return ranked[:topk], ranked

def make_sentence(top_metrics):
    names = [m for m, _ in top_metrics]
    pretty = {
        "latency_ms" : "latency" ,
        "throughput_mbps" : "throughput" ,
        "packet_loss_pct" : "packet loss" ,
        "jitter_ms" : "jitter" ,
        "cpu_pct" : "CPU usage" ,
        "mem_pct" : "memory usage" ,
        "io_ms" : "I/O delay"
    }
    names = [pretty.get(n,n) for n in names]
    if len(names) == 1:
        return f"Main driver: {names[0]}."
    return f"Main driver: {names[0]} and {names[1]}."

def plot_signals(df: pd.DataFrame, out_dir = "figs"):
    os.makedirs(out_dir,exist_ok=True)

    t = pd.to_datetime(df["timestamp"])
    to_plot = ["latency_ms","packet_loss_pct", "cpu_pct", "throughput_mbps"]

    for m in to_plot:
        plt.figure()
        plt.plot(t,df[m].values,label=m)

        #ground truth shadin
        gt = df["is_event"].values.astype(bool)
        intervals = group_intervals(gt)
        for(s,e) in intervals:
            plt.axvspan(t.iloc[s],t.iloc[e-1],alpha=0.15)

        an = df["is_anom"].values.astype(bool)
        idx = np.where(an)[0]
        if len(idx) > 0:
            plt.scatter(t.iloc[idx],df[m].iloc[idx],s=10, label="detected")
        
        plt.title(f"{m} (shaded= ground truth events, dots = detected)")
        plt.xlabel("time")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir,f"{m}.png")
        plt.savefig(path,dpi = 160)
        plt.close()

def main():
    in_path = os.path.join("data","synthetic_network_system.csv")
    df = pd.read_csv(in_path)

    df2 = detect_events(df,win = 60, z_thr = 3.0, min_metrics=2)

    intervals = group_intervals(df2["is_anom"].values.astype(bool))
    print(f" [INFO] Detected {len(intervals)} anomalous intervals \n")

    for k, (s,e) in enumerate(intervals[:10], start=1):
        top2, ranked = explain_interval(df2,s,e,topk=2)
        ts0 = df2.loc[s,"timestamp"]
        ts1 = df2.loc[e-1 , "timestamp"]
        print(f"Event #{k}: {ts0} -> {ts1} (len={e-s} min)")
        print(" " + make_sentence(top2))
        #print numeric contributions
        for m, sc in top2:
            print(f"  - {m}: mean|z| = {sc:.2f}")
        print("")

    plot_signals(df2,out_dir= "figs")

    out_path = os.path.join("data", "synthetic_with_detections.csv")
    df2.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")
    print("[OK] Saved plots in figs/")


if __name__ == "__main__":
    main()