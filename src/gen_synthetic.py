import os
import numpy as np
import pandas as pd

def set_seed(seed:int = 42):
    np.random.seed(seed)


def daily_pattern(t,period=24*60):
    return 0.5+0.5 * np.sin(2*np.pi*t/period)

def add_event_window(event_mask,start_idx, duration):
    end_idx = min(len(event_mask),start_idx+duration)
    event_mask[start_idx:end_idx] = True
    return start_idx, end_idx

def generate_synthetic(n_minutes=24*60,freq="1min",seed=42,p_event=0.02):
    set_seed(seed)

    #time axis
    ts = pd.date_range("2026-01-01",periods=n_minutes,freq=freq)
    t = np.arange(n_minutes)
    day=daily_pattern(t)

    # Base (normal) metrics with daily pattern + noise
    #You can adjust these to make more/less challenging scenarios
    latency_ms = 20 +25*day + np.random.normal(0,2.0,n_minutes)
    throughput_mbps = 200- 60*day + np.random.normal(0,5.0,n_minutes)
    packet_loss_pct = 0.2 + 0.5*day + np.random.normal(0,0.08,n_minutes)
    jitter_ms = 2 + 4*day + np.random.normal(0,0.4,n_minutes)

    cpu_pct = 20 + 35*day +np.random.normal(0,3.0,n_minutes)
    mem_pct = 35 + 15*day + np.random.normal(0,1.5,n_minutes)
    io_ms = 3 + 6*day + np.random.normal(0,0.6,n_minutes)

    #Clamp reasonable ranges
    packet_loss_pct = np.clip(packet_loss_pct,0.0, 5.0)
    cpu_pct = np.clip(cpu_pct,0.0, 100)
    mem_pct = np.clip(mem_pct,0.0, 100)
    io_ms = np.clip(io_ms,0.0, None)

    #Ground truth labels
    is_event = np.zeros(n_minutes,dtype=bool)
    event_type = np.array(["none"]*n_minutes,dtype=object)

    i = 0
    while i < n_minutes:
        if np.random.rand() < p_event:
            et = np.random.choice(["congestion","cpu","io"], p =[0.45,0.35,0.20])
            duration = int(np.random.randint(10,60)) # a event last between 10 to 59 min 
            severity = float(np.random.uniform(0.8,1.5)) #event intensity
            
            #set the array where we encouter an event to true in the interval
            start, end = add_event_window(is_event,i,duration)
            event_type[start:end] = et

            if et == "congestion":
                latency_ms[start:end]+=severity*np.linspace(15,40,end-start)
                packet_loss_pct[start:end] += severity*np.linspace(0.6,1.8,end-start)
                throughput_mbps[start:end] -= severity*np.linspace(25,70,end-start)
                cpu_pct[start:end] += severity * np.linspace(5,15,end-start)                
                jitter_ms[start:end] += severity * np.linspace(2,6,end-start)                

            elif et == "cpu":
                cpu_pct[start:end] += severity * np.linspace(30,60,end-start)
                latency_ms[start:end]+=severity*np.linspace(10,30,end-start)
                throughput_mbps[start:end] -= severity*np.linspace(5,25,end-start)
                jitter_ms[start:end] += severity*np.linspace(0.8,2.5,end-start)
            #io
            else:
                io_ms[start:end]+=severity*np.linspace(8,25,end-start)
                latency_ms[start:end] += severity*np.linspace(8,28,end-start)
                throughput_mbps[start:end] -= severity*np.linspace(10,40,end-start)
                cpu_pct[start:end] += severity * np.linspace(5,20,end-start)                
            i = end

        else:
            i +=1 

    packet_loss_pct = np.clip(packet_loss_pct,0.0,10.0)
    throughput_mbps = np.clip(throughput_mbps,0.0,None)
    latency_ms = np.clip(latency_ms,1.0,None)
    jitter_ms = np.clip(jitter_ms,0.0,None)
    cpu_pct = np.clip(cpu_pct,0.0,100.0)
    mem_pct = np.clip(mem_pct,0.0,100.0)
    io_ms = np.clip(io_ms,0.0,None)


    df = pd.DataFrame({"timestamp" : ts, "latency_ms":latency_ms,"throughput_mbps":throughput_mbps,"packet_loss_pct":packet_loss_pct,"jitter_ms":jitter_ms,
                       "cpu_pct":cpu_pct,"mem_pct":mem_pct,"io_ms":io_ms,"is_event":is_event.astype(int),"event_type":event_type})
    
    return df



def main():
    out_dir = "data"
    os.makedirs(out_dir,exist_ok=True)

    df = generate_synthetic(n_minutes=24*60, seed=7,p_event=0.01)
    out_path = os.path.join(out_dir,"synthetic_network_system.csv")
    df.to_csv(out_path,index=False)

    print(f"[OK] Saved: {out_path}")
    print(df.head())
    print("\nEvent counts:")
    print(df["event_type"].value_counts())

if __name__ == "__main__":
    main()