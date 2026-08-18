#psutil
import psutil
import pandas as pd
import numpy as np
from pythonping import ping
import os
import math
import time
import speedtest
from datetime import datetime
import platform    # For getting the operating system name
import subprocess  # For executing a shell command
#TO RUN PROGRAM
# sudo ./.venv/bin/python ./src/real_data.py

# gotten from https://stackoverflow.com/questions/2953462/pinging-servers-in-python
def ping(host,output,amount):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """

    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    # ping x times with 0.2 sec delay, any shorter delay requires sudo access. 
    command = ['ping','-i 0.2' ,param, amount, host]
    # we write to a file 
    
    return subprocess.call(command,stdout=output) == 0

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024     

ts = []
latency_ms = []
througput_mbps = []
packet_loss_pct =[]
io_ms = []
cpu_pct = []
mem_pct =[]
jitter_ms =[]

SPEEDTEST = 0

#for 20 iterations
for i in range(20):

    time_all = time.time()
    fd = os.open("./src/real_data_ping.txt", os.O_RDWR)
    #how many pings
    amount = 50
    ping("google.com",fd,str(amount))


    file = open("./src/real_data_ping.txt",'r')

    time1 = time.time()
    #readlines takes the whole file, and divides into lines
    Lines = file.readlines()

    time2 = time.time()
    total = time2-time1
    io_ms.append(total)
   
    arr_delay = []
    for line in Lines:
        
        arr = line.split()
        
        if(len(arr) > 8 and arr[7].startswith("time")):
            delay = arr[7].split("=")[1]
            arr_delay.append(float(delay))

    len_ping = len(arr_delay)
    avg_delay = np.mean(arr_delay)
    jitter = 0
    for i in range(len_ping):
        jitter += (arr_delay[i] - avg_delay)**2
    
    jitter = jitter/len_ping
    jitter = math.sqrt(jitter)
    
    pct_loss = len_ping /amount

    
    latency_ms.append(avg_delay)
    packet_loss_pct.append(1.0-pct_loss)
    jitter_ms.append(jitter)
    #latency
    #throughput
    if(SPEEDTEST == 1):
        isp = speedtest.Speedtest(timeout=3)
        srv = isp.get_best_server()
        

        downstream = "{0:,.2f} Mb".format(float(isp.download()/10**6))
        
        # upstream   = "{0:,.2f} Mb".format(float(isp.upload()/10**6))
        # latency    = "{0:,.0f} ms".format(float(srv['latency']))
        update     =  datetime.strftime(datetime.now(),'%d/%m/%y %I:%M%p')
        
        througput_mbps.append(float(isp.download()/10**6))
        #should use downstream for everyday users
        # and should use upstream for servers
    #if the stress test are used to heavily it can give a forbidden access error and we need to change to plan b
    else:
        io = psutil.net_io_counters()
        bytes_sent, bytes_recv = io.bytes_sent, io.bytes_recv
        time.sleep(1)
        io_2 = psutil.net_io_counters()
        bytes_sent2, bytes_recv2 = io_2.bytes_sent, io_2.bytes_recv
        total_bytes_recv = bytes_recv2-bytes_recv
        total_bytes_recv = float(total_bytes_recv)
        total_bytes_recv *= 8  #for bits
        total_bytes_recv = total_bytes_recv/(1024*1024) # to achive meaga   
        througput_mbps.append(total_bytes_recv)

        


    #cpu_pct
    cpu_pct.append(psutil.cpu_percent())

    #mem_pct
    mem = psutil.virtual_memory().percent
    mem_pct.append(mem)
    
    os.close(fd)
    time_all_end = time.time()
    #time for one iteration 
    print(time_all_end - time_all)

    #sleep for x seconds
    time.sleep(0.1)
    ts.append(datetime.now())

df = pd.DataFrame({"timestamp" : ts, "latency_ms":latency_ms,"throughput_mbps":througput_mbps,"packet_loss_pct":packet_loss_pct,"jitter_ms":jitter_ms,
                       "cpu_pct":cpu_pct,"mem_pct":mem_pct,"io_ms":io_ms})

df.to_csv("./data/real_network_system.csv",index=False,mode='a+',header=False)

    


