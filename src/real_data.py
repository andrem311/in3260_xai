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
    # fd_p = os.open(output,os.O_RDWR)
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    # ping 5 times with 0.2 sec delay, any shorter delay requires sudo access. 
    command = ['ping','-i 0.2' ,param, amount, host]
    # we write to a file 
    # os.close(fd_p)
    return subprocess.call(command,stdout=output) == 0
     
# alle features: latency, throughput, I/O latency, 
#trenger I/O util
#resumes when reached 
# net_io = psutil.net_io_counters()
# bytes_sent = 
ts = []
latency_ms = []
througput_mbps = []
packet_loss_pct =[]
io_ms = []
cpu_pct = []
mem_pct =[]
jitter_ms =[]


for i in range(1):

    time_all = time.time()
    fd = os.open("/home/andreas/code/python/in3260/in3260-xai-network/src/real_data_ping.txt", os.O_RDWR)
    amount = 50
    ping("google.com",fd,str(amount))


    file = open("/home/andreas/code/python/in3260/in3260-xai-network/src/real_data_ping.txt",'r')

    time1 = time.time()
    #readlines takes the whole file, and divides into lines
    Lines = file.readlines()

    # os.read(fd, 50)
    time2 = time.time()
    total = time2-time1
    # print(total*1000) #in miliseconds 
    io_ms.append(total)
    # os.close(fd)
    
    # fd = os.open("/home/andreas/code/python/in3260/in3260-xai-network/src/real_data_ping.txt", os.O_RDWR)
        
    # calculate the jitter from N packets
    #based on this: https://obkio.com/blog/how-to-measure-jitter/
    # count = 0
    arr_delay = []
    for line in Lines:
        # if(count == 0):
        #     count += 1
        #     continue
        arr = line.split()
        # print(arr[7])
        # count += 1
        if(len(arr) > 8 and arr[7].startswith("time")):
            delay = arr[7].split("=")[1]
            arr_delay.append(float(delay))

    len_ping = len(arr_delay)
    avg_delay = np.mean(arr_delay)
    # print("avg delay",avg_delay)
    jitter = 0
    # print(2**2)
    for i in range(len_ping):
        jitter += (arr_delay[i] - avg_delay)**2
    
    jitter = jitter/len_ping
    jitter = math.sqrt(jitter)
    # print("jitter: ",jitter)
    
    pct_loss = len_ping /amount
    # print("pct_loss=",pct_loss)

    
    latency_ms.append(avg_delay)
    packet_loss_pct.append(1.0-pct_loss)
    jitter_ms.append(jitter)
    #latency
    #throughput
    isp = speedtest.Speedtest(timeout=3)
    srv = isp.get_best_server()
    logfile    = "/var/log/speedtest.log"

    downstream = "{0:,.2f} Mb".format(float(isp.download()/10**6))
    upstream   = "{0:,.2f} Mb".format(float(isp.upload()/10**6))
    latency    = "{0:,.0f} ms".format(float(srv['latency']))
    update     =  datetime.strftime(datetime.now(),'%d/%m/%y %I:%M%p')
    
    # with open(logfile, 'a+') as file:
    #     file.write(F"{update},{downstream},{upstream},{latency}\n")
    # print (f"{logfile} updated")
    througput_mbps.append(float(isp.download()/10**6))
    #should use downstream for everyday users
    # and should use upstream for servers
    # print(upstream)
    # print(downstream)
    # print(latency)
    # loss = isp.results
    # print()
    # print(loss)

    #packet_loss_pct
    #mest sannsynlig:
    # print(net_io)
    #jitter_ms

    #cpu_pct
    cpu_pct.append(psutil.cpu_percent())

    #mem_pct
    mem = psutil.virtual_memory().percent
    mem_pct.append(mem)
    # print()
    # print(mem)
    #sleep for x seconds
    os.close(fd)
    time_all_end = time.time()
    print(time_all_end - time_all)
    time.sleep(0.1)
    ts.append(datetime.now())

df = pd.DataFrame({"timestamp" : ts, "latency_ms":latency_ms,"throughput_mbps":througput_mbps,"packet_loss_pct":packet_loss_pct,"jitter_ms":jitter_ms,
                       "cpu_pct":cpu_pct,"mem_pct":mem_pct,"io_ms":io_ms})

df.to_csv("/home/andreas/code/python/in3260/in3260-xai-network/data/real_network_system.csv",index=False)

    


