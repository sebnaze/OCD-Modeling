# Monitor optimization resource usage
#
# OCD Modeling project
#
# Sebastien Naze
# QIMR 2023

import argparse
from datetime import datetime
import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import psutil
import seaborn as sbn
import time


import OCD_modeling
from OCD_modeling.utils.utils import *

def monitor(args):
    """ Performs the monitoring """ 
    logs = [] 
    t0 = datetime.now()
    fname = os.path.join(proj_dir, 'traces', ".logmonitor{}.json".format(t0.strftime('_%Y%m%d_%H%M')))
    while(True):
        now = datetime.now()
        # exit if reached end time
        if args.time!=None:
            if (now-t0).seconds > args.time:
                break
        
        infos = []
        for proc in psutil.process_iter(['pid', 'username', 'name', 'cpu_percent', 'cpu_times', 'memory_info']):
            if ('python' in proc.info['name']) and ('seb' in proc.info['username']):
                infos.append(proc.info)

        ncpus = sum([info['cpu_percent'] for info in infos])/100
        rss = sum(info['memory_info'].rss for info in infos)/1000000000
        rss_not_shared = sum((info['memory_info'].rss - info['memory_info'].shared) for info in infos)/1000000000

        line = {'time': str(now), 'ncpus':ncpus, 'mem':rss, 'mem_not_shared':rss_not_shared}
        with open(fname, 'a+') as f:
            json.dump(line, f)
            f.write('\n')
        
        if args.verbose:
            print("{}, ncpu={:.2f}, mem={:.2f}Gb, mem(excl. shared)={:.2f}Gb".format(now, ncpus, rss, rss_not_shared))
        
        time.sleep(args.interval)

def plot_monitoring(args):
    """ display traces of monitored activity """
    args
    logs=[]
    with open(os.path.join(proj_dir, f'traces/.logmonitor_{args.log_id}.json'), 'r') as f:
        for line in f.readlines():
            log = json.loads(line)
            log['time'] = datetime.strptime(log['time'], "%Y-%m-%d %H:%M:%S.%f")
            logs.append(log)
    
    plt.figure(figsize=[16,4])
    logs_df = pd.DataFrame(logs)
    sbn.lineplot(data=pd.melt(logs_df, id_vars=['time'], value_vars=['ncpus', 'mem', 'mem_not_shared']), x='time', y='value', hue='variable')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=None, action='store', help='time to monitor (in sec). default: until process is stopped')
    parser.add_argument('--interval', type=float, default=1, action='store', help='sampling period (in seconds). default: every second')
    parser.add_argument('--verbose', default=False, action='store_true', help='show monitor in stdout')
    parser.add_argument('--log_id', default=None, type=str, action='store', help='used to read saved log (format YYYYMMDD_hhmm)')
    args = parser.parse_args()

    if args.log_id!=None:
        plot_monitoring(args)
    else:
        monitor(args)
