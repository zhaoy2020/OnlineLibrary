#!/bmp/backup/zhaosy/miniconda3/bin/python

from joblib import Parallel, delayed
import subprocess
import os
import sys

def mul(cmd:str):
    '''运行'''
    print(cmd)
    print("-"*100)
    results = subprocess.run(cmd,shell=True)
    return (results.args, results.returncode)

def assignment(n_jobs:int=-1, pre_dispatch:int=2, args:list="echo empty"):
    '''组织函数'''
    Parallel(n_jobs=n_jobs, backend='multiprocessing',pre_dispatch=pre_dispatch)(delayed(mul)(cmd) for cmd in args)

def arvs():
    '''接受参数'''
    n_jobs = int(sys.argv[1])
    pre_dispatch = int(sys.argv[2])
    file = sys.argv[3]
    with open(file, 'r') as f:
        cmds = []
        for rec in f.readlines():
            # print(rec.strip())
            cmds.append(rec)
        print("Total commands: ", len(cmds))
        print(f"Submit {pre_dispatch} tasks at a time")
        assignment(n_jobs=n_jobs, pre_dispatch=pre_dispatch, args=cmds)

if __name__ == "__main__":
    print("="*100)
    print("python parallel.py [threads] [pre_dispatch] [batch.txt]")
    print("threads: Default: all threads in your machine.")
    print("pre_dispatch: Submit number of tasks at a time.")
    print("batch.txt: Contains all the commands that need to be executed (line by line).")
    print("="*100)
    arvs()
