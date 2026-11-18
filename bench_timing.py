# -*- coding: utf-8 -*-
"""Clean inference-time benchmark for Table 10 (run ALONE on the GPU: close Jupyter, browsers and any other training first).
Usage:  python bench_timing.py        -> writes results_full\\timing_clean.json
Weights do not affect speed, so each architecture is instantiated untrained with the real vocabulary size (420).
Protocol: fixed-length 200-character inputs (worst case), 20,000 URLs, batch 1024, fp32, torch.no_grad(), eval mode,
CUDA synchronised before/after each pass, 3 warm-up passes discarded, 15 timed passes, median and IQR reported.
Tokenisation is excluded. Also reports single-URL latency (batch 1, 200 timed calls, median)."""
import os,sys,json,time,numpy as np,torch
import run_full as R
DEV=R.DEV; V=420; N=20000; B=1024
def sync():
    if DEV.type=="cuda": torch.cuda.synchronize()
def bench(name,passes=15,warm=3,n=N):
    torch.manual_seed(0); m=R.MODELS[name](V).to(DEV).eval()
    xb=torch.randint(1,V,(n,R.MAXLEN),device=DEV); L=torch.full((n,),R.MAXLEN,device=DEV); ts=[]
    with torch.no_grad():
        for r in range(warm+passes):
            sync(); t=time.perf_counter()
            for i in range(0,n,B): m(xb[i:i+B],L[i:i+B])
            sync(); dt=(time.perf_counter()-t)/n*1000*1000
            if r>=warm: ts.append(dt)
        x1=xb[:1]; l1=L[:1]; t1=[]
        for r in range(20): m(x1,l1)
        for r in range(200):
            sync(); t=time.perf_counter(); m(x1,l1); sync(); t1.append((time.perf_counter()-t)*1000)
    q=np.percentile(ts,[25,50,75]); return dict(ms_per_1000_median=float(q[1]),ms_per_1000_q1=float(q[0]),ms_per_1000_q3=float(q[2]),latency_ms_batch1_median=float(np.median(t1)),params=int(sum(p.numel() for p in m.parameters())))
if __name__=="__main__":
    out={"env":dict(torch=torch.__version__,device=str(DEV),gpu=(torch.cuda.get_device_name(0) if DEV.type=="cuda" else None),n_urls=N,batch=B)}
    names=["proposed","char_lstm","urlnet_char","hdpcnn_char","ccbla","bilstm_highway","cnn_only","bilstm_only","cnn_bilstm"]
    for nm in names:
        out[nm]=bench(nm,passes=(2 if R.SMOKE else 15),warm=(1 if R.SMOKE else 3),n=(2048 if R.SMOKE else N)); print(nm,out[nm],flush=True)
    json.dump(out,open(os.path.join(R.OUT,"timing_clean.json"),"w"),indent=1); print("saved",os.path.join(R.OUT,"timing_clean.json"))
