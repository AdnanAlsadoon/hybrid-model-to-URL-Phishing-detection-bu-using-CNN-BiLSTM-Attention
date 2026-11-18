# -*- coding: utf-8 -*-
"""
Full-scale experiment suite for the phishing-URL paper (single source of truth for Tables 2-10).
Run once:   python run_full.py        (resumable: finished jobs are skipped)
Data and results folder: BASE below (D:\\systems\\first Objective); results are written to BASE\\results_full.
Requires: torch, numpy, pandas, scikit-learn, tldextract   (pip install tldextract)
"""
import os, sys, platform
HERE=os.path.dirname(os.path.abspath(__file__))
BASE=r"D:\systems\first Objective"          # data AND results folder (results go to BASE\results_full)
def _first(*c): return next((p for p in c if os.path.exists(p)),None)
DATA=_first(os.path.join(BASE,"phishing_dataset_final.csv"),os.path.join(HERE,"phishing_dataset_final.csv"))
PT=_first(os.path.join(BASE,"PhishTank.csv"),os.path.join(BASE,"phishing DataSet","PhishTank.csv"),os.path.join(HERE,"phishing DataSet","PhishTank.csv"),os.path.join(HERE,"PhishTank.csv"))
if DATA is None: sys.exit("phishing_dataset_final.csv not found in "+BASE+" or "+HERE)
print("DATA:",DATA); print("PhishTank:",PT)
OUT=os.path.join(BASE,"results_full"); os.makedirs(OUT,exist_ok=True)
SMOKE=os.environ.get("SMOKE")=="1"            # tiny end-to-end test
EPOCHS=1 if SMOKE else 10
BATCH=256
PATIENCE=3
SEEDS_MAIN=[42] if SMOKE else [42,123,7]       # seeds for the proposed model and the five baselines (Table 9 / Table 3)
NTR=3000 if SMOKE else None                    # None = full training partition
import re, time, math, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
MAXLEN=200
def load():
    df=pd.read_csv(DATA); df["url"]=df["url"].astype(str).str.strip(); df["label"]=df["label"].astype(int)
    return df
def canon_split(df,seed=42):
    idx=np.arange(len(df)); y=df.label.values
    tv,te=train_test_split(idx,test_size=0.1,random_state=seed,stratify=y)
    tr,va=train_test_split(tv,test_size=0.1/0.9,random_state=seed,stratify=y[tv])
    return tr,va,te
def build_vocab(urls):
    ch=set()
    for u in urls: ch.update(u[:MAXLEN])
    v={"<PAD>":0}
    for i,c in enumerate(sorted(ch),1): v[c]=i
    return v
def encode(urls,vocab):
    X=np.zeros((len(urls),MAXLEN),dtype=np.int16); L=np.zeros(len(urls),dtype=np.int16)
    for i,u in enumerate(urls):
        ids=[vocab.get(c,0) for c in u[:MAXLEN]]; X[i,:len(ids)]=ids; L[i]=max(len(ids),8)
    return X,L
def batches(X,L,y,bs,shuffle,rng=None):
    n=len(L); order=np.argsort(L,kind="stable")
    bl=[order[i:i+bs] for i in range(0,n,bs)]
    if shuffle: rng.shuffle(bl)
    for b in bl:
        m=int(L[b].max()); yield torch.from_numpy(X[b,:m].astype(np.int64)), torch.from_numpy(L[b].astype(np.int64)), (torch.from_numpy(y[b]) if y is not None else None), b
def mask_of(L,m): return torch.arange(m,device=L.device)[None,:]<L[:,None]
class Attn(nn.Module):
    def __init__(s,h): super().__init__(); s.a=nn.Linear(h,1)
    def forward(s,x,mask):
        sc=s.a(x).squeeze(-1).float().masked_fill(~mask,-1e9); w=F.softmax(sc,1).to(x.dtype); return torch.bmm(w.unsqueeze(1),x).squeeze(1)
def head(inp,fh=128,dr=0.3): return nn.Sequential(nn.Linear(inp,fh),nn.LayerNorm(fh),nn.ReLU(),nn.Dropout(dr),nn.Linear(fh,fh//2),nn.ReLU(),nn.Dropout(dr/2),nn.Linear(fh//2,2))
def mean_pool(x,mask): m=mask.unsqueeze(-1).float(); return (x*m).sum(1)/m.sum(1)
def max_pool(x,mask): return x.masked_fill(~mask.unsqueeze(-1),(-1e9 if x.dtype==torch.float32 else torch.finfo(x.dtype).min)).max(1)[0]
class Proposed(nn.Module):
    """Same layers as the paper's PhishingDetector; use_cnn/use_lstm/use_att switch the ablation variants."""
    def __init__(s,V,use_cnn=True,use_lstm=True,use_att=True,E=64,Fc=128,K=(3,5,7),H=128,dr=0.3):
        super().__init__(); s.use_cnn,s.use_lstm,s.use_att=use_cnn,use_lstm,use_att
        s.emb=nn.Embedding(V,E,padding_idx=0)
        if use_cnn:
            s.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(E,Fc,k,padding=k//2),nn.BatchNorm1d(Fc),nn.ReLU()) for k in K]); s.proj=nn.Sequential(nn.Linear(Fc*len(K),H),nn.LayerNorm(H),nn.ReLU(),nn.Dropout(dr)); d=H
        else:
            s.proj=nn.Sequential(nn.Linear(E,H),nn.LayerNorm(H),nn.ReLU(),nn.Dropout(dr)); d=H
        if use_lstm: s.lstm=nn.LSTM(H,H,2,batch_first=True,bidirectional=True,dropout=dr); d=2*H
        if use_att: s.att=Attn(d)
        s.cls=head(d)
    def forward(s,x,L):
        mask=mask_of(L,x.size(1)); e=s.emb(x)
        if s.use_cnn: e=torch.cat([c(e.permute(0,2,1)) for c in s.convs],1).permute(0,2,1)
        h=s.proj(e)
        if s.use_lstm: h,_=s.lstm(h)
        z=s.att(h,mask) if s.use_att else (mean_pool(h,mask) if s.use_lstm else max_pool(h,mask))
        return s.cls(z)
class CharLSTM(nn.Module):   # Bahnsen et al. 2017: 100-d embedding -> LSTM(100) -> dropout -> dense
    def __init__(s,V): super().__init__(); s.emb=nn.Embedding(V,100,padding_idx=0); s.lstm=nn.LSTM(100,100,batch_first=True); s.dr=nn.Dropout(0.5); s.fc=nn.Linear(100,2)
    def forward(s,x,L):
        o,_=s.lstm(s.emb(x)); last=o[torch.arange(x.size(0),device=x.device),(L.clamp(max=x.size(1))-1)]; return s.fc(s.dr(last))
class URLNetChar(nn.Module): # Le et al. 2018, character-level CNN branch: emb 32, kernels 3,4,5,6 x 256, max-over-time, FC 512-256-128
    def __init__(s,V): super().__init__(); s.emb=nn.Embedding(V,32,padding_idx=0); s.convs=nn.ModuleList([nn.Conv1d(32,256,k,padding=k//2) for k in (3,4,5,6)]); s.fc=nn.Sequential(nn.Dropout(0.5),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,2))
    def forward(s,x,L):
        e=s.emb(x).permute(0,2,1); return s.fc(torch.cat([F.relu(c(e)).max(2)[0] for c in s.convs],1))
class Highway(nn.Module):
    def __init__(s,d): super().__init__(); s.h=nn.Linear(d,d); s.g=nn.Linear(d,d)
    def forward(s,x): g=torch.sigmoid(s.g(x)); return g*F.relu(s.h(x))+(1-g)*x
class HDPCNNChar(nn.Module): # Zheng et al. 2022, character branch: region embedding + deep-pyramid conv blocks (BatchNorm, shortcuts) + highway
    def __init__(s,V,C=128):
        super().__init__(); s.emb=nn.Embedding(V,64,padding_idx=0); s.region=nn.Sequential(nn.Conv1d(64,C,3,padding=1),nn.BatchNorm1d(C))
        def cb(): return nn.Sequential(nn.ReLU(),nn.Conv1d(C,C,3,padding=1),nn.BatchNorm1d(C))
        s.first=nn.Sequential(cb(),cb()); s.blocks=nn.ModuleList([nn.Sequential(cb(),cb()) for _ in range(4)])
        s.hw=Highway(C); s.fc=nn.Sequential(nn.Dropout(0.3),nn.Linear(C,2))
    def forward(s,x,L):
        h=s.region(s.emb(x).permute(0,2,1)); h=h+s.first(h)
        for blk in s.blocks:
            if h.size(2)>2: h=F.max_pool1d(h,3,2,padding=1)
            h=h+blk(h)
        return s.fc(s.hw(F.relu(h).max(2)[0]))
class CCBLA(nn.Module): # Zhu et al. 2023: single-kernel char CNN -> BiLSTM -> attention -> dense
    def __init__(s,V): super().__init__(); s.emb=nn.Embedding(V,64,padding_idx=0); s.conv=nn.Conv1d(64,128,3,padding=1); s.lstm=nn.LSTM(128,64,batch_first=True,bidirectional=True); s.att=Attn(128); s.fc=nn.Sequential(nn.Dropout(0.3),nn.Linear(128,2))
    def forward(s,x,L):
        h=F.relu(s.conv(s.emb(x).permute(0,2,1))).permute(0,2,1); h,_=s.lstm(h); return s.fc(s.att(h,mask_of(L,x.size(1))))
class BiLSTMHighway(nn.Module): # Nanda & Goel 2024: BiLSTM -> gated highway -> attention re-weighting -> conv block -> dense
    def __init__(s,V): super().__init__(); s.emb=nn.Embedding(V,64,padding_idx=0); s.lstm=nn.LSTM(64,64,batch_first=True,bidirectional=True); s.hw=Highway(128); s.att=nn.Linear(128,1); s.conv=nn.Conv1d(128,128,3,padding=1); s.fc=nn.Sequential(nn.Dropout(0.3),nn.Linear(128,2))
    def forward(s,x,L):
        mask=mask_of(L,x.size(1)); h,_=s.lstm(s.emb(x)); h=s.hw(h)
        w=F.softmax(s.att(h).squeeze(-1).float().masked_fill(~mask,-1e9),1).unsqueeze(-1).to(h.dtype); n=mask.sum(1,keepdim=True).unsqueeze(-1)
        h=F.relu(s.conv((h*w*n).permute(0,2,1))).permute(0,2,1); return s.fc(max_pool(h,mask))
MODELS={"proposed":lambda V:Proposed(V),"cnn_only":lambda V:Proposed(V,use_lstm=False,use_att=False),"bilstm_only":lambda V:Proposed(V,use_cnn=False,use_att=False),"cnn_bilstm":lambda V:Proposed(V,use_att=False),
        "char_lstm":CharLSTM,"urlnet_char":URLNetChar,"hdpcnn_char":HDPCNNChar,"ccbla":CCBLA,"bilstm_highway":BiLSTMHighway}
@torch.no_grad()
def predict(m,X,L,bs=1024):
    m.eval(); p=np.zeros(len(L),dtype=np.float32)
    for xb,lb,_,b in batches(X,L,None,bs,False): p[b]=torch.softmax(m(xb.to(DEV),lb.to(DEV)).float(),1)[:,1].cpu().numpy()
    return p
def metrics(y,p,th=0.5):
    pr=(p>=th).astype(int); tp=int(((pr==1)&(y==1)).sum()); fp=int(((pr==1)&(y==0)).sum()); fn=int(((pr==0)&(y==1)).sum()); tn=int(((pr==0)&(y==0)).sum())
    P=tp/(tp+fp) if tp+fp else 0.; R=tp/(tp+fn) if tp+fn else 0.; Fm=2*P*R/(P+R) if P+R else 0.
    d=dict(n=int(len(y)),pos=int(y.sum()),tn=tn,fp=fp,fn=fn,tp=tp,acc=(tp+tn)/len(y),precision=P,recall=R,f1=Fm,fpr=(fp/(fp+tn) if fp+tn else None))
    if 0<y.sum()<len(y): d["roc_auc"]=float(roc_auc_score(y,p)); d["pr_auc"]=float(average_precision_score(y,p))
    return d
def train(name,Xtr,Ltr,ytr,Xva,Lva,yva,V,epochs,bs=256,lr=1e-3,weighted=True,seed=42,patience=3,log=print):
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.default_rng(seed)
    if DEV.type=="cuda": torch.cuda.manual_seed_all(seed)
    m=MODELS[name](V).to(DEV); n0=(ytr==0).sum(); n1=(ytr==1).sum()
    w=torch.tensor([len(ytr)/(2*n0),len(ytr)/(2*n1)],dtype=torch.float32,device=DEV) if weighted else None
    crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    steps=epochs*math.ceil(len(ytr)/bs); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=1e-5)
    amp=DEV.type=="cuda"; scaler=torch.amp.GradScaler("cuda") if amp else None
    best=(1e9,None,0); hist=[]; t0=time.time(); yt=ytr.astype(np.int64); bad=0
    diverged=False
    for ep in range(1,epochs+1):
        m.train(); tl=0;n=0
        for xb,lb,yb,_ in batches(Xtr,Ltr,yt,bs,True,rng):
            xb,lb,yb=xb.to(DEV),lb.to(DEV),yb.to(DEV); opt.zero_grad()
            if amp:
                with torch.amp.autocast("cuda"): loss=crit(m(xb,lb),yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss=crit(m(xb,lb),yb); loss.backward(); opt.step()
            sched.step(); lv=loss.item()
            if not math.isfinite(lv): diverged=True; break
            tl+=lv*len(yb); n+=len(yb)
        pv=predict(m,Xva,Lva) if not diverged else None
        if diverged or not np.isfinite(pv).all():
            diverged=True; log(name,'seed',seed,'DIVERGED (non-finite loss/predictions) at epoch',ep,'- restoring best checkpoint'); break
        pv=np.clip(pv,1e-7,1-1e-7); w0,w1=((w[0].item(),w[1].item()) if weighted else (1.,1.)); wv=np.where(yva==1,w1,w0)
        vl=float((-(yva*np.log(pv)+(1-yva)*np.log(1-pv))*wv).sum()/wv.sum()); mv=metrics(yva,pv)
        ptr_acc=None
        hist.append(dict(epoch=ep,train_loss=tl/n,val_loss=vl,val_acc=mv["acc"],val_f1=mv["f1"],val_auc=mv.get("roc_auc"),sec=round(time.time()-t0)))
        log(name,"seed",seed,hist[-1])
        if vl<best[0]: best=(vl,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},ep); bad=0
        else:
            bad+=1
            if bad>=patience: log(name,"early stop at epoch",ep); break
    if best[1] is None: raise RuntimeError(name+' diverged before the first validation')
    m.load_state_dict(best[1]); return m,dict(diverged=diverged,best_epoch=best[2],epochs_run=len(hist),early_stopped=bad>=patience,history=hist,params=sum(p.numel() for p in m.parameters()),train_sec=round(time.time()-t0))

import tldextract, warnings; warnings.filterwarnings("ignore")
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG=open(os.path.join(OUT,"log.txt"),"a",encoding="utf8")
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
def done(j): return os.path.exists(os.path.join(OUT,j+".json"))
def save(j,d): json.dump(d,open(os.path.join(OUT,j+".json"),"w"),indent=1)
def sub(idx,y,n,seed=42):
    if n is None or len(idx)<=n: return idx
    s,_=train_test_split(idx,train_size=n,random_state=seed,stratify=y[idx]); return s
SCH=re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
def has_scheme(u): return np.array([bool(SCH.match(x)) for x in u])
@torch.no_grad()
def timing(m,X,reps=5):
    n=min(20000,len(X)); xb=torch.from_numpy(X[:n].astype(np.int64)).to(DEV); L=torch.full((n,),MAXLEN,device=DEV); m.eval(); ts=[]
    for r in range(reps+1):
        if DEV.type=="cuda": torch.cuda.synchronize()
        t=time.time()
        for i in range(0,n,1024): m(xb[i:i+1024],L[i:i+1024])
        if DEV.type=="cuda": torch.cuda.synchronize()
        ts.append((time.time()-t)/n*1000)
    return float(np.median(ts[1:])*1000)        # ms per 1,000 URLs, first pass discarded as warm-up
def main():
    env=dict(torch=torch.__version__,device=str(DEV),gpu=(torch.cuda.get_device_name(0) if DEV.type=="cuda" else None),python=platform.python_version(),os=platform.platform(),epochs=EPOCHS,batch=BATCH,patience=PATIENCE,seeds_main=SEEDS_MAIN,smoke=SMOKE)
    save("env",env); log("ENV",env)
    df=load(); y=df.label.values; urls=df.url.values; log("data",len(df),int(y.sum()))
    MAIN=["proposed","char_lstm","urlnet_char","hdpcnn_char","ccbla","bilstm_highway"]; ABL=[("proposed_nocw",False),("cnn_only",True),("bilstm_only",True),("cnn_bilstm",True)]
    # ---------------- suite R: random stratified 80/10/10 split, one split per seed ----------------
    for seed in SEEDS_MAIN:
        jobs=[(j,True) for j in MAIN]+(ABL if seed==42 else [])
        if all(done(f"R_{j}_s{seed}") for j,_ in jobs): continue
        tr,va,te=canon_split(df,seed); s_tr=sub(tr,y,NTR,seed); s_va=sub(va,y,None if NTR is None else 2000,seed)
        if SMOKE: te=sub(te,y,5000,seed)
        vocab=build_vocab(urls[s_tr]); V=len(vocab); Xtr,Ltr=encode(urls[s_tr],vocab); Xva,Lva=encode(urls[s_va],vocab); Xte,Lte=encode(urls[te],vocab); sch=has_scheme(urls[te])
        log("R seed",seed,"vocab",V,"train",len(s_tr),int(y[s_tr].sum()),"val",len(s_va),"test",len(te))
        for j,w in jobs:
            if done(f"R_{j}_s{seed}"): continue
            name="proposed" if j=="proposed_nocw" else j
            m,info=train(name,Xtr,Ltr,y[s_tr],Xva,Lva,y[s_va],V,EPOCHS,BATCH,weighted=w,seed=seed,patience=PATIENCE,log=log); p=predict(m,Xte,Lte)
            r=dict(job=j,seed=seed,weighted=w,vocab=V,n_train=int(len(s_tr)),n_train_pos=int(y[s_tr].sum()),n_val=int(len(s_va)),test=metrics(y[te],p),test_full_urls=metrics(y[te][sch],p[sch]),infer_ms_per_1000=timing(m,Xte),**info)
            if j=="proposed" and seed==42:
                np.save(os.path.join(OUT,"proposed_s42_test_prob.npy"),p); np.save(os.path.join(OUT,"proposed_s42_test_index.npy"),te)
                torch.save(dict(model_state_dict=m.state_dict(),vocab=vocab),os.path.join(OUT,"proposed_s42.pt"))
            save(f"R_{j}_s{seed}",r); log("RESULT",j,"seed",seed,{k:(round(v,4) if isinstance(v,float) else v) for k,v in r["test"].items()},"best_epoch",r["best_epoch"])
        del Xtr,Xva,Xte
    # ---------------- registered domains ----------------
    ex=tldextract.TLDExtract(suffix_list_urls=()); hostre=re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*://)?([^/?#]*)"); cache={}
    def rd(u):
        h=hostre.match(u).group(1).split("@")[-1].split(":")[0].lower()
        if h not in cache: cache[h]=ex(h).top_domain_under_public_suffix or h
        return cache[h]
    # ---------------- suite G: domain-grouped 80/10/10 split ----------------
    if not done("G_proposed"):
        dom=np.array([rd(u) for u in urls]); ud=np.unique(dom); perm=np.random.default_rng(42).permutation(len(ud)); n=len(ud)
        part=np.empty(n,dtype=np.int8); part[perm[:int(.8*n)]]=0; part[perm[int(.8*n):int(.9*n)]]=1; part[perm[int(.9*n):]]=2
        pu=part[np.searchsorted(ud,dom)]; g=[np.where(pu==k)[0] for k in range(3)]
        sizes={nm:dict(urls=int(len(ix)),phishing=int(y[ix].sum()),legitimate=int((y[ix]==0).sum()),domains=int((part==k).sum())) for k,(nm,ix) in enumerate(zip(("train","val","test"),g))}
        ov=len(set(dom[g[0]])&set(dom[g[2]]))+len(set(dom[g[0]])&set(dom[g[1]]))+len(set(dom[g[1]])&set(dom[g[2]])); log("G sizes",sizes,"overlap",ov)
        big={k:int(v) for k,v in pd.Series(dom).value_counts().head(10).items()}
        gt=sub(g[0],y,NTR); gv=sub(g[1],y,None if NTR is None else 2000); gte=g[2] if not SMOKE else sub(g[2],y,5000)
        vg=build_vocab(urls[gt]); Xtr,Ltr=encode(urls[gt],vg); Xva,Lva=encode(urls[gv],vg); Xte,Lte=encode(urls[gte],vg)
        m,info=train("proposed",Xtr,Ltr,y[gt],Xva,Lva,y[gv],len(vg),EPOCHS,BATCH,patience=PATIENCE,log=log); p=predict(m,Xte,Lte); sch=has_scheme(urls[gte])
        save("G_proposed",dict(sizes=sizes,domain_overlap=ov,unique_domains=int(n),largest_domains=big,n_train=int(len(gt)),n_train_pos=int(y[gt].sum()),test=metrics(y[gte],p),test_full_urls=metrics(y[gte][sch],p[sch]),**info)); log("RESULT G",metrics(y[gte],p))
        del Xtr,Xva,Xte
    # ---------------- suite C: chronological split on PhishTank submission time ----------------
    if not done("C_proposed"):
        if PT is None: log("PhishTank.csv not found - chronological split skipped")
        else:
            pt=pd.read_csv(PT,usecols=["url","submission_time"]); pt["url"]=pt.url.astype(str).str.strip(); pt["t"]=pd.to_datetime(pt.submission_time,utc=True)
            pt=pt.sort_values("t").drop_duplicates("url",keep="first"); tmap=dict(zip(pt.url,pt.t))
            tt=pd.Series([tmap.get(u) for u in urls]); dated=tt.notna().values&(y==1); T=tt[dated].quantile(0.8).floor("D")
            post=dated&np.array([(t is not None and t==t and t>=T) for t in tt]); pre=dated&~post
            tr,va,te=canon_split(df,42); intr=np.zeros(len(y),bool); intr[tr]=True; inva=np.zeros(len(y),bool); inva[va]=True; inte=np.zeros(len(y),bool); inte[te]=True
            pool=np.where(intr&~post)[0]; vpool=np.where(inva&~post)[0]; ct=sub(pool,y,NTR); cv=sub(vpool,y,None if NTR is None else 2000)
            test_post=np.where(post)[0]; test_leg=np.where(inte&(y==0))[0]; test_pre=np.where(inte&pre)[0]; test_und=np.where(inte&(y==1)&~dated)[0]
            if SMOKE: test_leg=test_leg[:5000]
            log("C dated",int(dated.sum()),"cutoff",T,"post",int(post.sum()),"pre",int(pre.sum()))
            vc=build_vocab(urls[ct]); Xtr,Ltr=encode(urls[ct],vc); Xva,Lva=encode(urls[cv],vc)
            m,info=train("proposed",Xtr,Ltr,y[ct],Xva,Lva,y[cv],len(vc),EPOCHS,BATCH,patience=PATIENCE,log=log)
            def ev(ix): X,L=encode(urls[ix],vc); return predict(m,X,L)
            pp=ev(test_post); pl=ev(test_leg); ppre=ev(test_pre); po=ev(test_und); yy=np.r_[np.ones(len(pp),int),np.zeros(len(pl),int)]; sch=has_scheme(urls[test_leg])
            res=dict(cutoff=str(T),dated_phishing=int(dated.sum()),first=str(tt[dated].min()),last=str(tt[dated].max()),post_cutoff=int(post.sum()),pre_cutoff=int(pre.sum()),pre_in_train_partition=int((intr&pre).sum()),
                     train_pool=int(len(pool)),n_train=int(len(ct)),n_train_pos=int(y[ct].sum()),n_train_dated_pre=int(pre[ct].sum()),test=metrics(yy,np.r_[pp,pl]),
                     recall_post_cutoff=float((pp>=.5).mean()),recall_pre_cutoff_heldout=float((ppre>=.5).mean()),n_pre_heldout=int(len(ppre)),recall_undated_heldout=float((po>=.5).mean()),n_undated_heldout=int(len(po)),
                     fpr_legit=float((pl>=.5).mean()),n_legit=int(len(pl)),fpr_legit_full_urls=float((pl[sch]>=.5).mean()),n_legit_full=int(sch.sum()),**info)
            save("C_proposed",res); log("RESULT C",{k:v for k,v in res.items() if k!="history"})
    log("ALL DONE - copy the folder",OUT,"back")
if __name__=="__main__": main()
