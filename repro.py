import numpy as np,pandas as pd,json
from sklearn.model_selection import train_test_split
df=pd.read_csv("/mnt/user-data/uploads/first objective/phishing_dataset_final.csv"); df["url"]=df["url"].astype(str).str.strip(); y=df.label.values.astype(int); urls=df.url.values
def canon(seed=42):
    idx=np.arange(len(df)); tv,te=train_test_split(idx,test_size=0.1,random_state=seed,stratify=y); tr,va=train_test_split(tv,test_size=0.1/0.9,random_state=seed,stratify=y[tv]); return tr,va,te
tr,va,te=canon(42)
te0=np.load('/home/claude/rf/proposed_s42_test_index.npy'); print('test idx match',np.array_equal(np.sort(te),np.sort(te0)))
out={'R':{k:dict(n=int(len(i)),pos=int(y[i].sum())) for k,i in zip(('train','val','test'),(tr,va,te))}}
pt=pd.read_csv("/mnt/user-data/uploads/first objective/phishing DataSet/PhishTank.csv",usecols=["url","submission_time"]); pt["url"]=pt.url.astype(str).str.strip(); pt["t"]=pd.to_datetime(pt.submission_time,utc=True)
pt=pt.sort_values("t").drop_duplicates("url",keep="first"); tmap=dict(zip(pt.url,pt.t))
tt=pd.Series([tmap.get(u) for u in urls]); dated=tt.notna().values&(y==1); T=tt[dated].quantile(0.8).floor("D"); print(T)
post=dated&np.array([(t is not None and t==t and t>=T) for t in tt]); pre=dated&~post
print(post.sum(),pre.sum())
ine=np.zeros(len(y),bool);ine[te]=True;inv=np.zeros(len(y),bool);inv[va]=True;intr=np.zeros(len(y),bool);intr[tr]=True
def d(m): return dict(n=int(m.sum()),pos=int(y[m].sum()))
out['C']=dict(train_pool=d(intr&~post),val_pool=d(inv&~post),post_in_train=d(intr&post),post_in_val=d(inv&post),post_in_test=d(ine&post),pre_heldout=d(ine&pre),undated_heldout=d(ine&(y==1)&~dated),legit_test=d(ine&(y==0)),pre_in_train=d(intr&pre),dated_in_train=d(intr&dated))
print(json.dumps(out,indent=1)); json.dump(out,open('/home/claude/repro.json','w'))
