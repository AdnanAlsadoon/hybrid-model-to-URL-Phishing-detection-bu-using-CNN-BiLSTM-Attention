# Error analysis of the seed-42 proposed model (unified run). Usage: python error_analysis.py DATA.csv PROB.npy INDEX.npy OUT.json
import sys,re,json,collections,numpy as np,pandas as pd,tldextract
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score
DATA,PROB,IDX,OUT=sys.argv[1:5]
df=pd.read_csv(DATA); df["url"]=df["url"].astype(str).str.strip(); df["label"]=df["label"].astype(int)
y=df.label.values; idx=np.arange(len(df))
tv,te=train_test_split(idx,test_size=0.1,random_state=42,stratify=y)
tr,va=train_test_split(tv,test_size=0.1/0.9,random_state=42,stratify=y[tv])
ti=np.load(IDX); p=np.load(PROB).astype(float)
assert len(ti)==len(te) and set(ti)==set(te), "test index differs from canonical split"
T=df.iloc[ti].reset_index(drop=True); T["p"]=p; T["pred"]=(p>=0.5).astype(int)
ext=tldextract.TLDExtract(suffix_list_urls=(),cache_dir=None)
def host_of(u):
    u2=u if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://",u) else "//"+u
    m=re.match(r"^[a-zA-Z0-9+.-]*://([^/?#]*)",u2) or re.match(r"^//([^/?#]*)",u2)
    h=m.group(1) if m else u; h=h.split("@")[-1]; return re.sub(r":\d+$","",h).lower()
def rd(u):
    e=ext(host_of(u)); return e.registered_domain or host_of(u)
def has_scheme(u): return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://",u))
def mets(d):
    tn=int(((d.label==0)&(d.pred==0)).sum());fp=int(((d.label==0)&(d.pred==1)).sum());fn=int(((d.label==1)&(d.pred==0)).sum());tp=int(((d.label==1)&(d.pred==1)).sum())
    P=tp/(tp+fp) if tp+fp else float("nan");R=tp/(tp+fn) if tp+fn else float("nan")
    r=dict(n=len(d),pos=int(d.label.sum()),tn=tn,fp=fp,fn=fn,tp=tp,acc=(tn+tp)/max(len(d),1),precision=P,recall=R,f1=(2*P*R/(P+R) if P==P and R==R and P+R else float("nan")),fpr=(fp/(fp+tn) if fp+tn else float("nan")))
    if d.label.nunique()==2: r["roc_auc"]=roc_auc_score(d.label,d.p); r["pr_auc"]=average_precision_score(d.label,d.p)
    return r
R={}
R["overall"]=mets(T)
# threshold sweep
R["thresholds"]=[]
for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    d=T.copy(); d["pred"]=(d.p>=th).astype(int); m=mets(d); m["theta"]=th; R["thresholds"].append(m)
best=max(([th/100,None] for th in range(5,96)),key=lambda a:0)
bf=(-1,0)
for th in np.arange(0.05,0.951,0.01):
    d=T.copy(); d["pred"]=(d.p>=th).astype(int); f=mets(d)["f1"]
    if f>bf[0]: bf=(f,round(float(th),2))
R["best_f1_threshold"]=bf
# training domain sets
trdf=df.iloc[tr]
print("computing train domains...",flush=True)
trdom={0:set(),1:set()}
for u,l in zip(trdf.url.values,trdf.label.values): trdom[int(l)].add(rd(u))
alltr=trdom[0]|trdom[1]
T["rd"]=[rd(u) for u in T.url]; T["scheme"]=[has_scheme(u) for u in T.url]
T["unseen"]=[r not in alltr for r in T.rd]
R["format"]={"bare_host":mets(T[~T.scheme]),"full_url":mets(T[T.scheme])}
R["novelty"]={"unseen":mets(T[T.unseen]),"seen":mets(T[~T.unseen]),"unseen_full_url":mets(T[T.unseen&T.scheme]),"seen_full_url":mets(T[(~T.unseen)&T.scheme])}
# rule lists
SHORT={"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","rebrand.ly","cutt.ly","shorturl.at","tiny.cc","rb.gy","t.ly","s.id","lnkd.in","bl.ink","soo.gd","v.gd","x.co","adf.ly","bit.do","mcaf.ee","tr.im"}
SHARED={"blogspot.com","blogspot.co.uk","wixsite.com","wix.com","weebly.com","webflow.io","github.io","gitlab.io","netlify.app","vercel.app","herokuapp.com","firebaseapp.com","web.app","appspot.com","workers.dev","pages.dev","amazonaws.com","azurewebsites.net","windows.net","digitaloceanspaces.com","storage.googleapis.com","googleusercontent.com","sharepoint.com","onedrive.live.com","dropbox.com","docs.google.com","drive.google.com","canva.com","sites.google.com","wordpress.com","000webhostapp.com","glitch.me","repl.co","ipfs.io","mybluehost.me","yolasite.com","jimdofree.com","godaddysites.com","square.site","myshopify.com","atspace.com","mefound.com","tripod.com","angelfire.com"}
KW=re.compile(r"login|signin|sign-in|verify|verification|account|secure|update|bank|paypal|wallet|password|confirm|auth|billing|support|invoice|payment",re.I)
def path_query(u):
    u2=u if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://",u) else "//"+u
    m=re.match(r"^(?:[a-zA-Z0-9+.-]*:)?//[^/?#]*(.*)$",u2); return m.group(1) if m else ""
def multiparam(u):
    q=path_query(u).split("?",1)[1] if "?" in path_query(u) else ""
    return q.count("=")>=2 or bool(re.search(r"(session|sid|token|utm_|tracking|clickid|uid)=",q,re.I))
def hostonly(u): return path_query(u).strip("/")==""
def cat_fn(u,r):
    if len(u)>200: return "Truncated (URL longer than 200 characters)"
    if r in SHORT or re.search(r"https?://",path_query(u),re.I): return "URL shortener or embedded redirect"
    if r in SHARED or any(r.endswith("."+s) for s in SHARED): return "Shared hosting / site-builder / cloud platform"
    if r in trdom[0]: return "Registered domain also present among legitimate training URLs"
    if hostonly(u): return "Host-only URL (no path or query)"
    if not KW.search(u): return "No phishing-typical keyword in the URL"
    return "Other"
def cat_fp(u,r):
    if len(u)>200: return "Truncated (URL longer than 200 characters)"
    if r in SHORT or re.search(r"https?://",path_query(u),re.I): return "URL shortener or embedded redirect"
    if r in SHARED or any(r.endswith("."+s) for s in SHARED): return "Shared hosting / site-builder / cloud platform"
    if r in trdom[1]: return "Registered domain also present among phishing training URLs"
    if KW.search(u): return "Phishing-typical keyword in the URL"
    if multiparam(u): return "Multi-parameter or session query string"
    if hostonly(u): return "Host-only URL (no path or query)"
    if r not in alltr: return "Registered domain unseen in training"
    return "Other"
FN=T[(T.label==1)&(T.pred==0)]; FP=T[(T.label==0)&(T.pred==1)]
order_fn=["Truncated (URL longer than 200 characters)","URL shortener or embedded redirect","Shared hosting / site-builder / cloud platform","Registered domain also present among legitimate training URLs","Host-only URL (no path or query)","No phishing-typical keyword in the URL","Other"]
order_fp=["Truncated (URL longer than 200 characters)","URL shortener or embedded redirect","Shared hosting / site-builder / cloud platform","Registered domain also present among phishing training URLs","Phishing-typical keyword in the URL","Multi-parameter or session query string","Host-only URL (no path or query)","Registered domain unseen in training","Other"]
cf=collections.Counter(cat_fn(u,r) for u,r in zip(FN.url,FN.rd)); cp=collections.Counter(cat_fp(u,r) for u,r in zip(FP.url,FP.rd))
R["fn_cats"]=[(k,cf.get(k,0)) for k in order_fn]; R["fp_cats"]=[(k,cp.get(k,0)) for k in order_fp]
# overlap (non-exclusive) counts so the reader sees how much the ordering matters
def flags_fn(u,r): return {"long>200":len(u)>200,"shortener/redirect":r in SHORT or bool(re.search(r"https?://",path_query(u),re.I)),"shared platform":r in SHARED or any(r.endswith("."+s) for s in SHARED),"domain in legit train":r in trdom[0],"host-only":hostonly(u),"no keyword":not KW.search(u)}
def flags_fp(u,r): return {"long>200":len(u)>200,"shortener/redirect":r in SHORT or bool(re.search(r"https?://",path_query(u),re.I)),"shared platform":r in SHARED or any(r.endswith("."+s) for s in SHARED),"domain in phish train":r in trdom[1],"keyword":bool(KW.search(u)),"multi-param":multiparam(u),"host-only":hostonly(u),"unseen domain":r not in alltr}
def fsum(sub,fl):
    c=collections.Counter()
    for u,r in zip(sub.url,sub.rd):
        for k,v in fl(u,r).items(): c[k]+=int(v)
    return dict(c)
R["fn_flags"]=fsum(FN,flags_fn); R["fp_flags"]=fsum(FP,flags_fp)
tp_=T[(T.label==1)&(T.pred==1)]; allph=T[T.label==1]; allfull=T[(T.label==0)&T.scheme]
def share(sub,f): return float(np.mean([f(u) for u in sub.url])) if len(sub) else float("nan")
www=lambda u: bool(re.match(r"^(https?://)?www\.",u,re.I))
R["facts"]=dict(fn_len_median=float(FN.url.str.len().median()),phish_len_median=float(allph.url.str.len().median()),fn_www=share(FN,www),phish_www=share(allph,www),fn_unseen=float(FN.unseen.mean()),fp_www=share(FP,www),fp_noWww=1-share(FP,www),legfull_noWww=1-share(allfull,www),fp_unseen=float(FP.unseen.mean()),fp_scheme=float(FP.scheme.mean()),fn_mean_prob=float(FN.p.mean()),fp_mean_prob=float(FP.p.mean()),fn_median_prob=float(FN.p.median()),fp_median_prob=float(FP.p.median()),n_fn=len(FN),n_fp=len(FP))
ex={}
for nm,sub,fn in [("FN",FN,cat_fn),("FP",FP,cat_fp)]:
    g=collections.defaultdict(list)
    for u,r in zip(sub.url,sub.rd): g[fn(u,r)].append(u[:90])
    ex[nm]={k:v[:4] for k,v in g.items()}
R["examples"]=ex
json.dump(R,open(OUT,"w"),indent=1,default=float)
print(json.dumps({k:R[k] for k in ["overall","best_f1_threshold","fn_cats","fp_cats","facts","fn_flags","fp_flags"]},indent=1,default=float))
