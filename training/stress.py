import pickle,json,time,base64,os
import pandas as pd
from lib import *
d=np.load('work/windows.npz'); X,Y,S=d['X'],d['Y'],d['S']
def load(p): return pickle.load(open(p,'rb'))
def rebuild(kind,r):
    if kind=='lr':
        m=Linear(67); sd={k:torch.from_numpy(v) for k,v in r['state'].items()}; m.load_state_dict(sd)
    else:
        m=CNN(); m.load_state_dict({k:torch.from_numpy(v) for k,v in r['state'].items()})
    m.eval(); return {'model':m,'norm':r['norm'],'T':r['T'],'kind':kind}
te=np.isin(S,[6,13,20,28]); Xt=X[te]; Yt=Y[te]
import sys
kinds=sys.argv[1:]
stress={}
for kind in kinds:
    rows={}
    for sd_ in range(3):
        m=rebuild(kind,load(f'work/out/show_{kind}_0_{sd_}.pkl'))
        for c in STRESS:
            Xp=perturb(Xt,c,np.random.default_rng(1)); lg=infer(m,Xp); P=softmax(lg/m['T']); p=P.argmax(1)
            rows.setdefault(c,[]).append({'f1':macro_f1(Yt,p)[0],'acc':float((p==Yt).mean()),'conf':float(P.max(1).mean())})
    stress[kind]={c:{'f1_mean':float(np.mean([x['f1'] for x in v])),'f1_std':float(np.std([x['f1'] for x in v])),'acc_mean':float(np.mean([x['acc'] for x in v])),'conf_mean':float(np.mean([x['conf'] for x in v]))} for c,v in rows.items()}
    print(kind,{c:round(v['f1_mean'],2) for c,v in stress[kind].items()})
for k in kinds: json.dump(stress[k],open(f'work/out/stress_{k}.json','w'))
