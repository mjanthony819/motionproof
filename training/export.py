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
# ---- export weights (seed 0) ----
def r6(a): return np.round(np.asarray(a,dtype=np.float64),6).tolist()
def fold(conv,bn):
    w=conv.weight.detach().numpy().astype(np.float64); b=conv.bias.detach().numpy().astype(np.float64)
    g=bn.weight.detach().numpy(); be=bn.bias.detach().numpy(); mu=bn.running_mean.numpy(); var=bn.running_var.numpy(); s=g/np.sqrt(var+bn.eps)
    return {'w':r6(w*s[:,None,None]),'b':r6((b-mu)*s+be)}
export={'names':NAMES,'models':{}}
for kind in ['lr','cnn','cnn_aug']:
    r=load(f'work/out/show_{kind}_0_0.pkl'); m=rebuild(kind,r)
    if kind=='lr':
        export['models'][kind]={'type':'lr','mu':r6(r['norm']['mu']),'sd':r6(r['norm']['sd']),'W':r6(m['model'].fc.weight.detach().numpy()),'b':r6(m['model'].fc.bias.detach().numpy()),'T':r['T']}
    else:
        mm=m['model']
        export['models'][kind]={'type':'cnn','mu':r6(r['norm']['mu']),'sd':r6(r['norm']['sd']),'layers':[fold(mm.c1,mm.b1),fold(mm.c2,mm.b2),fold(mm.c3,mm.b3)],'fcW':r6(mm.fc.weight.detach().numpy()),'fcb':r6(mm.fc.bias.detach().numpy()),'T':r['T']}
os.makedirs('data',exist_ok=True)
# golden vectors: 40 test windows (+ a perturbed one each) logits from python models
rng=np.random.default_rng(5); gi=rng.choice(len(Xt),40,replace=False); gx=Xt[gi]
gold={'windows':r6(gx),'y':Yt[gi].tolist(),'logits':{}}
for kind in ['lr','cnn','cnn_aug']:
    m=rebuild(kind,load(f'work/out/show_{kind}_0_0.pkl')); gold['logits'][kind]=r6(infer(m,gx))
json.dump(gold,open('data/golden.json','w'))
# perturbation golden
pg={}
for c in STRESS:
    if c.startswith('noise'): continue
    pg[c]=r6(perturb(gx[:3],c))
json.dump(pg,open('data/golden_perturb.json','w'))
json.dump(export,open('data/models.json','w'))
print('models.json bytes',os.path.getsize('data/models.json'))
