import sys,os,time,pickle
from lib import *
torch.set_num_threads(1)
d=np.load('work/windows.npz'); X,Y,S=d['X'],d['Y'],d['S']
subs=sorted(set(S.tolist())); cnt={s:int((S==s).sum()) for s in subs}
# balanced 5 folds (greedy)
order=sorted(subs,key=lambda s:-cnt[s]); folds=[[] for _ in range(5)]; tot=[0]*5
for s in order:
    i=tot.index(min(tot)); folds[i].append(s); tot[i]+=cnt[s]
os.makedirs('work/out',exist_ok=True)
SHOW_TE=[6,13,20,28]; SHOW_VA=[9,15,24]
jobs=[]
for kind in ['cnn','cnn_aug','lr']:
    for f in range(5): jobs.append(('cv',kind,f,0))
for kind in ['cnn','cnn_aug','lr']:
    for seed in range(3): jobs.append(('show',kind,0,seed))
def run(job):
    sp,kind,f,seed=job; name=f'{sp}_{kind}_{f}_{seed}'; path=f'work/out/{name}.pkl'
    if os.path.exists(path): return False
    if sp=='cv':
        te=folds[f]; rest=[s for s in subs if s not in te]; r=np.random.default_rng(100+f); va=list(r.choice(rest,3,replace=False))
    else: te=SHOW_TE; va=SHOW_VA
    mte=np.isin(S,te); mva=np.isin(S,va); mtr=~(mte|mva); mtr&=(np.arange(len(S))%2==0)
    t0=time.time()
    m=train_model(kind,X[mtr],Y[mtr],X[mva],Y[mva],seed=seed,epochs=14,log=lambda s:None)
    lt=infer(m,X[mte]); 
    res={'job':job,'te':te,'va':va,'idx':np.where(mte)[0],'logits':lt,'vlog':None,'T':m['T'],'best_epoch':m.get('best_epoch'),'secs':time.time()-t0}
    if sp=='show': res['state']=({k:v.numpy() for k,v in m['model'].state_dict().items()}); res['norm']=m['norm']
    pickle.dump(res,open(path,'wb')); print(name,'F1',round(macro_f1(Y[mte],lt.argmax(1))[0],4),'T',round(m['T'],2),round(res['secs']),'s',flush=True); return True
if __name__=='__main__':
    w=int(sys.argv[1]); t0=time.time()
    for i,j in enumerate(jobs):
        if i%2!=w: continue
        if time.time()-t0>38: break
        run(j)
