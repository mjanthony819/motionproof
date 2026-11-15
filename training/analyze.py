import pickle,glob,json,time
from lib import *
d=np.load('work/windows.npz'); X,Y,S=d['X'],d['Y'],d['S']
def load(p): return pickle.load(open(p,'rb'))
def ece(P,y,bins=10):
    conf=P.max(1); pred=P.argmax(1); acc=(pred==y); e=0; rel=[]
    for b in range(bins):
        m=(conf>b/bins)&(conf<=(b+1)/bins)
        if m.any(): e+=m.mean()*abs(acc[m].mean()-conf[m].mean()); rel.append([float(conf[m].mean()),float(acc[m].mean()),int(m.sum())])
    return float(e),rel
def nll(P,y): return float(-np.log(P[np.arange(len(y)),y]+1e-9).mean())
def metrics(logits,y,T):
    P=softmax(logits/T); p=P.argmax(1); f1,pc=macro_f1(y,p)
    e,rel=ece(P,y)
    cm=np.zeros((K,K),int)
    for a,b in zip(y,p): cm[a,b]+=1
    # coverage-accuracy
    conf=P.max(1); cov=[]
    for th in [0,0.5,0.6,0.7,0.8,0.9,0.95]:
        m=conf>=th; cov.append([th,float(m.mean()),float((p[m]==y[m]).mean()) if m.any() else None])
    return {'acc':float((p==y).mean()),'macro_f1':f1,'per_class_f1':[None if np.isnan(v) else v for v in pc],'ece':e,'nll':nll(P,y),'reliability':rel,'cm':cm.tolist(),'coverage':cov}
R={'names':NAMES,'models':{}}
for kind in ['lr','cnn','cnn_aug']:
    # CV out-of-fold
    lg=np.zeros((len(Y),K)); Ts=[]; per_fold=[]
    msk=np.zeros(len(Y),bool); P=np.zeros((len(Y),K))
    for f in range(5):
        r=load(f'work/out/cv_{kind}_{f}_0.pkl'); i=r['idx']; Pf=softmax(r['logits']/r['T']); P[i]=Pf; msk[i]=True; Ts.append(r['T'])
        per_fold.append({'subjects':[int(s) for s in r['te']],'macro_f1':macro_f1(Y[i],Pf.argmax(1))[0],'acc':float((Pf.argmax(1)==Y[i]).mean())})
    p=P.argmax(1); f1,pc=macro_f1(Y,p); e,rel=ece(P,Y)
    cm=np.zeros((K,K),int)
    for a,b in zip(Y,p): cm[a,b]+=1
    conf=P.max(1); cov=[]
    for th in [0,0.5,0.6,0.7,0.8,0.9,0.95]:
        m=conf>=th; cov.append([th,float(m.mean()),float((p[m]==Y[m]).mean())])
    # per-subject macro-F1 (subject-level spread)
    ps={}
    for s in sorted(set(S.tolist())):
        m=S==s; ps[int(s)]=float(macro_f1(Y[m],p[m])[0])
    R['models'][kind]={'cv':{'macro_f1':f1,'acc':float((p==Y).mean()),'per_class_f1':pc,'ece':e,'nll':nll(P,Y),'reliability':rel,'cm':cm.tolist(),'coverage':cov,'per_fold':per_fold,'temps':Ts,'per_subject_f1':ps,'fold_f1_mean':float(np.mean([x['macro_f1'] for x in per_fold])),'fold_f1_std':float(np.std([x['macro_f1'] for x in per_fold]))}}
    sh=[load(f'work/out/show_{kind}_0_{s}.pkl') for s in range(3)]
    R['models'][kind]['show']=[metrics(r['logits'],Y[r['idx']],r['T']) for r in sh]
    R['models'][kind]['show_T']=[r['T'] for r in sh]
    R['models'][kind]['show_f1_seeds']=[m['macro_f1'] for m in R['models'][kind]['show']]
    print(kind,'CV F1 %.3f acc %.3f ECE %.3f | folds %.3f±%.3f | show seeds'%(f1,R['models'][kind]['cv']['acc'],e,R['models'][kind]['cv']['fold_f1_mean'],R['models'][kind]['cv']['fold_f1_std']),[round(x,3) for x in R['models'][kind]['show_f1_seeds']])
json.dump(R,open('work/out/results_cv_show.json','w'))
print('classes F1 CV:',{k:[round(x,2) for x in R['models'][k]['cv']['per_class_f1']] for k in R['models']})
