import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, json
torch.set_num_threads(2)
NAMES=['walking','running','shuffling','stairs up','stairs down','standing','sitting','lying','cycling']
K=9; W=100; FS=50
BANDS=[(0.5,1),(1,2),(2,3.5),(3.5,6),(6,12)]
# ---------- features (mirrored in JS) ----------
def _basis():
    n=np.arange(W); cs=[];sn=[]
    for (lo,hi) in BANDS:
        ks=[k for k in range(1,W//2+1) if lo<=k*FS/W<hi]
        cs.append(np.array([np.cos(2*np.pi*k*n/W) for k in ks])); sn.append(np.array([np.sin(2*np.pi*k*n/W) for k in ks]))
    return cs,sn
CS,SN=_basis()
def features(X):
    # X: (N,100,6) -> (N,F)
    X=X.astype(np.float64); N=len(X); out=[]
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1)
    jerk=np.abs(np.diff(X,axis=1)).mean(1)
    out+= [mean,std,mn,mx,jerk]
    for a in (0,3):
        m=np.sqrt((X[:,:,a:a+3]**2).sum(2))
        out+=[m.mean(1,keepdims=True), m.std(1,keepdims=True)]
    Xc=X-mean[:,None,:]
    bands=[]
    tot=np.zeros((N,6))+1e-9
    pw=[]
    for b in range(len(BANDS)):
        re=np.einsum('nwc,kw->nck',Xc,CS[b]); im=np.einsum('nwc,kw->nck',Xc,SN[b])
        p=((re**2+im**2).sum(2))/W; pw.append(p); tot+=p
    for p in pw: out.append(p/tot)
    for i in range(3):
        a=Xc[:,:,i]; b=Xc[:,:,3+i]
        out.append(((a*b).mean(1)/(a.std(1)*b.std(1)+1e-6))[:,None])
    return np.concatenate(out,1).astype(np.float32)
# ---------- models ----------
class CNN(nn.Module):
    def __init__(s,c1=24,c2=48,c3=64):
        super().__init__()
        s.c1=nn.Conv1d(6,c1,5,padding=2); s.b1=nn.BatchNorm1d(c1)
        s.c2=nn.Conv1d(c1,c2,5,padding=2); s.b2=nn.BatchNorm1d(c2)
        s.c3=nn.Conv1d(c2,c3,3,padding=1); s.b3=nn.BatchNorm1d(c3)
        s.fc=nn.Linear(c3*2,K); s.dp=nn.Dropout(0.2)
    def forward(s,x):  # x (N,6,100) standardized
        x=F.max_pool1d(F.relu(s.b1(s.c1(x))),2)
        x=F.max_pool1d(F.relu(s.b2(s.c2(x))),2)
        x=F.relu(s.b3(s.c3(x)))
        x=torch.cat([x.mean(2),x.amax(2)],1)
        return s.fc(s.dp(x))
class Linear(nn.Module):
    def __init__(s,d): super().__init__(); s.fc=nn.Linear(d,K)
    def forward(s,x): return s.fc(x)
# ---------- perturbations (mirrored in JS) ----------
def rot(X,ch0,deg,axis):
    t=np.deg2rad(deg); c,s=np.cos(t),np.sin(t)
    R=np.eye(3)
    i,j=[(1,2),(0,2),(0,1)][axis]
    R[i,i]=c;R[i,j]=-s;R[j,i]=s;R[j,j]=c
    Y=X.copy(); Y[:,:,ch0:ch0+3]=X[:,:,ch0:ch0+3]@R.T.astype(X.dtype); return Y
def stretch(X,f):
    # resample time axis by factor f (f>1: slower motion seen as lower freq): y[t]=x[t*f] edge clamped with linear interp
    N,Wn,C=X.shape; t=np.arange(Wn)*f; t=np.clip(t,0,Wn-1); i0=np.floor(t).astype(int); i1=np.minimum(i0+1,Wn-1); fr=(t-i0)[None,:,None]
    return (X[:,i0,:]*(1-fr)+X[:,i1,:]*fr).astype(X.dtype)
def perturb(X,name,rng=None):
    if name=='clean': return X
    kind,_,v=name.partition(':'); v=float(v) if v else 0
    if kind=='noise': r=rng or np.random.default_rng(0); return (X+r.normal(0,v,X.shape)).astype(X.dtype)
    if kind=='gain': return (X*v).astype(X.dtype)
    if kind=='thigh_rot': return rot(X,3,v,1)
    if kind=='back_rot': return rot(X,0,v,1)
    if kind=='drop_thigh': Y=X.copy(); Y[:,:,3:]=0; return Y
    if kind=='drop_back': Y=X.copy(); Y[:,:,:3]=0; return Y
    if kind=='stretch': return stretch(X,v)
    if kind=='clip': return np.clip(X,-v,v).astype(X.dtype)
    raise ValueError(name)
STRESS=['clean','noise:0.05','noise:0.15','gain:0.85','gain:1.15','thigh_rot:15','thigh_rot:30','back_rot:15','back_rot:30','drop_thigh','drop_back','stretch:0.8','stretch:1.25','clip:2']
# augmentation for torch batch (N,100,6) numpy
def augment(X,rng):
    X=X.copy(); N=len(X)
    # random small rotations per sensor
    for ch0 in (0,3):
        deg=rng.normal(0,12,N); ax=rng.integers(0,3,N)
        for a in range(3):
            m=ax==a
            if m.any():
                t=np.deg2rad(deg[m]); c,s=np.cos(t),np.sin(t)
                i,j=[(1,2),(0,2),(0,1)][a]
                sub=X[m][:,:,ch0:ch0+3].copy(); o=sub.copy()
                o[:,:,i]=c[:,None]*sub[:,:,i]-s[:,None]*sub[:,:,j]
                o[:,:,j]=s[:,None]*sub[:,:,i]+c[:,None]*sub[:,:,j]
                X[np.where(m)[0],:,ch0:ch0+3]=o
    X*=rng.uniform(0.9,1.1,(N,1,1)).astype(np.float32)
    X+=rng.normal(0,rng.uniform(0,0.1,(N,1,1)),X.shape).astype(np.float32)
    # sensor dropout 6%
    d=rng.random(N); X[d<0.03,:,3:]=0; X[(d>=0.03)&(d<0.06),:,:3]=0
    return X.astype(np.float32)
# ---------- training ----------
def macro_f1(y,p):
    f=[]
    for k in range(K):
        tp=((y==k)&(p==k)).sum(); fp=((y!=k)&(p==k)).sum(); fn=((y==k)&(p!=k)).sum()
        f.append(2*tp/max(2*tp+fp+fn,1) if (y==k).any() else np.nan)
    return float(np.nanmean(f)),f
def predict_logits(model,Xin,bs=4096):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(Xin),bs): out.append(model(torch.from_numpy(Xin[i:i+bs])))
    return torch.cat(out).numpy()
def fit_temperature(logits,y):
    best=(1e9,1.0)
    for T in np.arange(0.2,4.01,0.05):
        z=logits/T; z=z-z.max(1,keepdims=True); lp=z-np.log(np.exp(z).sum(1,keepdims=True))
        nll=-lp[np.arange(len(y)),y].mean()
        if nll<best[0]: best=(nll,T)
    return float(best[1])
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def train_model(kind,Xtr,ytr,Xva,yva,seed=0,epochs=14,log=print):
    """kind: 'lr','cnn','cnn_aug'. Xtr raw windows (N,100,6). returns dict(model,norm,T)"""
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    cw=np.bincount(ytr,minlength=K).astype(np.float64); cw=(cw.sum()/(K*np.maximum(cw,1)))**0.5
    cwt=torch.tensor(cw,dtype=torch.float32)
    if kind=='lr':
        Ftr=features(Xtr); Fva=features(Xva)
        mu=Ftr.mean(0); sd=Ftr.std(0)+1e-6
        norm={'mu':mu,'sd':sd}
        A=torch.from_numpy((Ftr-mu)/sd); B=(Fva-mu)/sd
        model=Linear(A.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)
        yt=torch.from_numpy(ytr)
        for ep in range(30):
            model.train(); perm=torch.randperm(len(A))
            for i in range(0,len(A),1024):
                b=perm[i:i+1024]; loss=F.cross_entropy(model(A[b]),yt[b],weight=cwt)
                opt.zero_grad(); loss.backward(); opt.step()
        lv=predict_logits(model,B.astype(np.float32))
        return {'model':model,'norm':norm,'T':fit_temperature(lv,yva),'kind':kind}
    mu=Xtr.reshape(-1,6).mean(0); sd=Xtr.reshape(-1,6).std(0)+1e-6
    norm={'mu':mu,'sd':sd}
    model=CNN(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-3)
    steps=epochs*((len(Xtr)+255)//256); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=4e-3,total_steps=steps)
    Xva_n=((Xva-mu)/sd).transpose(0,2,1).astype(np.float32).copy()
    best=(-1,None)
    for ep in range(epochs):
        model.train(); perm=rng.permutation(len(Xtr)); t0=time.time(); tl=0
        for i in range(0,len(perm),256):
            idx=perm[i:i+256]; xb=Xtr[idx]
            if kind=='cnn_aug': xb=augment(xb,rng)
            xb=torch.from_numpy(((xb-mu)/sd).transpose(0,2,1).astype(np.float32).copy()); yb=torch.from_numpy(ytr[idx])
            loss=F.cross_entropy(model(xb),yb,weight=cwt,label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl+=loss.item()*len(idx)
        lv=predict_logits(model,Xva_n); f1=macro_f1(yva,lv.argmax(1))[0]
        log(f'  {kind} ep{ep+1} loss {tl/len(Xtr):.3f} valF1 {f1:.3f} {time.time()-t0:.1f}s')
        if f1>best[0]: best=(f1,{k:v.clone() for k,v in model.state_dict().items()},ep+1)
    model.load_state_dict(best[1]); lv=predict_logits(model,Xva_n)
    return {'model':model,'norm':norm,'T':fit_temperature(lv,yva),'kind':kind,'best_epoch':best[2]}
def infer(m,X):
    if m['kind']=='lr':
        Fx=((features(X)-m['norm']['mu'])/m['norm']['sd']).astype(np.float32); return predict_logits(m['model'],Fx)
    Xn=((X-m['norm']['mu'])/m['norm']['sd']).transpose(0,2,1).astype(np.float32).copy(); return predict_logits(m['model'],Xn)
