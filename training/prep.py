import pandas as pd, numpy as np, glob, os, json
os.makedirs('work',exist_ok=True)
files=sorted(glob.glob('work/harth/S*.csv'))
MAP={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,13:8,14:8,130:8,140:8}
NAMES=['walking','running','shuffling','stairs up','stairs down','standing','sitting','lying','cycling']
W=100; STRIDE=50
X=[];Y=[];S=[];T=[]
info={}
for f in files:
    sid=os.path.basename(f)[:-4]
    d=pd.read_csv(f)
    ts=pd.to_datetime(d['timestamp'])
    dt=ts.diff().dt.total_seconds().fillna(0.02).values
    gaps=int((dt>0.1).sum())
    a=d[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values.astype(np.float32)
    lab=d['label'].map(MAP).values
    n=len(d); info[sid]={'rows':n,'gaps':gaps,'minutes':round(n/50/60,1)}
    # segment index breaks at gaps
    brk=np.where(dt>0.1)[0]
    segs=np.split(np.arange(n),brk)
    cnt=0
    for sg in segs:
        for s in range(0,len(sg)-W+1,STRIDE):
            idx=sg[s:s+W]
            l=lab[idx]
            if (l==l[0]).all():   # pure window
                X.append(a[idx]);Y.append(int(l[0]));S.append(int(sid[1:]));T.append(float(idx[0])/50.0); cnt+=1
    info[sid]['windows']=cnt
X=np.stack(X);Y=np.array(Y);S=np.array(S);T=np.array(T)
np.savez_compressed('work/windows.npz',X=X,Y=Y,S=S,T=T)
print(X.shape, np.bincount(Y,minlength=9))
print(json.dumps(info)[:1500])
print({s:int((S==s).sum()) for s in sorted(set(S))})
json.dump({'names':NAMES,'info':info},open('work/info.json','w'))
