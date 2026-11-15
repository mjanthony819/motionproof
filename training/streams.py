import pandas as pd, numpy as np, json, base64, glob
names=['walking','running','shuffling','stairs up','stairs down','standing','sitting','lying','cycling']
MAP={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,13:8,14:8,130:8,140:8}
out=[]
for sid in [6,13,20,28]:
    d=pd.read_csv(f'work/harth/S{sid:03d}.csv')
    ts=pd.to_datetime(d['timestamp']); dt=ts.diff().dt.total_seconds().fillna(0.02).values
    lab=d['label'].map(MAP).values; a=d[['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']].values
    L=12000; best=(-1,0)
    for st in range(0,len(d)-L,250):
        if (dt[st+1:st+L]>0.1).any(): continue
        l=lab[st:st+L]; c=np.bincount(l,minlength=9)/L; ent=-(c[c>0]*np.log(c[c>0])).sum()
        # prefer many classes, each at least 2%
        sc=(c>0.03).sum()+ent*0.3
        if sc>best[0]: best=(sc,st)
    st=best[1]; seg=a[st:st+L]; l=lab[st:st+L]
    q=np.clip(np.round(seg*1000),-32767,32767).astype('<i2')
    runs=[];s0=0
    for i in range(1,L+1):
        if i==L or l[i]!=l[s0]: runs.append([s0,i,int(l[s0])]); s0=i
    out.append({'id':f'S{sid:03d}','samples':L,'fs':50,'b64':base64.b64encode(q.tobytes()).decode(),'runs':runs,'classes':sorted(set(int(x) for x in l))})
    print(sid,st,best[0],out[-1]['classes'],len(runs))
json.dump({'names':names,'streams':out,'note':'4-minute excerpts from HARTH subjects held out from the shipped models; accelerometer values in milli-g (int16)'},open('data/streams.json','w'))
# sample CSV for import
seg=np.frombuffer(base64.b64decode(out[0]['b64']),dtype='<i2').reshape(-1,6)/1000
with open('sample_import.csv','w') as f:
    f.write('back_x,back_y,back_z,thigh_x,thigh_y,thigh_z\n')
    for r in seg[:3000]: f.write(','.join(f'{v:.3f}' for v in r)+'\n')
