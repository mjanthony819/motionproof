"""Merge analysis outputs into data/results.json (read by the app)."""
import json, numpy as np
R=json.load(open('work/out/results_cv_show.json'))
info=json.load(open('work/info.json'))
out={'names':R['names'],'models':{}}
for k,v in R['models'].items():
    cv=v['cv']; sh=v['show']
    out['models'][k]={'cv':{x:cv[x] for x in ['macro_f1','acc','per_class_f1','ece','nll','reliability','cm','coverage','per_fold','temps','per_subject_f1','fold_f1_mean','fold_f1_std']},
      'show':{'f1_seeds':v['show_f1_seeds'],'acc_seeds':[s['acc'] for s in sh],'f1_mean':float(np.mean(v['show_f1_seeds'])),'f1_std':float(np.std(v['show_f1_seeds'])),'ece_seeds':[s['ece'] for s in sh],'cm_seed0':sh[0]['cm'],'per_class_f1_seed0':sh[0]['per_class_f1']},
      'stress':json.load(open(f'work/out/stress_{k}.json'))}
tot=sum(i['rows'] for i in info['info'].values())
out['data']={'subjects':22,'raw_rows':tot,'hours':round(tot/50/3600,1),'windows':118495,'class_windows':[21233,5675,1781,1096,838,11654,57854,8431,9933]}
out['latency_py_ms']=None
json.dump(out,open('data/results.json','w'),separators=(',',':'))
