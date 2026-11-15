const MP=require('../src/core.js'), fs=require('fs');
const models=JSON.parse(fs.readFileSync(__dirname+'/../data/models.json')).models;
const gold=JSON.parse(fs.readFileSync(__dirname+'/../data/golden.json'));
const pg=JSON.parse(fs.readFileSync(__dirname+'/../data/golden_perturb.json'));
let pass=0,fail=0; function ok(c,m){ if(c){pass++;} else {fail++; console.log('FAIL',m);} }
let maxd={lr:0,cnn:0,cnn_aug:0}, agree={lr:0,cnn:0,cnn_aug:0};
gold.windows.forEach((w,i)=>{
  const x=Float64Array.from(w.flat());
  for(const k of Object.keys(models)){
    const z=MP.logits(models[k],x); const g=gold.logits[k][i];
    let d=0; z.forEach((v,j)=>d=Math.max(d,Math.abs(v-g[j]))); maxd[k]=Math.max(maxd[k],d);
    const a=z.indexOf(Math.max(...z)), b=g.indexOf(Math.max(...g)); if(a===b) agree[k]++;
  }
});
for(const k of Object.keys(models)){ ok(maxd[k]<5e-3,`${k} logits max abs diff ${maxd[k]}`); ok(agree[k]===40,`${k} argmax agreement ${agree[k]}/40`); console.log(k,'max logit diff',maxd[k].toExponential(2),'argmax agree',agree[k]+'/40'); }
// perturbation parity
const base=gold.windows.slice(0,3).map(w=>Float64Array.from(w.flat()));
const specs={'gain:0.85':['gain',0.85],'gain:1.15':['gain',1.15],'thigh_rot:15':['thigh_rot',15],'thigh_rot:30':['thigh_rot',30],'back_rot:15':['back_rot',15],'back_rot:30':['back_rot',30],'drop_thigh':['drop_thigh',0],'drop_back':['drop_back',0],'stretch:0.8':['stretch',0.8],'stretch:1.25':['stretch',1.25],'clip:2':['clip',2],'clean':['clean',0]};
for(const [name,[kind,v]] of Object.entries(specs)){
  let d=0; base.forEach((x,i)=>{ const y=MP.perturbWindow(x,kind,v,null); const g=pg[name][i].flat(); y.forEach((q,j)=>d=Math.max(d,Math.abs(q-g[j]))); });
  ok(d<1e-5,`perturb ${name} max diff ${d}`);
}
// features sanity: dimension
ok(MP.features(base[0]).length===67,'feature dim');
// CSV validation
let r=MP.parseCSV('a,b\n1,2'); ok(!r.ok && r.errors[0].includes('Missing columns'),'missing cols');
r=MP.parseCSV(''); ok(!r.ok,'empty');
let rows=['back_x,back_y,back_z,thigh_x,thigh_y,thigh_z']; for(let i=0;i<300;i++) rows.push([9.8,0.1,0.2,9.7,0.3,0.1].join(','));
r=MP.parseCSV(rows.join('\n')); ok(r.ok && r.warnings.some(w=>w.includes('m/s^2')) && Math.abs(r.data[0]-9.8/9.80665)<1e-4,'m/s2 autoconvert');
rows=['timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z']; for(let i=0;i<1000;i++) rows.push([i*0.025,1,0,0,1,0,0].join(','));
r=MP.parseCSV(rows.join('\n')); ok(r.ok && Math.abs(r.info.estimatedHz-40)<0.5 && Math.abs(r.n-Math.floor(24.975*50))<=1,'resample 40Hz->50Hz n='+r.n);
rows=['back_x,back_y,back_z,thigh_x,thigh_y,thigh_z']; for(let i=0;i<150;i++) rows.push(i%10==0?'1,x,0,1,0,0':'1,0,0,1,0,0');
r=MP.parseCSV(rows.join('\n')); ok(r.ok && r.warnings.some(w=>w.includes('skipped')),'bad rows skipped');
rows=['back_x,back_y,back_z,thigh_x,thigh_y,thigh_z','1,0,0,1,0,0']; r=MP.parseCSV(rows.join('\n')); ok(!r.ok && r.errors[0].includes('need at least'),'too short');
// sample csv parses
const sc=fs.readFileSync(__dirname+'/../sample_import.csv','utf8'); r=MP.parseCSV(sc); ok(r.ok&&r.n===3000,'sample csv');
// streams decode + accuracy sanity on one stream with cnn_aug
const st=JSON.parse(fs.readFileSync(__dirname+'/../data/streams.json')); const s0=st.streams[2]; const data=MP.decodeStream(s0.b64); ok(data.length===s0.samples*6,'stream decode length');
const wins=MP.windowsOf(data,s0.samples,50); let c=0,t=0;
const lab=new Int8Array(s0.samples); s0.runs.forEach(([a,b,l])=>{for(let i=a;i<b;i++)lab[i]=l;});
wins.forEach(w=>{ const ls=lab.subarray(w.start,w.start+100); if(ls.every(v=>v===ls[0])){ t++; if(MP.predict(models.cnn_aug,w.data).label===ls[0]) c++; }});
console.log('stream S020 pure windows',t,'cnn_aug acc',(c/t).toFixed(3)); ok(c/t>0.6,'stream acc sanity');
console.log(`${pass} passed, ${fail} failed`); process.exit(fail?1:0);
