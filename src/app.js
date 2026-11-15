(function(){
'use strict';
var D=window.MP_DATA, M=window.MP, MODELS=D.models.models, RES=D.results, K=M.K, NAMES=M.NAMES;
var MK=['lr','cnn','cnn_aug'], MN={lr:'Feature logistic regression',cnn:'1D CNN',cnn_aug:'1D CNN + augmentation'}, MS={lr:'LR',cnn:'CNN',cnn_aug:'CNN-aug'};
var PAL=['#2a9d8f','#e4572e','#9b5de5','#f4a261','#e76f51','#4361ee','#8d99ae','#2b2d42','#e9c46a'];
var $=function(s,r){return (r||document).querySelector(s)};
var state={tab:'bench',src:null,preds:null,thr:0.6,smooth:1,sel:0,pert:{kind:'clean',v:0},imported:null,busy:false};
function el(t,a,h){var e=document.createElement(t);if(a)for(var k in a)e.setAttribute(k,a[k]);if(h!=null)e.innerHTML=h;return e}
function fmt(x,d){return x==null?'-':Number(x).toFixed(d==null?3:d)}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
// ---------- sources ----------
var streams=D.streams.streams.map(function(s){var data=M.decodeStream(s.b64),lab=new Int8Array(s.samples);s.runs.forEach(function(r){for(var i=r[0];i<r[1];i++)lab[i]=r[2]});return {name:'HARTH subject '+s.id+' (held out)',id:s.id,data:data,n:s.samples,labels:lab,kind:'demo'}});
function srcList(){var l=streams.slice();if(state.imported)l.push(state.imported);return l}
// ---------- inference ----------
function windowLabel(src,start){if(!src.labels)return -1;var l=src.labels[start];for(var i=start;i<start+100;i++)if(src.labels[i]!==l)return -1;return l}
function runAll(src,pert,cb){
  var wins=M.windowsOf(src.data,src.n,50),rng=M.mulberry(7),out={wins:wins.length,starts:wins.map(function(w){return w.start}),truth:wins.map(function(w){return windowLabel(src,w.start)}),m:{}};
  MK.forEach(function(k){out.m[k]={label:new Array(wins.length),conf:new Array(wins.length),probs:new Array(wins.length)}});
  var i=0,t0=performance.now();
  (function step(){
    var end=Math.min(i+20,wins.length);
    for(;i<end;i++){var x=pert&&pert.kind!=='clean'?M.perturbWindow(wins[i].data,pert.kind,pert.v,rng):wins[i].data;
      MK.forEach(function(k){var p=M.predict(MODELS[k],x);out.m[k].label[i]=p.label;out.m[k].conf[i]=p.conf;out.m[k].probs[i]=p.probs})}
    if(cb)cb(i/wins.length);
    if(i<wins.length)setTimeout(step,0);else{out.ms=(performance.now()-t0)/wins.length/MK.length;state._done&&state._done(out)}
  })();
}
function compute(src,pert){return new Promise(function(res){state._done=res;runAll(src,pert,function(f){var b=$('#prog');if(b)b.textContent='Running models... '+Math.round(f*100)+'%'})})}
function stats(P,k,thr,sm){
  var lab=sm>1?M.rollingMajority(P.m[k].label,P.m[k].conf,sm):P.m[k].label,n=0,cov=0,corr=0,ab=0,tot=0;
  for(var i=0;i<P.wins;i++){tot++;var abst=P.m[k].conf[i]<thr;if(abst)ab++;if(P.truth[i]>=0){n++;if(!abst){cov++;if(lab[i]===P.truth[i])corr++}}}
  var hasT=n>0,allc=0;for(i=0;i<P.wins;i++)if(P.truth[i]>=0&&lab[i]===P.truth[i])allc++;
  return {labels:lab,abstain:ab/tot,labeled:n,covered:cov,acc:hasT&&cov?corr/cov:null,accAll:hasT?allc/n:null,coverage:hasT?cov/n:null}
}
// ---------- drawing ----------
function setup(cv,h){var dpr=window.devicePixelRatio||1,w=cv.clientWidth||600;cv.width=w*dpr;cv.height=h*dpr;cv.style.height=h+'px';var c=cv.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);return {c:c,w:w,h:h}}
function legend(){return '<div class="legend">'+NAMES.map(function(n,i){return '<span><i style="background:'+PAL[i]+'"></i>'+n+'</span>'}).join('')+'<span><i style="background:#d5dbdf"></i>abstain / no label</span></div>'}
function drawTimeline(cv,src,P){
  var rows=['truth'].concat(MK),rh=24,top=80,H=top+rows.length*(rh+6)+14,s=setup(cv,H),c=s.c,w=s.w,lm=70,pw=w-lm-6;
  var tot=src.n;c.fillStyle='#fff';c.fillRect(0,0,w,H);
  // raw traces thigh_x (down) and back_x
  [[3,'#0b7a75','thigh x'],[0,'#e4572e','back x']].forEach(function(t,ti){c.strokeStyle=t[1];c.lineWidth=1;c.beginPath();var step=Math.max(1,Math.floor(tot/pw));
    for(var px=0;px<pw;px++){var a=Math.floor(px/pw*tot),mn=1e9,mx=-1e9;for(var j=a;j<Math.min(tot,a+step);j++){var v=src.data[j*6+t[0]];if(v<mn)mn=v;if(v>mx)mx=v}
      var y1=top/2-(mx*14)+ (ti? 0:0),y2=top/2-(mn*14);c.moveTo(lm+px,Math.max(2,Math.min(top-2,y1)));c.lineTo(lm+px,Math.max(2,Math.min(top-2,y2)))}c.stroke();
    c.fillStyle=t[1];c.font='11px system-ui';c.fillText(t[2],4,14+ti*12)});
  rows.forEach(function(k,ri){var y=top+ri*(rh+6);c.fillStyle='#12212b';c.font='12px system-ui';c.fillText(k==='truth'?'Truth':MS[k],4,y+16);
    for(var i=0;i<P.wins;i++){var x0=lm+P.starts[i]/tot*pw,x1=lm+(P.starts[i]+50)/tot*pw+0.6,col;
      if(k==='truth'){col=P.truth[i]>=0?PAL[P.truth[i]]:'#d5dbdf';}
      else{var st=P._st[k];col=P.m[k].conf[i]<state.thr?'#d5dbdf':PAL[st.labels[i]]}
      c.fillStyle=col;c.fillRect(x0,y,Math.max(1,x1-x0),rh)}});
  var sx=lm+P.starts[state.sel]/tot*pw;c.strokeStyle='#000';c.lineWidth=2;c.strokeRect(sx,top-4,100/tot*pw,rows.length*(rh+6)+2);
  cv._geo={lm:lm,pw:pw,tot:tot}
}
function drawHeat(cv,cm,norm,names){
  var n=cm.length,s=setup(cv,Math.max(260,Math.min(cv.clientWidth,520))),c=s.c,w=s.w,m=84,cell=(w-m-6)/n;
  c.font='11px system-ui';var mx=0;cm.forEach(function(r){r.forEach(function(v){mx=Math.max(mx,v)})});
  for(var i=0;i<n;i++){var rs=cm[i].reduce(function(a,b){return a+b},0)||1;c.fillStyle='#12212b';c.textAlign='right';c.fillText(names[i],m-4,m/2+ i*cell+cell/2+4-m/2+14);
    for(var j=0;j<n;j++){var v=norm?cm[i][j]/rs:cm[i][j]/mx;var a=Math.pow(v,norm?0.7:0.4);c.fillStyle='rgba(11,122,117,'+a+')';c.fillRect(m+j*cell,28+i*cell,cell-1,cell-1);
      if(norm){c.fillStyle=a>0.5?'#fff':'#12212b';c.textAlign='center';c.fillText(Math.round(cm[i][j]/rs*100),m+j*cell+cell/2,28+i*cell+cell/2+4)}}}
  c.textAlign='center';c.fillStyle='#5b6b76';for(j=0;j<n;j++){c.save();c.translate(m+j*cell+cell/2,24);c.rotate(-0.5);c.textAlign='left';c.fillText(names[j],0,0);c.restore()}
  cv.style.height=(28+n*cell+4)+'px'
}
function drawLines(cv,series,xl,yl,xr,diag){
  var s=setup(cv,220),c=s.c,w=s.w,h=s.h,l=40,b=28,t=8,r=10;
  c.strokeStyle='#dbe3e8';c.fillStyle='#5b6b76';c.font='11px system-ui';
  for(var i=0;i<=4;i++){var y=t+(h-t-b)*i/4;c.beginPath();c.moveTo(l,y);c.lineTo(w-r,y);c.stroke();c.textAlign='right';c.fillText((xr[3]-(xr[3]-xr[2])*i/4).toFixed(2),l-4,y+4)}
  for(i=0;i<=4;i++){var x=l+(w-l-r)*i/4;c.textAlign='center';c.fillText((xr[0]+(xr[1]-xr[0])*i/4).toFixed(2),x,h-b+14)}
  c.fillText(xl,l+(w-l-r)/2,h-4);
  if(diag){c.setLineDash([4,4]);c.beginPath();c.moveTo(l,h-b);c.lineTo(w-r,t);c.stroke();c.setLineDash([])}
  var X=function(v){return l+(v-xr[0])/(xr[1]-xr[0])*(w-l-r)},Y=function(v){return h-b-(v-xr[2])/(xr[3]-xr[2])*(h-t-b)};
  series.forEach(function(sr,si){c.strokeStyle=sr.col;c.fillStyle=sr.col;c.lineWidth=2;c.beginPath();sr.pts.forEach(function(p,i){i?c.lineTo(X(p[0]),Y(p[1])):c.moveTo(X(p[0]),Y(p[1]))});c.stroke();sr.pts.forEach(function(p){c.beginPath();c.arc(X(p[0]),Y(p[1]),3,0,7);c.fill()});c.fillRect(l+8+si*120,t+2,10,10);c.fillStyle='#12212b';c.textAlign='left';c.fillText(sr.name,l+22+si*120,t+11)})
}
// ---------- views ----------
var V=$('#view');
function render(){
  document.querySelectorAll('nav button').forEach(function(b){b.className=b.dataset.tab===state.tab?'on':''});
  V.innerHTML='';({bench:vBench,evidence:vEvidence,stress:vStress,imp:vImport,hist:vHistory}[state.tab])()
}
function sourceSelect(onchange){
  var s=el('select',{id:'src'});srcList().forEach(function(x,i){s.appendChild(el('option',{value:i},esc(x.name)))});
  s.value=Math.max(0,srcList().indexOf(state.src));s.onchange=function(){state.src=srcList()[s.value];state.preds=null;onchange()};return s}
function ensure(){
  if(!state.src)state.src=streams[2];
  return state.preds&&state.preds._src===state.src&&state.preds._pert===JSON.stringify(state.pert)?Promise.resolve(state.preds):compute(state.src,state.pert).then(function(P){P._src=state.src;P._pert=JSON.stringify(state.pert);state.preds=P;return P})
}
function vBench(){
  state.pert={kind:'clean',v:0};
  var card=el('div',{class:'card'});
  card.innerHTML='<h2>Live bench</h2><p class="mut small">Pick a recording. All three trained models run in your browser on every 2-second window (50% overlap). The top two rows of colour show the truth label from the dataset and each model prediction. Raise the abstain threshold to see where models are unsure.</p>';
  var row=el('div',{class:'row'});row.appendChild(sourceSelect(render));
  row.appendChild(el('label',{class:'small'},'Abstain below confidence <b id="tv">'+state.thr.toFixed(2)+'</b> <input type="range" id="thr" min="0" max="0.95" step="0.05" value="'+state.thr+'">'));
  row.appendChild(el('label',{class:'small'},'Smoothing <select id="sm"><option value="1">none</option><option value="3">majority of 3 windows</option><option value="5">majority of 5 windows</option></select>'));
  card.appendChild(row);card.appendChild(el('div',{id:'prog',class:'mut small'},''));
  card.appendChild(el('div',{html:'',id:'lg'},legend()));
  var cv=el('canvas',{id:'tl'});card.appendChild(cv);V.appendChild(card);
  var det=el('div',{class:'card',id:'det'});V.appendChild(det);var sum=el('div',{class:'card',id:'sum'});V.appendChild(sum);
  $('#sm').value=state.smooth;
  ensure().then(function(P){if(state.tab!=='bench'||!$('#tl')||$('#tl')!==cv)return;$('#prog').textContent='Done. '+P.wins+' windows, '+fmt(P.ms,2)+' ms per model per window in this browser.';redraw(P);
    $('#thr').oninput=function(){state.thr=+this.value;$('#tv').textContent=state.thr.toFixed(2);redraw(P)};
    $('#sm').onchange=function(){state.smooth=+this.value;redraw(P)};
    cv.onclick=function(e){var g=cv._geo,r=cv.getBoundingClientRect(),x=(e.clientX-r.left-g.lm)/g.pw*g.tot,best=0,bd=1e9;P.starts.forEach(function(s,i){var d=Math.abs(s+50-x);if(d<bd){bd=d;best=i}});state.sel=best;redraw(P)}})
}
function redraw(P){
  P._st={};MK.forEach(function(k){P._st[k]=stats(P,k,state.thr,state.smooth)});
  if(state.sel>=P.wins)state.sel=0;
  drawTimeline($('#tl'),state.src,P);
  var d='<h2>Window '+state.sel+' <span class="mut small">('+(P.starts[state.sel]/50).toFixed(1)+' s to '+((P.starts[state.sel]+100)/50).toFixed(1)+' s)</span></h2><p class="small mut">Truth: <b>'+(P.truth[state.sel]>=0?NAMES[P.truth[state.sel]]:(state.src.labels?'mixed activity in window':'unlabelled'))+'</b>. Click the timeline to inspect another window.</p><div class="grid2">';
  MK.forEach(function(k){var pr=P.m[k].probs[state.sel],ord=pr.map(function(p,i){return [p,i]}).sort(function(a,b){return b[0]-a[0]}).slice(0,4);
    d+='<div><b>'+MN[k]+'</b>'+(P.m[k].conf[state.sel]<state.thr?' <span class="pill" style="background:#eee;color:#555">abstains</span>':'')+ord.map(function(o){return '<div class="pb"><span>'+NAMES[o[1]]+'</span><div class="bar"><b style="width:'+(o[0]*100)+'%;background:'+PAL[o[1]]+'"></b></div><span>'+Math.round(o[0]*100)+'%</span></div>'}).join('')+'</div>'});
  $('#det').innerHTML=d+'</div>';
  var t='<h2>Scores on this recording</h2><table><tr><th>Model</th><th>Accuracy on answered</th><th>Answered</th><th>Accuracy if forced</th><th>Abstained</th></tr>';
  MK.forEach(function(k){var s=P._st[k];t+='<tr><td>'+MN[k]+'</td><td>'+(s.acc==null?'-':fmt(s.acc*100,1)+'%')+'</td><td>'+(s.coverage==null?'-':fmt(s.coverage*100,0)+'%')+'</td><td>'+(s.accAll==null?'-':fmt(s.accAll*100,1)+'%')+'</td><td>'+fmt(s.abstain*100,0)+'%</td></tr>'});
  t+='</table><p class="small mut">Only windows with one clean activity label are scored. This is one recording from a held-out subject, so it is an illustration, not the headline result. The Evidence tab has the full subject-held-out evaluation.</p><div class="row"><button class="btn" id="save">Save session</button><button class="btn alt" id="csv">Download predictions CSV</button></div>';
  $('#sum').innerHTML=t;
  $('#save').onclick=function(){saveSession(P)};$('#csv').onclick=function(){download('motionproof_predictions.csv',predCSV(P),'text/csv')}
}
function predCSV(P){var r=['start_s,end_s,truth,'+MK.map(function(k){return k+'_label,'+k+'_confidence'}).join(',')];for(var i=0;i<P.wins;i++)r.push([(P.starts[i]/50).toFixed(2),((P.starts[i]+100)/50).toFixed(2),P.truth[i]>=0?NAMES[P.truth[i]]:''].concat(MK.map(function(k){return NAMES[P._st[k].labels[i]]+','+P.m[k].conf[i].toFixed(4)})).join(','));return r.join('\n')}
function download(name,text,type){var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:type||'text/plain'}));a.download=name;document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},500)}
// ---- sessions (localStorage; falls back to memory) ----
var mem=[];function loadS(){try{return JSON.parse(localStorage.getItem('motionproof.sessions')||'[]')}catch(e){return mem}}
function storeS(l){try{localStorage.setItem('motionproof.sessions',JSON.stringify(l))}catch(e){mem=l}}
function saveSession(P){var l=loadS(),o={id:Date.now(),when:new Date().toISOString(),source:state.src.name,thr:state.thr,smooth:state.smooth,windows:P.wins,models:{}};MK.forEach(function(k){var s=P._st[k];o.models[k]={acc:s.acc,coverage:s.coverage,accAll:s.accAll,abstain:s.abstain}});l.unshift(o);storeS(l.slice(0,30));$('#save').textContent='Saved ('+Math.min(l.length,30)+')'}
function vHistory(){
  var l=loadS(),c=el('div',{class:'card'});c.innerHTML='<h2>Session history</h2><p class="small mut">Sessions are stored only in this browser (localStorage). Nothing is uploaded.</p>';
  if(!l.length)c.innerHTML+='<p>No saved sessions yet. Open the Bench tab and press Save session.</p>';
  else{var t='<table><tr><th>When</th><th>Source</th><th>Abstain</th><th>LR</th><th>CNN</th><th>CNN-aug</th></tr>';l.forEach(function(s){t+='<tr><td>'+esc(new Date(s.when).toLocaleString())+'</td><td>'+esc(s.source)+'</td><td>'+s.thr.toFixed(2)+'/sm'+s.smooth+'</td>'+MK.map(function(k){var m=s.models[k];return '<td>'+(m.acc==null?'-':fmt(m.acc*100,1)+'%')+' @ '+(m.coverage==null?'-':fmt(m.coverage*100,0)+'%')+'</td>'}).join('')+'</tr>'});c.innerHTML+=t+'</table><p class="small mut">Each model cell shows accuracy on answered windows @ share answered.</p><div class="row"><button class="btn alt" id="ex">Export JSON</button><button class="btn alt" id="clr">Clear history</button></div>'}
  V.appendChild(c);if($('#ex')){$('#ex').onclick=function(){download('motionproof_sessions.json',JSON.stringify(l,null,2),'application/json')};$('#clr').onclick=function(){storeS([]);render()}}
  var mc=el('div',{class:'card'});mc.innerHTML='<h2>Model card</h2><pre style="white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;margin:0">'+esc(modelCard())+'</pre><p><button class="btn" id="mcd">Download model card</button> <button class="btn alt" id="mx">Download experiment export (JSON)</button></p>';V.appendChild(mc);
  $('#mcd').onclick=function(){download('MODEL_CARD.md',modelCard())};$('#mx').onclick=function(){download('motionproof_experiments.json',JSON.stringify(RES,null,1),'application/json')}
}
function modelCard(){
  var d=RES.data,cv=function(k){return RES.models[k].cv},sh=function(k){return RES.models[k].show};
  return '# MotionProof model card\n\nTask: classify 2-second windows (50 Hz, two tri-axial accelerometers: lower back and right thigh) into 9 everyday activities: '+NAMES.join(', ')+'.\n\nIntended use: a transparent benchmark and teaching tool for comparing activity-recognition models, checking calibration and stress robustness. Not a medical device. Not for diagnosis, fall detection, clinical gait assessment or monitoring of a person without their consent.\n\nData: HARTH (Logacjov et al., 2021), UCI ML Repository dataset 779, CC BY 4.0. '+d.subjects+' adults, about '+d.hours+' hours of free-living recordings with camera-annotated labels. The four cycling codes (sitting/standing, active/inactive) are merged into one class. Windows with mixed labels are dropped. '+d.windows.toLocaleString()+' windows.\n\nEvaluation: subject-held-out. Headline numbers are 5-fold cross-validation over subjects (every test window comes from a person the model never trained on). Models shipped in this app were trained separately on 15 subjects, with 3 for validation (early stopping and temperature scaling) and 4 held out (subjects 6, 13, 20, 28); 3 seeds were trained for each model to show variation.\n\nResults (macro-F1, subject-held-out CV): LR '+fmt(cv('lr').macro_f1,3)+', CNN '+fmt(cv('cnn').macro_f1,3)+', CNN+augmentation '+fmt(cv('cnn_aug').macro_f1,3)+'. On the 4 held-out showcase subjects (mean of 3 seeds): LR '+fmt(sh('lr').f1_mean,3)+', CNN '+fmt(sh('cnn').f1_mean,3)+', CNN+aug '+fmt(sh('cnn_aug').f1_mean,3)+'.\n\nKnown weaknesses: shuffling, stairs up and stairs down are the hardest classes (few training windows, similar to walking). Models degrade sharply when one sensor is lost or rotated far from the training placement. Results are for healthy adults wearing sensors in this exact placement; other populations, placements and devices are untested.\n\nHow it was built: models were trained with PyTorch on CPU, exported as plain JSON weights and re-implemented in JavaScript; the test suite checks that JavaScript outputs match the PyTorch outputs. Built with AI assistance (see README).\n\nCitation: Logacjov A., Bach K., Kongsvold A., Bardstu H.B., Mork P.J. HARTH: a human activity recognition dataset for machine learning. Sensors 21(23):7853 (2021).\n'
}
function vEvidence(){
  var d=RES.data,c=el('div',{class:'card'});
  c.innerHTML='<h2>Evidence: how well do the models work on new people?</h2><p class="small mut">Data: HARTH, '+d.subjects+' adults, about '+d.hours+' h of free-living accelerometer data (CC BY 4.0). Every number below is measured on people the model did not train on. Cross-validation (CV) uses 5 subject-grouped folds, so each of the '+d.windows.toLocaleString()+' windows is predicted once by a model that never saw that person.</p>';
  var t='<table><tr><th>Model</th><th>Macro-F1, pooled over all folds</th><th>Macro-F1, mean of 5 folds (sd)</th><th>Accuracy, pooled</th><th>Calibration error (ECE)</th><th>Macro-F1, 4 held-out subjects (mean of 3 seeds, +/- sd)</th></tr>';
  MK.forEach(function(k){var m=RES.models[k];t+='<tr><td>'+MN[k]+'</td><td><b>'+fmt(m.cv.macro_f1)+'</b></td><td>'+fmt(m.cv.fold_f1_mean)+' ('+fmt(m.cv.fold_f1_std)+')</td><td>'+fmt(m.cv.acc)+'</td><td>'+fmt(m.cv.ece)+'</td><td>'+fmt(m.show.f1_mean)+' +/- '+fmt(m.show.f1_std)+'</td></tr>'});
  c.innerHTML+=t+'</table><p class="small">Reading it honestly: the augmented CNN is the best model, but the plain CNN is not better than the simple logistic regression on pooled cross-validation. Macro-F1 gives rare activities (stairs, shuffling) the same weight as common ones, so it is lower than accuracy. Pooled means every out-of-fold window is scored together; mean of folds averages the five fold scores. They differ because folds have different people and class mixes (the plain CNN looks better by fold average than pooled). The last column is a separate split: four subjects held out, with a model trained on the other 18, repeated with three random seeds. Differences between models are about the size of the spread between folds, so treat small gaps as ties.</p>';
  V.appendChild(c);
  var c2=el('div',{class:'card'});c2.innerHTML='<h2>Per-activity F1 (cross-validation)</h2><canvas id="pc"></canvas>';V.appendChild(c2);
  var g=el('div',{class:'grid2'});var c3=el('div',{class:'card'});c3.innerHTML='<h2>Confusion matrix</h2><div class="row"><select id="cmm">'+MK.map(function(k){return '<option value="'+k+'">'+MN[k]+'</option>'}).join('')+'</select><label class="small"><input type="checkbox" id="cmn" checked> row-normalised %</label></div><canvas id="cm"></canvas><p class="small mut">Rows are the true activity, columns the prediction.</p>';
  var c4=el('div',{class:'card'});c4.innerHTML='<h2>Calibration</h2><canvas id="rel"></canvas><p class="small mut">Dashed line is perfect calibration. After temperature scaling fitted on validation subjects.</p><h2>Answer fewer, be right more</h2><canvas id="cov"></canvas><p class="small mut">Horizontal: share of windows the model answers when it abstains below a confidence threshold. Vertical: accuracy on those windows.</p>';
  g.appendChild(c3);g.appendChild(c4);V.appendChild(g);
  var c5=el('div',{class:'card'});c5.innerHTML='<h2>Does it work the same for everyone?</h2><p class="small mut">Macro-F1 per subject (out-of-fold). The weakest person is the realistic worst case.</p><div id="sj"></div><p class="small mut" id="lat"></p>';V.appendChild(c5);
  // per-class bars
  setTimeout(function(){
    var cv=$('#pc'),s=setup(cv,250),c=s.c,w=s.w,l=30,b=44,n=K,gw=(w-l-4)/n;c.font='11px system-ui';
    for(var i=0;i<n;i++){MK.forEach(function(k,ki){var v=RES.models[k].cv.per_class_f1[i],bw=gw/4,x=l+i*gw+ki*bw+2,hh=(250-b-8)*v;c.fillStyle=['#8d99ae','#4361ee','#0b7a75'][ki];c.fillRect(x,250-b-hh,bw-1,hh)});c.fillStyle='#12212b';c.save();c.translate(l+i*gw+gw/2,250-b+12);c.rotate(0);c.textAlign='center';c.fillText(NAMES[i],0,0);c.restore()}
    c.textAlign='left';[['#8d99ae','LR'],['#4361ee','CNN'],['#0b7a75','CNN-aug']].forEach(function(q,i){c.fillStyle=q[0];c.fillRect(l+i*70,2,10,10);c.fillStyle='#12212b';c.fillText(q[1],l+14+i*70,11)});
    c.fillStyle='#5b6b76';c.textAlign='right';[0,0.5,1].forEach(function(v){c.fillText(v.toFixed(1),l-3,250-b-(250-b-8)*v+4)});
    function drawCM(){var k=$('#cmm').value;drawHeat($('#cm'),RES.models[k].cv.cm,$('#cmn').checked,NAMES)}
    $('#cmm').onchange=drawCM;$('#cmn').onchange=drawCM;drawCM();
    drawLines($('#rel'),MK.map(function(k,i){return {name:MS[k],col:['#8d99ae','#4361ee','#0b7a75'][i],pts:RES.models[k].cv.reliability.map(function(r){return [r[0],r[1]]})}}),'confidence',' ',[0.1,1,0,1],true);
    drawLines($('#cov'),MK.map(function(k,i){return {name:MS[k],col:['#8d99ae','#4361ee','#0b7a75'][i],pts:RES.models[k].cv.coverage.map(function(r){return [r[1],r[2]]})}}),'share answered',' ',[0.5,1,0.8,1],false);
    var sj='<table><tr><th>Subject</th>'+MK.map(function(k){return '<th>'+MS[k]+'</th>'}).join('')+'</tr>';Object.keys(RES.models.cnn_aug.cv.per_subject_f1).forEach(function(s){sj+='<tr><td>S'+(('00'+s).slice(-3))+'</td>'+MK.map(function(k){var v=RES.models[k].cv.per_subject_f1[s];return '<td style="background:rgba(11,122,117,'+Math.max(0,(v-0.5)*1.6)+');color:'+(v>0.8?'#fff':'#12212b')+'">'+fmt(v,2)+'</td>'}).join('')+'</tr>'});$('#sj').innerHTML=sj+'</table>';
    // latency
    var wins=M.windowsOf(streams[0].data,streams[0].n,50).slice(0,60),txt='In-browser latency per 2 s window: ';
    MK.forEach(function(k){var t0=performance.now();wins.forEach(function(w){M.predict(MODELS[k],w.data)});txt+=MS[k]+' '+((performance.now()-t0)/wins.length).toFixed(2)+' ms  '});
    $('#lat').textContent=txt+'(measured just now on this device).'
  },0)
}
function vStress(){
  var c=el('div',{class:'card'});c.innerHTML='<h2>Stress lab: what breaks each model?</h2><p class="small mut">Real sensors drift, rotate, clip and fail. The first table is the measured result on the 4 held-out subjects (mean of 3 seeds, macro-F1). The live panel below applies the same kind of damage to the recording you choose.</p>';
  var conds=['clean','noise:0.05','noise:0.15','gain:0.85','gain:1.15','thigh_rot:15','thigh_rot:30','back_rot:15','back_rot:30','drop_thigh','drop_back','stretch:0.8','stretch:1.25','clip:2'];
  var lab={'clean':'No damage','noise:0.05':'Noise sd 0.05 g','noise:0.15':'Noise sd 0.15 g','gain:0.85':'Gain x0.85','gain:1.15':'Gain x1.15','thigh_rot:15':'Thigh sensor rotated 15 deg','thigh_rot:30':'Thigh sensor rotated 30 deg','back_rot:15':'Back sensor rotated 15 deg','back_rot:30':'Back sensor rotated 30 deg','drop_thigh':'Thigh sensor lost','drop_back':'Back sensor lost','stretch:0.8':'Time stretch x0.8','stretch:1.25':'Time stretch x1.25','clip:2':'Clipped at 2 g'};
  var t='<table class="heat"><tr><th>Condition</th>'+MK.map(function(k){return '<th>'+MS[k]+'</th>'}).join('')+'</tr>';
  conds.forEach(function(q){t+='<tr><td style="text-align:left">'+lab[q]+'</td>'+MK.map(function(k){var v=RES.models[k].stress[q].f1_mean;return '<td style="background:rgba(228,87,46,'+(0.85*(1-v)).toFixed(2)+');color:#12212b">'+fmt(v,2)+'</td>'}).join('')+'</tr>'});
  c.innerHTML+=t+'</table><p class="small">Takeaways: training with rotation, noise and sensor-dropout augmentation helped most for noise, rotation and a lost back sensor. Losing the thigh sensor hurts every model badly because the thigh carries the posture signal that separates sitting from standing. Confidence stays high while accuracy falls, so abstaining on confidence alone does not catch sensor failure.</p>';V.appendChild(c);
  var lv=el('div',{class:'card'});lv.innerHTML='<h2>Live damage on a recording</h2>';var row=el('div',{class:'row'});row.appendChild(sourceSelect(function(){render()}));
  var ps=el('select',{id:'pk'});Object.keys(M.PERTURBATIONS).forEach(function(k){ps.appendChild(el('option',{value:k},M.PERTURBATIONS[k].label))});row.appendChild(ps);
  row.appendChild(el('label',{class:'small'},'Strength <b id="pv"></b> <input type="range" id="pr">'));row.appendChild(el('button',{class:'btn',id:'go'},'Run'));lv.appendChild(row);lv.appendChild(el('div',{id:'prog',class:'mut small'},''));lv.appendChild(el('div',{id:'lr'}));V.appendChild(lv);
  function setR(){var p=M.PERTURBATIONS[ps.value],r=$('#pr');r.min=p.min;r.max=p.max;r.step=p.step||1;r.value=p.def;r.disabled=p.min===p.max;$('#pv').textContent=p.min===p.max?'(fixed)':p.def}
  ps.onchange=setR;$('#pr').oninput=function(){$('#pv').textContent=this.value};setR();
  $('#go').onclick=function(){var pert={kind:ps.value,v:+$('#pr').value};if(!state.src)state.src=streams[2];$('#go').disabled=true;
    compute(state.src,{kind:'clean',v:0}).then(function(A){return compute(state.src,pert).then(function(B){return [A,B]})}).then(function(r){if(!$('#go'))return;var A=r[0],B=r[1];$('#go').disabled=false;$('#prog').textContent='';
      var h='<table><tr><th>Model</th><th>Accuracy clean</th><th>Accuracy damaged</th><th>Predictions changed</th><th>Mean confidence damaged</th></tr>';
      MK.forEach(function(k){var n=0,ca=0,cb=0,ch=0,cf=0,tot=A.wins;for(var i=0;i<tot;i++){if(A.truth[i]>=0){n++;if(A.m[k].label[i]===A.truth[i])ca++;if(B.m[k].label[i]===A.truth[i])cb++}if(A.m[k].label[i]!==B.m[k].label[i])ch++;cf+=B.m[k].conf[i]}
        h+='<tr><td>'+MN[k]+'</td><td>'+(n?fmt(ca/n*100,1)+'%':'-')+'</td><td>'+(n?fmt(cb/n*100,1)+'%':'-')+'</td><td>'+fmt(ch/tot*100,0)+'%</td><td>'+fmt(cf/tot*100,0)+'%</td></tr>'});
      $('#lr').innerHTML=h+'</table><p class="small mut">Same recording, same windows, damage applied before the models see the data (seeded noise, so results repeat). Accuracy uses windows with a single clean activity label.</p>'})};
}
function vImport(){
  var c=el('div',{class:'card'});c.innerHTML='<h2>Bring your own recording</h2><p class="small mut">Upload a CSV with columns <code>back_x,back_y,back_z,thigh_x,thigh_y,thigh_z</code> in g (optional <code>timestamp</code> column, seconds or ISO time). The file is read in your browser and never uploaded. A lower-back and a right-thigh accelerometer at about 50 Hz match what the models were trained on; other placements will give unreliable answers.</p><div class="row"><input type="file" id="f" accept=".csv,.txt,text/csv"><button class="btn alt" id="sample">Download a sample CSV</button></div><div id="rep"></div>';V.appendChild(c);
  $('#sample').onclick=function(){var d=streams[0].data,r=['back_x,back_y,back_z,thigh_x,thigh_y,thigh_z'];for(var i=0;i<3000;i++){var q=[];for(var j=0;j<6;j++)q.push(d[i*6+j].toFixed(3));r.push(q.join(','))}download('motionproof_sample.csv',r.join('\n'),'text/csv')};
  $('#f').onchange=function(){var f=this.files[0];if(!f)return;if(f.size>60e6){$('#rep').innerHTML='<div class="err">File is larger than 60 MB. Trim it to the part you want to analyse.</div>';return}
    var rd=new FileReader();rd.onload=function(){var r=M.parseCSV(String(rd.result));var h='';r.errors.forEach(function(e){h+='<div class="err">'+esc(e)+'</div>'});r.warnings.forEach(function(e){h+='<div class="warn">'+esc(e)+'</div>'});
      if(r.ok){h='<div class="okb">Read '+r.info.rows+' rows ('+r.info.seconds+' s at 50 Hz). Mean magnitude per sensor: '+r.info.meanMagnitude.join(' / ')+' g.</div>'+h+'<button class="btn" id="use">Analyse this recording</button>';
        $('#rep').innerHTML=h;$('#use').onclick=function(){state.imported={name:'Imported: '+f.name,data:r.data,n:r.n,labels:null,kind:'import'};state.src=state.imported;state.preds=null;state.tab='bench';render()}}
      else $('#rep').innerHTML=h};rd.readAsText(f)}
}
document.querySelectorAll('nav button').forEach(function(b){b.onclick=function(){state.tab=b.dataset.tab;render()}});
window.MP_APP={state:state,render:render,MK:MK};
render();
})();
