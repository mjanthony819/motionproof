const puppeteer=require('puppeteer-core'); const fs=require('fs');
const dur=[17.0,20.92,21.52,18.24,14.6,13.8]; const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const segs=process.argv.slice(2).map(Number);
(async()=>{
 const b=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--disable-gpu','--window-size=1280,760'],headless:'new'});
 const p=await b.newPage(); await p.setViewport({width:1280,height:720});
 await p.goto('file://'+process.cwd()+'/index.html');
 const ready=()=>p.waitForFunction(()=>document.querySelector('#prog')&&document.querySelector('#prog').textContent.startsWith('Done'),{timeout:60000});
 const tab=async t=>{await p.click('nav button[data-tab='+t+']')};
 const selSrc=async v=>{await p.select('#src',String(v)); await ready()};
 const scrollTo=async y=>{await p.evaluate(y=>window.scrollTo({top:y,behavior:'smooth'}),y)};
 const slide=async(from,to,ms)=>{const n=8;for(let i=0;i<=n;i++){const v=from+(to-from)*i/n;await p.$eval('#thr',(e,v)=>{e.value=v;e.dispatchEvent(new Event('input'))},v);await sleep(ms/n)}};
 await ready();
 for(const s of segs){
   // setup (not recorded)
   if(s===1||s===2){await tab('bench'); await ready(); await p.evaluate(()=>window.scrollTo(0,0));}
   if(s===3){await tab('evidence'); await sleep(900); await p.evaluate(()=>window.scrollTo(0,0));}
   if(s===4){await tab('stress'); await sleep(500); await p.evaluate(()=>window.scrollTo(0,0));}
   if(s===5){await tab('imp'); await sleep(300);}
   if(s===6){await tab('bench'); await ready(); await p.click('#save'); await tab('hist'); await sleep(300); await p.evaluate(()=>window.scrollTo(0,0));}
   const rec=await p.screencast({path:`../video/seg${s}.webm`}); const t0=Date.now(); const T=dur[s-1]*1000+500;
   const at=async(sec,fn)=>{const w=sec*1000-(Date.now()-t0); if(w>0) await sleep(w); await fn()};
   if(s===1){ await at(6,()=>scrollTo(260)); await at(12,()=>scrollTo(0)); }
   if(s===2){ await at(2,()=>p.mouse.click(640,480)); await at(5,()=>scrollTo(0)); await at(7,()=>slide(0.6,0.9,3500)); await at(14,async()=>{await p.select('#sm','3')}); await at(18,async()=>{await p.select('#src','1'); }); }
   if(s===3){ await at(6,()=>scrollTo(250)); await at(11,()=>scrollTo(640)); await at(15,()=>scrollTo(1000)); await at(19,()=>scrollTo(1300)); }
   if(s===4){ await at(5,()=>scrollTo(560)); await at(8,async()=>{await p.select('#pk','thigh_rot'); await p.click('#go')}); await at(13,()=>scrollTo(700)); }
   if(s===5){ await at(1,async()=>{const i=await p.$('#f'); await i.uploadFile(process.cwd()+'/sample_import.csv')}); await at(6,()=>scrollTo(150)); await at(9,async()=>{await p.click('#use')}); }
   if(s===6){ await at(3,()=>scrollTo(420)); await at(8,()=>scrollTo(700)); }
   const rest=T-(Date.now()-t0); if(rest>0) await sleep(rest);
   await rec.stop(); console.log('seg',s,'recorded',((Date.now()-t0)/1000).toFixed(1));
 }
 await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
