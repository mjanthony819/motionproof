const puppeteer=require('puppeteer-core');
require("fs").mkdirSync("shots",{recursive:true});
(async()=>{
 const b=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--disable-gpu'],headless:'new'});
 const p=await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message)); p.on('console',m=>{if(m.type()==='error')errs.push(m.text())});
 let pass=0,fail=0; const ok=(c,m)=>{c?pass++:(fail++,console.log('FAIL',m))};
 const shots=process.argv[2]||'shots';
 await p.setViewport({width:1100,height:900});
 await p.goto('file://'+process.cwd()+'/index.html');
 await p.waitForFunction(()=>document.querySelector('#prog')&&document.querySelector('#prog').textContent.startsWith('Done'),{timeout:60000});
 ok(true,'bench ready'); console.log(await p.$eval('#prog',e=>e.textContent));
 await p.screenshot({path:shots+'/bench.png'});
 ok((await p.$$eval('#sum table tr',r=>r.length))===4,'summary rows');
 // threshold + click
 await p.$eval('#thr',e=>{e.value=0.9;e.dispatchEvent(new Event('input'))});
 await p.mouse.click(400,215);
 ok((await p.$eval('#det h2',e=>e.textContent)).startsWith('Window'),'detail');
 await p.screenshot({path:shots+'/bench_thr.png'});
 // save session
 await p.click('#save'); 
 // evidence
 await p.click('nav button[data-tab=evidence]'); await new Promise(r=>setTimeout(r,800)); await p.screenshot({path:shots+'/evidence.png',fullPage:true});
 ok((await p.$eval('#lat',e=>e.textContent)).includes('ms'),'latency measured');
 // stress
 await p.click('nav button[data-tab=stress]'); await p.select('#pk','thigh_rot'); await p.click('#go');
 await p.waitForFunction(()=>document.querySelector('#lr table'),{timeout:60000}); await p.screenshot({path:shots+'/stress.png',fullPage:true});
 console.log(await p.$$eval('#lr table tr',r=>r.map(x=>x.innerText.replace(/\s+/g,' ')).join('\n')));
 // import good + bad
 await p.click('nav button[data-tab=imp]');
 const inp=await p.$('#f'); await inp.uploadFile(process.cwd()+'/sample_import.csv'); await p.waitForSelector('#use');
 await p.screenshot({path:shots+'/import.png'});
 const fs=require('fs'); fs.writeFileSync('/tmp/bad.csv','a,b,c\n1,2,3\n'); await inp.uploadFile('/tmp/bad.csv'); await p.waitForSelector('.err');
 ok((await p.$eval('.err',e=>e.textContent)).includes('Missing columns'),'bad csv error shown');
 await inp.uploadFile(process.cwd()+'/sample_import.csv'); await p.waitForSelector('#use'); await p.click('#use');
 await p.waitForFunction(()=>document.querySelector('#prog')&&document.querySelector('#prog').textContent.startsWith('Done'),{timeout:60000});
 ok((await p.$eval('#src',e=>e.options[e.selectedIndex].text)).startsWith('Imported'),'imported source analysed'); await p.screenshot({path:shots+'/imported.png'});
 // history persists after reload
 await p.reload(); await p.click('nav button[data-tab=hist]'); ok((await p.$$eval('#view table tr',r=>r.length))>=2,'history persisted after reload');
 await p.screenshot({path:shots+'/history.png',fullPage:true});
 // mobile
 await p.setViewport({width:390,height:800}); await p.click('nav button[data-tab=bench]'); await p.waitForFunction(()=>document.querySelector('#prog')&&document.querySelector('#prog').textContent.startsWith('Done'),{timeout:60000}); await p.screenshot({path:shots+'/mobile.png'});
 ok(errs.length===0,'no console errors: '+errs.join('|'));
 console.log(pass+' passed, '+fail+' failed'); await b.close(); process.exit(fail?1:0);
})().catch(e=>{console.error(e);process.exit(1)});
