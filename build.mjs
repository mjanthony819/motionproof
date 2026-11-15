import fs from 'fs';
const r=p=>fs.readFileSync(new URL(p,import.meta.url),'utf8');
const data={models:JSON.parse(r('./data/models.json')),results:JSON.parse(r('./data/results.json')),streams:JSON.parse(r('./data/streams.json'))};
let h=r('./src/index.template.html');
// use function replacers so $ in content is literal
h=h.replace('/*CSS*/',()=>r('./src/style.css')).replace('/*DATA*/',()=>'window.MP_DATA='+JSON.stringify(data)+';').replace('/*CORE*/',()=>r('./src/core.js')).replace('/*APP*/',()=>r('./src/app.js'));

fs.writeFileSync(new URL('./index.html',import.meta.url),h);
console.log('index.html',(h.length/1024).toFixed(0)+' KB');
