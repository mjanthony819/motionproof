from playwright.sync_api import sync_playwright
from pathlib import Path
import json
P=Path(__file__).resolve().parent
with sync_playwright() as q:
 b=q.chromium.launch(headless=True,args=['--no-sandbox'])
 c=b.new_context(viewport={'width':1280,'height':1000},accept_downloads=True)
 p=c.new_page();errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
 p.goto((P/'index.html').as_uri())
 # 1 default tab is analyse with import card
 assert p.locator('nav button.on').inner_text().startswith('Analyse')
 assert p.locator('#f').count()==1
 assert p.locator('#tryme').count()==1
 # 2 built-in sample analysis
 p.locator('#tryme').click()
 p.wait_for_function("document.querySelector('#prog') && document.querySelector('#prog').textContent.startsWith('Done')",timeout=120000)
 assert p.locator('#tl').count()==1
 assert p.locator('#anaB table tr').count()>2
 assert 'model\'s account, not a measurement' in p.locator('#anaB').inner_text()
 # 3 breakdown model switch + abstain updates
 p.locator('#anaM').select_option('lr')
 assert 'logistic' in p.locator('#anaB').inner_text().lower()
 p.locator('#thr').evaluate("e=>{e.value=0.9;e.dispatchEvent(new Event('input'))}")
 assert 'Abstained' in p.locator('#anaB').inner_text()
 # 4 CSV import path
 p.locator('#f').set_input_files(str(P/'sample_import.csv'))
 p.wait_for_selector('#use',timeout=15000)
 p.locator('#use').click()
 p.wait_for_function("document.querySelector('#prog') && document.querySelector('#prog').textContent.startsWith('Done')",timeout=120000)
 assert p.locator('#anaB table tr').count()>2
 # 5 bad CSV shows error
 p.locator('#f').set_input_files(str(P/'tests/bad_tmp.csv'))
 assert 'Missing columns' in p.locator('#rep').inner_text()
 # 6 bench tab still works
 p.locator('nav button[data-tab=bench]').click()
 p.wait_for_function("document.querySelector('#prog') && document.querySelector('#prog').textContent.startsWith('Done')",timeout=120000)
 assert len(p.locator('#sum table tr').all())>=4
 # 7 save session persists
 p.locator('#save').click();p.reload()
 p.locator('nav button[data-tab=hist]').click()
 assert len(p.locator('#view table tr').all())>=2
 # 8 evidence + stress render
 p.locator('nav button[data-tab=evidence]').click();p.wait_for_function("document.querySelector('#lat') && document.querySelector('#lat').textContent.includes('ms')",timeout=30000)
 p.locator('nav button[data-tab=stress]').click()
 p.locator('#pk').select_option('thigh_rot');p.locator('#go').click()
 p.wait_for_selector('#lr table',timeout=120000)
 # 9 mobile
 m=b.new_context(viewport={'width':390,'height':844},is_mobile=True).new_page()
 m.goto((P/'index.html').as_uri())
 m.locator('#tryme').click()
 m.wait_for_function("document.querySelector('#prog') && document.querySelector('#prog').textContent.startsWith('Done')",timeout=120000)
 assert m.evaluate('document.documentElement.scrollWidth<=391')
 for t in ['bench','evidence','stress','hist']:
  m.locator('nav button[data-tab='+t+']').click();assert m.evaluate('document.documentElement.scrollWidth<=391')
 assert not errors,errors
 b.close()
print('MotionProof UI checks passed')
