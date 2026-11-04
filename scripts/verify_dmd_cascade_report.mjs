import path from 'node:path';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const packages=path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const {chromium}=require(path.join(packages,'playwright'));
const root=path.resolve('.');
const output=path.join(root,'eval_results/dmd_cascade_reuse_20260915');
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--disable-gpu']});
try {
  const page=await browser.newPage({viewport:{width:1300,height:1000}});
  await page.goto(pathToFileURL(path.join(root,'docs/research_docs/RDD_DMD_CASCADE_FINETUNE_20260915_ZH.html')).href);
  await page.waitForLoadState('load');
  const audit=await page.evaluate(()=>({
    title:document.title,
    tables:document.querySelectorAll('table').length,
    katex_errors:document.querySelectorAll('.katex-error').length,
    images:document.images.length,
    broken_images:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),
    desktop_overflow:document.documentElement.scrollWidth>innerWidth+1,
    speed_gate_values_present:document.body.textContent.includes('22.17%')&&document.body.textContent.includes('21.66%'),
  }));
  await page.screenshot({path:path.join(output,'report_preview.png')});
  await page.setViewportSize({width:390,height:850});
  audit.mobile_overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  await fs.writeFile(path.join(output,'browser_audit.json'),JSON.stringify(audit,null,2));
  console.log(JSON.stringify(audit,null,2));
  if(audit.katex_errors||audit.broken_images.length||audit.desktop_overflow||audit.mobile_overflow||!audit.speed_gate_values_present)process.exitCode=1;
} finally {await browser.close();}
