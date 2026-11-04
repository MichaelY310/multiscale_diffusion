import path from 'node:path';
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(import.meta.url);
const packages=path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const {chromium}=require(path.join(packages,'playwright'));
const root=path.resolve('.');
const docs=path.join(root,'docs/research_docs');
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--disable-gpu']});
try {
 const page=await browser.newPage({viewport:{width:1080,height:318},deviceScaleFactor:1});
 for(const name of ['RDD_REUSE_TIERS_FINE24','RDD_REUSE_TIERS_CASCADE24','RDD_REUSE_TIERS_HEAD1','RDD_REUSE_TIERS_FINE28']) {
  await page.goto(pathToFileURL(path.join(docs,name+'.svg')).href);
  await page.screenshot({path:path.join(docs,name+'.png')});
 }
 await page.setViewportSize({width:1300,height:950});
 await page.goto(pathToFileURL(path.join(docs,'RDD_REUSE_SPEED_TIERS_20260915_ZH.html')).href);
 await page.waitForLoadState('load');
 await page.screenshot({path:path.join(root,'eval_results/reuse_latency_tiers_20260915/report_preview.png')});
 const audit=await page.evaluate(()=>({
  katex_errors:document.querySelectorAll('.katex-error').length,
  formulas:document.querySelectorAll('.katex').length,
  videos:document.querySelectorAll('video').length,
  broken_images:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.getAttribute('src')),
  horizontal_overflow:document.documentElement.scrollWidth>innerWidth+1,
 }));
 await page.setViewportSize({width:390,height:850});
 audit.narrow_horizontal_overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
 await page.setViewportSize({width:1300,height:950});
 const video=page.locator('video').filter({hasNot:page.locator('[data-nonexistent]')}).first();
 audit.first_video=await video.evaluate(async el=>{
  el.preload='metadata';
  const done=new Promise(resolve=>{el.onloadedmetadata=()=>resolve(true);el.onerror=()=>resolve(false)});
  el.load();
  const ready=await Promise.race([done,new Promise(resolve=>setTimeout(()=>resolve(false),5000))]);
  return {ready,width:el.videoWidth,height:el.videoHeight,duration:el.duration};
 });
 audit.all_video_metadata=[];
 for(const handle of await page.locator('video').all()) {
  audit.all_video_metadata.push(await handle.evaluate(async el=>{
   if(!Number.isFinite(el.duration)) {
    el.preload='metadata';
    const ready=new Promise(resolve=>{el.onloadedmetadata=()=>resolve(true);el.onerror=()=>resolve(false)});
    el.load();await Promise.race([ready,new Promise(resolve=>setTimeout(resolve,5000))]);
   }
   return {src:el.getAttribute('src'),width:el.videoWidth,height:el.videoHeight,duration:el.duration};
  }));
 }
 audit.invalid_videos=audit.all_video_metadata.filter(v=>!v.width||!v.height||!Number.isFinite(v.duration));
 await fs.writeFile(path.join(root,'eval_results/reuse_latency_tiers_20260915/browser_audit.json'),JSON.stringify(audit,null,2));
 const {all_video_metadata,...brief}=audit;
 console.log(JSON.stringify({...brief,checked_video_metadata:all_video_metadata.length},null,2));
 if(audit.katex_errors||audit.broken_images.length||audit.horizontal_overflow||audit.narrow_horizontal_overflow||audit.invalid_videos.length||!audit.first_video.ready)process.exitCode=1;
} finally {await browser.close();}
