import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
export const here=path.dirname(fileURLToPath(import.meta.url));
export const work=path.resolve(here,'..');
export const root=path.resolve(work,'..');
export const out=path.join(root,'06_模型与地图','交互地图');
export const html=path.join(out,'磷灰石分类结果_交互地图.html');
export const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
export const sha256=data=>createHash('sha256').update(data).digest('hex');
export const readJson=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
export async function connect(width,height,figure=false){
  const record=await readJson(path.join(here,'renderer.json'));
  if(record.purpose!=='apatite-20261022-map-renderer'||path.resolve(record.profile_path).toLowerCase()!==path.join(here,'browser-profile').toLowerCase())throw Error('Dedicated renderer ownership record mismatch');
  if(path.resolve(record.project_root).toLowerCase()!==root.toLowerCase())throw Error('Renderer belongs to a different project');
  const tabs=await(await fetch(`http://127.0.0.1:${record.port}/json`)).json();
  const expected=pathToFileURL(html).href;
  const tab=tabs.find(t=>t.type==='page'&&t.url.startsWith(expected))||tabs.find(t=>t.type==='page'&&t.url==='about:blank');
  if(!tab)throw Error('No owned blank/current-map tab in dedicated renderer');
  const ws=new WebSocket(tab.webSocketDebuggerUrl);
  await Promise.race([new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});}),pause(10000).then(()=>{throw Error('Renderer WebSocket timeout');})]);
  let seq=0;const pending=new Map();const runtimeErrors=[];
  ws.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id&&pending.has(message.id)){const item=pending.get(message.id);pending.delete(message.id);clearTimeout(item.timer);message.error?item.reject(Error(JSON.stringify(message.error))):item.resolve(message.result);}if(message.method==='Runtime.exceptionThrown')runtimeErrors.push(message.params.exceptionDetails);});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{pending.delete(id);reject(Error(`CDP timeout: ${method}`));},20000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});
  const ev=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
  const close=()=>{for(const item of pending.values())clearTimeout(item.timer);pending.clear();ws.close();};
  try{
    await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    await call('Page.navigate',{url:expected+(figure?'?figure=1':'')});
    let ready=false;for(let i=0;i<60;i++){await pause(300);if(await ev('Boolean(window.mapReady)')){ready=true;break;}}
    if(!ready)throw Error('Map did not initialize');
    const meta=await readJson(path.join(out,'地图数据审计.json'));
    const live=await ev('({model_version:META.model_version,training_sha256:META.training_sha256})');
    if(live.model_version!==meta.model_version||live.training_sha256!==meta.training_sha256)throw Error('HTML predictions and current application audit differ');
    return {call,ev,close,runtimeErrors,meta,html_sha256:sha256(await fs.readFile(html)),port:record.port};
  }catch(error){close();throw error;}
}
