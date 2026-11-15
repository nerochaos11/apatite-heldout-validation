import fs from 'node:fs/promises';
import path from 'node:path';
import {connect,out,pause,sha256} from './cdp.mjs';
const client=await connect(1000,620,true);
try{
  const {call,ev,meta,runtimeErrors}=client;
  const audit={model_version:meta.model_version,training_sha256:meta.training_sha256,html_sha256:client.html_sha256,viewport:{width:1000,height:620,deviceScaleFactor:1},views:[],errors:runtimeErrors};
  for(const name of ['global','regional','county','local']){
    const view=await ev(`window.setFigureView(${JSON.stringify(name)})`);let state;let stable=0;
    for(let i=0;i<55;i++){
      await pause(600);
      state=await ev('({loaded:document.querySelectorAll(".leaflet-tile-loaded").length,active:document.querySelectorAll(".leaflet-tile:not(.leaflet-tile-loaded)").length,tiles:window.tileStats,visible:window.visibleSummary,zoom:map.getZoom(),hide_matches:!document.getElementById("includeMatches").checked,attribution:document.querySelector(".leaflet-control-attribution")?.textContent,scale:document.querySelector(".leaflet-control-scale-line")?.textContent})');
      stable=state.loaded>0&&state.active===0?stable+1:0;if(stable>=2)break;
    }
    if(stable<2||state.tiles.errors!==0)throw Error(`Incomplete or failed imagery tiles for ${name}: ${JSON.stringify(state)}`);
    if(!state.hide_matches||!state.attribution?.includes('Esri')||!state.scale)throw Error(`Missing hide-match filter, basemap credit or scale at ${name}`);
    await pause(350);const shot=await call('Page.captureScreenshot',{format:'png',fromSurface:true});const data=Buffer.from(shot.data,'base64');
    await fs.writeFile(path.join(out,name+'.png'),data);
    audit.views.push({name,bounds:view.bounds,png_sha256:sha256(data),...state});
    console.log('Captured',name,JSON.stringify(state.visible));
  }
  audit.defaultSummary=await ev('window.visibleSummary');
  if(audit.defaultSummary.records!==meta.display_nonmatch_records||audit.defaultSummary.coordinates!==meta.nonmatch_coordinates)throw Error('Default counts differ from current-model metadata');
  if(runtimeErrors.length)throw Error('Browser runtime errors: '+JSON.stringify(runtimeErrors));
  await fs.writeFile(path.join(out,'地图视野参数.json'),JSON.stringify(audit,null,2));
  console.log('All four current-model views captured; 地图视野参数.json saved');
}finally{client.close();}
