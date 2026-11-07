importScripts('recog.js','decode.js');
const ready=fetch('model.json').then(r=>r.json()).then(j=>Recog.loadNet(j));
onmessage=async e=>{
  const {id,mode,imgData}=e.data;
  try{
    await ready;
    const r=Recog.readImage(imgData);
    const tiles=r.tiles.map(t=>({tile:t.tile,red:t.red,conf:t.conf,side:t.side}));
    if(mode==='hand'){const sol=Decode.solve(r.tiles);postMessage({id,ok:true,tiles,sol,uncertain:sol.uncertain});}
    else postMessage({id,ok:true,tiles});
  }catch(err){postMessage({id,ok:false,msg:String(err&&err.message||err)});}
};
