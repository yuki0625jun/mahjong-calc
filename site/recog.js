// ===== 牌認識（ブラウザ内で完結） =====
// 1) 写真から牌の白い面を探して1枚ずつ切り出す  2) 小さなニューラルネットで37種（34種＋赤5×3）を判定
const Recog=(()=>{
let BONUS=1.0;
const OW=48,OH=64;
function resample(src,sw,sh,x0,y0,cw,ch){ // src: Float32Array RGB (sw*sh*3), 切り出し範囲を OWxOH にバイリニア
  const out=new Float32Array(OW*OH*3);
  for(let i=0;i<OH;i++){let ys=(i+0.5)*ch/OH-0.5;ys=Math.min(Math.max(ys,0),ch-1);const yy0=Math.floor(ys),yy1=Math.min(yy0+1,ch-1),fy=ys-yy0;
    for(let j=0;j<OW;j++){let xs=(j+0.5)*cw/OW-0.5;xs=Math.min(Math.max(xs,0),cw-1);const xx0=Math.floor(xs),xx1=Math.min(xx0+1,cw-1),fx=xs-xx0;
      for(let k=0;k<3;k++){
        const p=(y,x)=>src[((y0+y)*sw+(x0+x))*3+k];
        const a=p(yy0,xx0),b=p(yy0,xx1),c=p(yy1,xx0),d=p(yy1,xx1);
        out[(i*OW+j)*3+k]=(a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy;}}}
  return out;
}
function features(x){ // x: OWxOHx3 Float32 0..1
  const N=OW*OH,L=new Float32Array(N);
  for(let i=0;i<N;i++)L[i]=0.299*x[i*3]+0.587*x[i*3+1]+0.114*x[i*3+2];
  const s=Float32Array.from(L).sort();const white=Math.max(s[Math.floor(0.9*(N-1))],0.05);
  for(let i=0;i<N*3;i++)x[i]=Math.min(Math.max(x[i]/white,0),1.2);
  for(let i=0;i<N;i++)L[i]=Math.min(Math.max(L[i]/white,0),1.2);
  const f=new Float32Array(1280);let o=0;
  const blk=(arr)=>{const r=new Float32Array(192);for(let by=0;by<16;by++)for(let bx=0;bx<12;bx++){let t=0;for(let y=0;y<4;y++)for(let q=0;q<4;q++)t+=arr[(by*4+y)*OW+bx*4+q];r[by*12+bx]=t/16;}return r;};
  const g=blk(L);let m=0;for(const v of g)m+=v;m/=192;let sd=0;for(const v of g)sd+=(v-m)*(v-m);sd=Math.sqrt(sd/192);
  for(let i=0;i<192;i++)f[o++]=(g[i]-m)/(sd+0.05);
  const hog=new Float32Array(512);
  for(let y=0;y<OH;y++)for(let xx=0;xx<OW;xx++){
    const gx=(xx>0&&xx<OW-1)?L[y*OW+xx+1]-L[y*OW+xx-1]:0, gy=(y>0&&y<OH-1)?L[(y+1)*OW+xx]-L[(y-1)*OW+xx]:0;
    const mag=Math.sqrt(gx*gx+gy*gy);let ang=Math.atan2(gy,gx);ang=((ang%Math.PI)+Math.PI)%Math.PI;
    const b=Math.min(Math.floor(ang/Math.PI*8),7);hog[(Math.floor(y/8)*8+Math.floor(xx/6))*8+b]+=mag;}
  let nrm=0;for(const v of hog)nrm+=v*v;nrm=Math.sqrt(nrm)+1e-3;
  for(let i=0;i<512;i++)f[o++]=Math.sqrt(hog[i]/nrm)*4;
  const R=new Float32Array(N),G=new Float32Array(N),B=new Float32Array(N);
  for(let i=0;i<N;i++){const r=x[i*3],gg=x[i*3+1],b=x[i*3+2];R[i]=Math.max(r-Math.max(gg,b),0);G[i]=Math.max(gg-Math.max(r,b),0);B[i]=Math.max(b-Math.max(r,gg),0);}
  for(const arr of [R,G,B]){const v=blk(arr);for(let i=0;i<192;i++)f[o++]=v[i]*4;}
  return f;
}
let NET=null;
function loadNet(j){ // {mean,scale,layers:[{w(base64 int8),s(scale per out),b,in,out}]}
  const dec=b64=>{const s=atob(b64),a=new Int8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i)<<24>>24;return a;};
  NET={mean:Float32Array.from(j.mean),scale:Float32Array.from(j.scale),layers:j.layers.map(l=>({w:dec(l.w),s:Float32Array.from(l.s),b:Float32Array.from(l.b),n:l.in,m:l.out}))};
}
function predict(f){ // -> probabilities (37)
  let h=new Float32Array(f.length);for(let i=0;i<f.length;i++)h[i]=(f[i]-NET.mean[i])/NET.scale[i];
  NET.layers.forEach((l,li)=>{const o=new Float32Array(l.m);
    for(let j=0;j<l.m;j++){let t=0;const base=j*l.n;for(let i=0;i<l.n;i++)t+=h[i]*l.w[base+i];o[j]=t*l.s[j]+l.b[j];}
    if(li<NET.layers.length-1)for(let j=0;j<l.m;j++)o[j]=Math.max(o[j],0);h=o;});
  let mx=-1e9;for(const v of h)mx=Math.max(mx,v);let s=0;const p=h.map(v=>{const e=Math.exp(v-mx);s+=e;return e});return p.map(v=>v/s);
}
// ---- 牌の検出 ----
function toFloat(imgData){const d=imgData.data,n=imgData.width*imgData.height,a=new Float32Array(n*3);for(let i=0;i<n;i++){a[i*3]=d[i*4]/255;a[i*3+1]=d[i*4+1]/255;a[i*3+2]=d[i*4+2]/255;}return a;}
function otsu(vals){const hist=new Array(256).fill(0);vals.forEach(v=>hist[Math.min(255,Math.max(0,Math.round(v*255)))]++);const tot=vals.length;let sum=0;for(let i=0;i<256;i++)sum+=i*hist[i];
  let sB=0,wB=0,best=0,th=128;for(let i=0;i<256;i++){wB+=hist[i];if(!wB)continue;const wF=tot-wB;if(!wF)break;sB+=i*hist[i];const mB=sB/wB,mF=(sum-sB)/wF,v=wB*wF*(mB-mF)*(mB-mF);if(v>best){best=v;th=i;}}return th/255;}
function morph(mask,w,h,r,dil){ // 正方形カーネルの膨張/収縮（分離可能）
  const tmp=new Uint8Array(w*h),out=new Uint8Array(w*h);
  for(let y=0;y<h;y++){let run=0;for(let x=0;x<w;x++){} // placeholder
    for(let x=0;x<w;x++){let v=dil?0:1;for(let k=-r;k<=r;k++){const xx=x+k;const m=(xx<0||xx>=w)?(dil?0:1):mask[y*w+xx];if(dil&&m){v=1;break;}if(!dil&&!m){v=0;break;}}tmp[y*w+x]=v;}}
  for(let x=0;x<w;x++)for(let y=0;y<h;y++){let v=dil?0:1;for(let k=-r;k<=r;k++){const yy=y+k;const m=(yy<0||yy>=h)?(dil?0:1):tmp[yy*w+x];if(dil&&m){v=1;break;}if(!dil&&!m){v=0;break;}}out[y*w+x]=v;}
  return out;
}
function components(mask,w,h){const lab=new Int32Array(w*h).fill(-1),comps=[];const st=[];
  for(let i=0;i<w*h;i++){if(!mask[i]||lab[i]>=0)continue;const id=comps.length;let c={id,n:0,x0:w,y0:h,x1:0,y1:0,sx:0,sy:0,sxx:0,syy:0,sxy:0};lab[i]=id;st.push(i);
    while(st.length){const p=st.pop(),x=p%w,y=(p/w)|0;c.n++;c.sx+=x;c.sy+=y;c.sxx+=x*x;c.syy+=y*y;c.sxy+=x*y;if(x<c.x0)c.x0=x;if(x>c.x1)c.x1=x;if(y<c.y0)c.y0=y;if(y>c.y1)c.y1=y;
      for(const q of [p-1,p+1,p-w,p+w]){if(q<0||q>=w*h)continue;if((q===p-1&&x===0)||(q===p+1&&x===w-1))continue;if(mask[q]&&lab[q]<0){lab[q]=id;st.push(q);}}}
    comps.push(c);}
  return {lab,comps};}
function faceMask(a,w,h){
  const sc=new Float32Array(w*h);
  for(let i=0;i<w*h;i++){const r=a[i*3],g=a[i*3+1],b=a[i*3+2],mx=Math.max(r,g,b),mn=Math.min(r,g,b),S=mx>0?(mx-mn)/mx:0;sc[i]=mx*(1-S)*(1-S);}
  const th=otsu(Array.from(sc));const m=new Uint8Array(w*h);for(let i=0;i<w*h;i++)m[i]=sc[i]>th?1:0;return m;}
function rotateImage(a,w,h,ang){ // ang(rad)だけ回転（中心まわり）、同サイズ
  const o=new Float32Array(w*h*3),c=Math.cos(ang),s=Math.sin(ang),cx=w/2,cy=h/2;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const dx=x-cx,dy=y-cy;const sx=Math.round(c*dx-s*dy+cx),sy=Math.round(s*dx+c*dy+cy);
    if(sx>=0&&sx<w&&sy>=0&&sy<h){const p=(sy*w+sx)*3,q=(y*w+x)*3;o[q]=a[p];o[q+1]=a[p+1];o[q+2]=a[p+2];}}
  return o;}
function maskOf(a,w,h){let m=faceMask(a,w,h);const rc=Math.max(2,Math.round(Math.min(w,h)*0.012)),ro=Math.max(1,Math.round(Math.min(w,h)*0.003));
  m=morph(morph(m,w,h,rc,true),w,h,rc,false); // 文字の穴を埋める
  m=morph(morph(m,w,h,ro,false),w,h,ro,true); // 小さなゴミを消す
  return m;}
function cropVariants(a,w,h,x0,y0,x1,y1,side){
  x0=Math.max(0,Math.round(x0));y0=Math.max(0,Math.round(y0));x1=Math.min(w,Math.round(x1));y1=Math.min(h,Math.round(y1));
  const cw=x1-x0,ch=y1-y0;if(cw<3||ch<3)return [];
  if(!side)return [resample(a,w,h,x0,y0,cw,ch)];
  const out=[];for(const dir of [1,-1]){const tmp=new Float32Array(cw*ch*3);
    for(let y=0;y<ch;y++)for(let x=0;x<cw;x++){const p=((y0+y)*w+(x0+x))*3;const nx=dir>0?(ch-1-y):y,ny=dir>0?x:(cw-1-x);const q=(ny*ch+nx)*3;tmp[q]=a[p];tmp[q+1]=a[p+1];tmp[q+2]=a[p+2];}
    out.push(resample(tmp,ch,cw,0,0,ch,cw));}
  return out;}
function scoreBox(a,w,h,x0,y0,x1,y1,side){
  let best=null;for(const v of cropVariants(a,w,h,x0,y0,x1,y1,side)){const p=predict(features(v));const pj=p[37];let mi=0;for(let i=1;i<37;i++)if(p[i]>p[mi])mi=i;
    const sc=Math.log(p[mi]+1e-6)+Math.log(1-pj+1e-6);if(!best||sc>best.sc)best={sc,p};}
  return best;}
function detect(a,w,h){
  let m=maskOf(a,w,h);
  let comps=components(m,w,h).comps;const minA=w*h*0.0008;comps=comps.filter(c=>c.n>minA);
  if(!comps.length)return {tiles:[],img:a};
  const angs=[];comps.forEach(c=>{const mx=c.sx/c.n,my=c.sy/c.n,vxx=c.sxx/c.n-mx*mx,vyy=c.syy/c.n-my*my,vxy=c.sxy/c.n-mx*my;const t=0.5*Math.atan2(2*vxy,vxx-vyy);
    const elong=Math.sqrt((vxx-vyy)**2+4*vxy*vxy)/(vxx+vyy+1e-6);if(elong>0.5&&Math.abs(t)<Math.PI/4)angs.push([t,c.n]);});
  let ang=0;if(angs.length){angs.sort((p,q)=>p[0]-q[0]);const tot=angs.reduce((s,p)=>s+p[1],0);let acc=0;for(const p of angs){acc+=p[1];if(acc>=tot/2){ang=p[0];break;}}}
  if(Math.abs(ang)>0.01){a=rotateImage(a,w,h,ang);m=maskOf(a,w,h);}
  const all=components(m,w,h);let cs=all.comps.filter(c=>c.n>minA&&c.n/((c.x1-c.x0+1)*(c.y1-c.y0+1))>0.5);
  if(!cs.length)return {tiles:[],img:a};
  const Hmax=Math.max(...cs.map(c=>c.y1-c.y0+1));
  cs=cs.filter(c=>(c.y1-c.y0+1)>Hmax*0.4);
  const tiles=[];
  for(const c of cs){
    const W=c.x1-c.x0+1;const top=new Float32Array(W),bot=new Float32Array(W);
    for(let x=0;x<W;x++){let t=-1,b=-1;for(let y=c.y0;y<=c.y1;y++)if(all.lab[y*w+c.x0+x]===c.id){if(t<0)t=y;b=y;}top[x]=t<0?c.y1:t;bot[x]=b<0?c.y0:b;}
    const hts=Array.from(top,(t,i)=>bot[i]-t).filter(v=>v>0).sort((p,q)=>p-q);const hRef=hts[Math.floor(hts.length*0.8)]||(c.y1-c.y0);
    const s=Math.max(2,Math.round(hRef*0.08));const K=Math.ceil(W/s);
    const pos=k=>Math.min(c.x0+k*s,c.x1+1);
    const winH=(k,k2)=>{let t=1e9,b=-1;for(let x=k*s;x<Math.min(k2*s,W);x++){if(top[x]<t)t=top[x];if(bot[x]>b)b=bot[x];}return [t,b];};
    const medH=(k,k2)=>{const v=[];for(let x=k*s;x<Math.min(k2*s,W);x++)v.push(bot[x]-top[x]);v.sort((p,q)=>p-q);return v[v.length>>1]||0;};
    const best=new Float64Array(K+1).fill(-1e18),from=new Int32Array(K+1).fill(-1),info=new Array(K+1);
    for(let k=0;k<=K&&k*s<=hRef*0.35;k++){best[k]=-0.6*k;from[k]=-2;}
    const memo=new Map();
    for(let k=0;k<K;k++){if(best[k]<-1e17)continue;
      // スキップ
      if(best[k]-0.6>best[k+1]){best[k+1]=best[k]-0.6;from[k+1]=k;info[k+1]=null;}
      for(let k2=k+1;k2<=K;k2++){const wpx=(k2-k)*s;if(wpx>hRef*2.3)break;const [t,b]=winH(k,k2);const hh=b-t+1;if(hh<hRef*0.3)continue;const asp=wpx/hh;const mh=medH(k,k2);
        let side=null;if(asp>=0.55&&asp<=1.3&&mh>hRef*0.84)side=false;else if(asp>=1.2&&asp<=2.3&&mh<hRef*0.9)side=true;else continue;
        const r=scoreBox(a,w,h,pos(k),t,pos(k2),b+1,side);if(!r)continue;const sc=best[k]+r.sc+BONUS;
        if(sc>best[k2]){best[k2]=sc;from[k2]=k;info[k2]={x0:pos(k),x1:pos(k2),y0:t,y1:b,side,p:r.p};}}}
    let end=K;for(let k=K;k>=0&&(K-k)*s<=hRef*0.35;k--){if(best[k]-0.6*(K-k)>best[end]-0.6*(K-end))end=k;}
    const seq=[];let k=end;while(k>0&&from[k]>=0){if(info[k])seq.push(info[k]);k=from[k];}
    seq.reverse().forEach(t=>tiles.push({...t,cy:(t.y0+t.y1)/2}));
  }
  tiles.sort((p,q)=>p.cy-q.cy);
  const rows=[];for(const t of tiles){const row=rows.find(rw=>Math.abs(rw.cy-t.cy)<Hmax*0.6);if(row){row.t.push(t);}else rows.push({cy:t.cy,t:[t]});}
  const out=[];rows.sort((p,q)=>p.cy-q.cy).forEach((rw,ri)=>{rw.t.sort((p,q)=>p.x0-q.x0);rw.t.forEach(t=>t.row=ri);out.push(...rw.t);});
  return {tiles:out,img:a,H:Hmax};
}
const CLS=[];for(let i=0;i<34;i++)CLS.push(i);CLS.push(4,13,22); // 34-36は赤5
function readImage(imgData){
  const w=imgData.width,h=imgData.height;let a=toFloat(imgData);
  const det=detect(a,w,h);a=det.img;
  const tiles=det.tiles.map(t=>{ // 少しずらした切り出しでも判定して平均（安定化）
    const bw=t.x1-t.x0,bh=t.y1-t.y0,p=Float64Array.from(t.p);let n=1;
    for(const [dx0,dy0,dx1,dy1] of [[0.05,0.04,0,0],[0,0,-0.05,-0.04],[-0.04,-0.03,0.04,0.03],[0.04,0.03,-0.04,-0.03]]){
      const r=scoreBox(a,w,h,t.x0+dx0*bw,t.y0+dy0*bh,t.x1+dx1*bw,t.y1+1+dy1*bh,t.side);if(r){r.p.forEach((v,i)=>p[i]+=v);n++;}}
    for(let i=0;i<p.length;i++)p[i]/=n;
    let pu=null;if(t.side){const r=scoreBox(a,w,h,t.x0,t.y0,t.x1,t.y1+1,false);if(r)pu=Array.from(r.p);}
    const merged=new Float32Array(34);for(let i=0;i<37;i++)merged[CLS[i]]+=p[i];let mi=0;for(let i=1;i<34;i++)if(merged[i]>merged[mi])mi=i;
    const red=(mi===4&&p[34]>p[4])||(mi===13&&p[35]>p[13])||(mi===22&&p[36]>p[22]);
    return {x0:t.x0,x1:t.x1,y0:t.y0,y1:t.y1,side:t.side,row:t.row,tile:mi,red,conf:merged[mi],p:Array.from(p),pu};});
  return {tiles,H:det.H};
}
return {features,resample,loadNet,predict,readImage,detect,toFloat,maskOf,setBonus:v=>BONUS=v};
})();
if(typeof module!=='undefined')module.exports=Recog;Recog._detect=null;
