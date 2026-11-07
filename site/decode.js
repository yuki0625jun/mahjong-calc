// ===== 読み取り結果の整理：並び→手の内/和了牌/副露、和了形になるよう候補から補正 =====
const Decode=(()=>{
function canSets(c,need){let i=0;while(i<34&&!c[i])i++;if(i===34)return need===0;if(need===0)return false;
  if(c[i]>=3){c[i]-=3;const ok=canSets(c,need-1);c[i]+=3;if(ok)return true;}
  if(i<27&&i%9<=6&&c[i+1]&&c[i+2]){c[i]--;c[i+1]--;c[i+2]--;const ok=canSets(c,need-1);c[i]++;c[i+1]++;c[i+2]++;if(ok)return true;}
  return false;}
function isAgari(conc,nMelds){const c=new Array(34).fill(0);conc.forEach(t=>c[t]++);
  if(c.some(v=>v>4))return false;
  if(nMelds===0){if(c.filter(v=>v===2).length===7)return true;
    const y=[0,8,9,17,18,26,27,28,29,30,31,32,33];if(conc.length===14&&y.every(t=>c[t]>=1)&&y.reduce((s,t)=>s+c[t],0)===14)return true;}
  const need=4-nMelds;if(conc.length!==need*3+2)return false;
  for(let p=0;p<34;p++)if(c[p]>=2){c[p]-=2;const ok=canSets(c,need);c[p]+=2;if(ok)return true;}return false;}
function meldOf(ts){ // ts: 牌の種類の配列 → {type,t} or null
  const s=ts.slice().sort((a,b)=>a-b);
  if(s.length===4&&s.every(x=>x===s[0]))return {type:'minkan',t:s[0]};
  if(s.length===2&&s[0]===s[1])return {type:'ankan',t:s[0]};
  if(s.length===3&&s.every(x=>x===s[0]))return {type:'pon',t:s[0]};
  if(s.length===3&&s[0]<27&&s[1]===s[0]+1&&s[2]===s[0]+2&&Math.floor(s[0]/9)===Math.floor(s[2]/9))return {type:'chi',t:s[0]};
  return null;}
// 並びからグループ分け
function group(tiles){
  if(!tiles.length)return [];
  const ws=tiles.filter(t=>!t.side).map(t=>t.x1-t.x0).sort((a,b)=>a-b);const Wt=ws[ws.length>>1]||40;
  const groups=[];let cur=[tiles[0]];
  for(let i=1;i<tiles.length;i++){const a=tiles[i-1],b=tiles[i];const gap=(b.row!==a.row)?1e9:b.x0-a.x1;if(gap>Wt*0.2){groups.push(cur);cur=[];}cur.push(b);}
  groups.push(cur);return groups;}
// 手の内・副露を決めて、和了形になる中で一番確からしい読みを選ぶ
function cands(p){const m=new Float64Array(34);for(let i=0;i<37;i++)m[i<34?i:[4,13,22][i-34]]+=p[i];
  return Array.from(m,(v,i)=>[i,Math.log(v+1e-6)]).sort((a,b)=>b[1]-a[1]).slice(0,4);}
function hypotheses(tiles){
  const n=tiles.length,H=[];const sides=tiles.map((t,i)=>t.side?i:-1).filter(i=>i>=0);
  H.push({conc:n,sizes:[]});
  // 右端から、横向き牌を1枚ずつ含む3〜4枚の組に分ける
  const rec=(end,left,sizes)=>{ if(left===0){const conc=end;if(sizes.length&&conc>=2)H.push({conc,sizes:sizes.slice().reverse()});return;}
    for(const sz of [3,4]){const st=end-sz;if(st<0)continue;const inside=sides.filter(i=>i>=st&&i<end).length;if(inside!==1)continue;
      if(sides.some(i=>i<st)&&left===1)continue;rec(st,left-1,[...sizes,sz]);}};
  for(let m=1;m<=Math.min(4,sides.length);m++)rec(n,m,[]);
  // 2枚だけ離れた組（暗槓：両端が伏せ牌）
  const gs=group(tiles);let idx=0;const pos=[];gs.forEach(g=>{pos.push([idx,g.length]);idx+=g.length;});
  const twos=pos.filter(([i,l],gi)=>l===2&&gi>0&&i>=5);
  if(twos.length){const first=twos[0][0];const tail=[];let ok=true;for(let i=first;i<n;){const g=pos.find(p=>p[0]===i);if(!g){ok=false;break;}if(g[1]===2||g[1]===3||g[1]===4){tail.push(g[1]);i+=g[1];}else if(g[1]%3===0){for(let k=0;k<g[1]/3;k++)tail.push(3);i+=g[1];}else{ok=false;break;}}
    if(ok)H.push({conc:first,sizes:tail});}
  return H;}
function solve(tiles){
  const H=hypotheses(tiles);let best=null;
  for(const h of H){
    const C=tiles.map((t,i)=>cands(i<h.conc&&t.pu?t.pu:t.p));
    const nC=h.conc,sizes=h.sizes,n=tiles.length;
    const nm=sizes.length;if(nC+nm*3!==14&&!(nm===0&&nC===14))continue;
    const build=as=>{const cc=as.slice(0,nC);let o=nC;const ms=[];for(const sz of sizes){ms.push(meldOf(as.slice(o,o+sz)));o+=sz;}return {cc,ms};};
    const valid=as=>{const {cc,ms}=build(as);if(ms.some(m=>!m))return false;const cnt=new Array(34).fill(0);cc.forEach(t=>cnt[t]++);
      ms.forEach(m=>{(m.type==='chi'?[m.t,m.t+1,m.t+2]:Array(m.type.endsWith('kan')?4:3).fill(m.t)).forEach(t=>cnt[t]++);});
      return !cnt.some(v=>v>4)&&isAgari(cc,ms.length);};
    const cost=as=>as.reduce((s,t,i)=>s-(C[i].find(c=>c[0]===t)||[0,-20])[1],0);
    const base=C.map(c=>c[0][0]);let hb=null,hc=1e9;const tryA=a=>{if(valid(a)){const c=cost(a);if(c<hc){hc=c;hb=a.slice();}}};
    tryA(base);
    if(!hb){for(let i=0;i<n;i++)for(let a=1;a<C[i].length;a++){const x=base.slice();x[i]=C[i][a][0];tryA(x);}}
    if(!hb){for(let i=0;i<n;i++)for(let a=1;a<C[i].length;a++)for(let j=i+1;j<n;j++)for(let b=1;b<C[j].length;b++){const x=base.slice();x[i]=C[i][a][0];x[j]=C[j][b][0];tryA(x);}}
    if(!hb){for(let i=0;i<n;i++)for(let a=1;a<3;a++)for(let j=i+1;j<n;j++)for(let b=1;b<3;b++)for(let k=j+1;k<n;k++)for(let c=1;c<3;c++){const x=base.slice();x[i]=C[i][a][0];x[j]=C[j][b][0];x[k]=C[k][c][0];tryA(x);}}
    const res=hb?{as:hb,cost:hc,valid:true}:{as:base,cost:cost(base)+50,valid:false};
    // 副露の仮説には小さなペナルティ（横向きの誤検出対策）
    res.cost+=nm*0.3;
    if(!best||res.cost<best.cost)best={...res,h,C,build,base};
  }
  if(!best){const C=tiles.map(t=>cands(t.p));return {concealed:C.map(c=>c[0][0]),melds:[],red:0,valid:false,fixed:false,uncertain:C.map(()=>true)};}
  const {cc,ms}=best.build(best.as);
  const reds=tiles.reduce((s,t,i)=>{const k=[4,13,22].indexOf(best.as[i]);const p=(i<best.h.conc&&t.pu)?t.pu:t.p;return s+(k>=0&&p[34+k]>p[best.as[i]]?1:0);},0);
  return {concealed:cc,melds:ms.map((m,i)=>m||{type:'pon',t:best.as[best.h.conc+best.h.sizes.slice(0,i).reduce((s,v)=>s+v,0)]}),red:reds,valid:best.valid,
    fixed:best.as.some((t,i)=>t!==best.base[i]),uncertain:best.as.map((t,i)=>best.C[i][0][1]<Math.log(0.6)||t!==best.base[i])};
}
return {solve,isAgari,group};
})();
if(typeof module!=='undefined')module.exports=Decode;
