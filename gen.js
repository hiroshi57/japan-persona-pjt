// 全国分布から都度ペルソナを生成する（画面側）。各属性は全国統計に一致、主要な相関を反映。
// 厳密な結合分布は本体エンジン(population.py)が担保。ここは近似（線形ブレンドの擬似copula）。
(function(global){
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function clamp(x,a,b){return x<a?a:x>b?b:x}
function randn(rng){let u=1-rng(),v=rng();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function normcdf(x){const t=1/(1+0.2316419*Math.abs(x)),d=0.3989423*Math.exp(-x*x/2);
  let p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return x>0?1-p:p}
function norminv(p){
  const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00];
  const b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01];
  const c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00];
  const d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
  const pl=0.02425,ph=1-pl;let q,r;
  if(p<pl){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
  if(p<=ph){q=p-0.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)}
  q=Math.sqrt(-2*Math.log(1-p));return-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
function corrUniform(rank,rho,rng){const z=rho*norminv(clamp(rank,1e-4,1-1e-4))+Math.sqrt(1-rho*rho)*randn(rng);return clamp(normcdf(z),1e-4,0.9999)}
function pick(dist,rng,allowed){ // 重み付き抽選。allowed(Set)があれば制限
  let keys=Object.keys(dist).filter(k=>!allowed||allowed.has(k));
  if(!keys.length)keys=Object.keys(dist);
  let ws=keys.map(k=>dist[k]),s=ws.reduce((a,b)=>a+b,0)||1,u=rng()*s,c=0;
  for(let i=0;i<keys.length;i++){c+=ws[i];if(u<c)return keys[i]}return keys[keys.length-1];
}
function pickOrdered(dist,order,rank,rho,rng,allowed){ // 順序尺度を rank と相関させて抽選
  let keys=order.filter(k=>k in dist&&(!allowed||allowed.has(k)));
  if(!keys.length)keys=order.filter(k=>k in dist);
  let ws=keys.map(k=>dist[k]),s=ws.reduce((a,b)=>a+b,0)||1;ws=ws.map(w=>w/s);
  let u=corrUniform(rank,rho,rng),c=0;
  for(let i=0;i<keys.length;i++){c+=ws[i];if(u<c)return keys[i]}return keys[keys.length-1];
}
function tilt(p,driver,k){const x=k*(driver-0.5);return p/(p+(1-p)*Math.exp(-x))} // オッズ傾斜（周辺分布を保つ）
function multiSel(dist,driver,strength,rng,force){ // 複数選択。なしは排他。force=強制値(配列)
  if(force)return force;
  const none=dist["なし"]||0;
  if(rng()<none)return["なし"];
  const out=[];
  for(const k in dist){if(k==="なし")continue;
    let p=tilt(dist[k]/(1-none||1),driver,strength*2);
    if(rng()<p)out.push(k)}
  return out.length?out:["なし"];
}
function carMaxent(mean){ // 平均台数に合わせた 0/1/2/3+ 最大エントロピー分布
  const ks=[0,1,2,3.3],labels=["0","1","2","3+"];let m=clamp(mean,0.02,3.2),lo=-6,hi=6;
  for(let i=0;i<60;i++){let th=(lo+hi)/2,w=ks.map(k=>Math.exp(th*k)),s=w.reduce((a,b)=>a+b,0);
    let cur=ks.reduce((a,k,j)=>a+w[j]/s*k,0);if(cur<m)lo=th;else hi=th}
  let th=(lo+hi)/2,w=ks.map(k=>Math.exp(th*k)),s=w.reduce((a,b)=>a+b,0),d={};
  labels.forEach((l,j)=>d[l]=w[j]/s);return d;
}
function rankIn(order,val){let i=order.indexOf(val);return i<0?0.5:(i+0.5)/order.length}

function genOne(G,F,rng){ // G=genspec, F=filters(各fieldはSet or null)
  F=F||{};const A=k=>F[k]; // allowed set for a field
  // psych由来の属性制約を F に畳み込む（derived）
  // prefecture, household
  const pref=pick(G.pref_weights,rng,A("都道府県"));
  const size=pick(G.household_size_dist,rng,A("世帯人員"));
  // 子
  let child;const childRate=clamp(G.child_rate_national/0.586,0,0.95);
  if(size==="1")child="なし";else child=rng()<childRate?"あり":"なし";
  if(A("子")){child=[...A("子")][0]} // フィルタ優先（単身×子ありは稀だが尊重）
  // 年収五分位（人口五分位＝各20%）
  const q=pick({"I":.2,"II":.2,"III":.2,"IV":.2,"V":.2},rng,A("年収五分位"));
  const band=G.income_bands[q];const within=rng();const incYen=Math.round(band[0]+(band[1]-band[0])*within);
  const incRank=(["I","II","III","IV","V"].indexOf(q)+within)/5;
  // ペット
  const pet=pick(G.pet_by_size[size]||G.pet_by_size["2"],rng,A("ペット"));
  // 共働き（子ありのみ）
  let dual;if(child==="あり")dual=rng()<clamp(G.dual_income_rate*(0.8+0.4*incRank),0,.97)?"共働き":"専業中心";else dual="不明";
  if(A("共働き"))dual=[...A("共働き")][0];
  // 学歴・職業・業種
  const edu=pickOrdered(G.education_dist,G.edu_order,incRank,0.45,rng,A("学歴"));
  const indF=A("業種"),indForceEmp=indF&&!indF.has("非該当");
  const empSet=new Set(G.employed);
  let occ=pick(G.occupation_dist,rng,indForceEmp?empSet:A("職業"));
  let ind;
  if(empSet.has(occ))ind=pick(G.industry_dist,rng,indF);
  else ind=indForceEmp?"非該当":"非該当";
  if(indF&&!indF.has(ind)&&empSet.has(occ))ind=pick(G.industry_dist,rng,indF);
  if(indF&&!indF.has(ind)){ // 非該当を要求なら職業を非就業へ
    ind="非該当";}
  // 車（県別平均）
  const carMean=G.car_by_pref[pref]||G.car_national;
  const car=pick(carMaxent(carMean),rng,A("車"));
  // 住居
  const houseOrder=Object.keys(G.housing_dist);
  const house=pickOrdered(G.housing_dist,houseOrder,incRank,0.38,rng,A("住居"));
  // 金融資産（世帯類型別なし率＋所得相関）
  const nashi=size==="1"?G.financial_nashi["単身"]:G.financial_nashi["二人以上"];
  const finBands=Object.keys(G.financial_dist);
  const finDist={"なし":nashi};finBands.forEach(k=>finDist[k]=(1-nashi)*G.financial_dist[k]);
  const finOrder=["なし",...finBands];
  const fin=pickOrdered(finDist,finOrder,incRank,0.72,rng,A("金融資産"));
  // 年齢・性別・配偶・免許
  const age=pickOrdered(G.age_dist,G.age_order,incRank,0.42,rng,A("年齢帯"));
  const ageRank=rankIn(G.age_order,age),young=1-ageRank;
  const gender=pick(G.gender_dist,rng,A("性別"));
  const marOrder=Object.keys(G.marital_by_gender[gender]||G.marital_by_gender["男性"]);
  const marital=pickOrdered(G.marital_by_gender[gender]||G.marital_by_gender["男性"],marOrder,ageRank,0.5,rng,A("配偶"));
  const dl=pick(G.dl_by_gender[gender]||G.dl_by_gender["男性"],rng,A("運転免許"));
  // デジタル
  const net=pickOrdered(G.internet_dist,Object.keys(G.internet_dist),ageRank,0.45,rng,A("ネット頻度"));
  const ec=pickOrdered(G.ec_dist,Object.keys(G.ec_dist),incRank,0.30,rng,A("EC"));
  const sp=pick(G.smartphone_dist,rng,A("スマホ"));
  const sns=multiSel(G.sns_dist,young,0.64,rng);
  const inv=multiSel(G.investment_dist,incRank,0.56,rng);
  const cc=pickOrdered(G.credit_dist,Object.keys(G.credit_dist),incRank,0.30,rng,A("クレカ"));
  // 氏名
  const N=G.names,s=N.sei[(rng()*N.sei.length)|0];
  const mpool=gender==="男性"?N.mei_m:N.mei_f;const m=mpool[(rng()*mpool.length)|0];
  const p={ name:s+" "+m,
    basic:{性別:gender,年齢帯:age,都道府県:pref,配偶:marital,学歴:edu,業種:ind,職業:occ,
      世帯人員:size,子:child,年収五分位:q,年収万円:Math.round(incYen/10000)},
    detail:{ペット:pet,金融資産:fin,運転免許:dl,車:car,ネット頻度:net,EC:ec,スマホ:sp,SNS:sns,
      投資:inv,保険:null,住居:house,クレカ:cc,共働き:dual},
    _incRank:incRank,_ageRank:ageRank,_young:young};
  // 保険（学資は子あり限定）
  const insBase=Object.assign({},G.insurance_dist);
  if(child!=="あり")insBase["学資保険"]=0;
  p.detail.保険=multiSel(insBase,incRank,0.3,rng);
  return p;
}

// 心理フラグ付与（psych_items の規則、derived は確定、probabilistic は条件付き確率）
function assignPsych(G,p,rng){
  const ps=G.psych,flags={},tiers={};
  const D={no_savings:()=>p.detail.金融資産==="なし",low_income:()=>["I","II"].includes(p.basic.年収五分位),
    uninsured:()=>(p.detail.保険||[]).includes("なし"),no_car:()=>p.detail.車==="0",
    ec_nonuser:()=>p.detail.EC==="非利用"};
  ps.derived.forEach(d=>{flags[d.id]=!!D[d.id]();tiers[d.id]=d.tier});
  const attr=f=>({...p.basic,...p.detail})[f];
  ps.probabilistic.forEach(it=>{let pr=it.base;
    (it.conditions||[]).forEach(c=>{let ok=true;for(const f in c.if){const w=c.if[f];const v=attr(f);
      ok=ok&&(Array.isArray(w)?w.includes(v):v===w)}if(ok)pr*=c.mult});
    flags[it.id]=rng()<clamp(pr,0.01,0.95);tiers[it.id]=it.tier});
  p.psych=flags;p.psych_tier=tiers;return p;
}

// フィルタに合う人をcount名生成。derived心理フィルタは属性制約へ、probabilistic心理は強制true。
function genMatching(G,filters,psychReq,count,seed){
  const rng=mulberry32(seed||12345);
  const F={};for(const g of["basic","detail"])for(const f in filters[g])F[f]=filters[g][f];
  // derived心理→属性制約
  const derivedMap={no_savings:["金融資産",["なし"]],low_income:["年収五分位",["I","II"]],
    no_car:["車",["0"]],ec_nonuser:["EC",["非利用"]]};
  const forceUninsured=psychReq.has("uninsured");
  psychReq.forEach(id=>{if(derivedMap[id]){const[f,vs]=derivedMap[id];
    F[f]=new Set((F[f]?[...F[f]].filter(x=>vs.includes(x)):vs));}});
  const probReq=[...psychReq].filter(id=>G.psych.probabilistic.find(x=>x.id===id));
  const out=[];let tries=0;
  while(out.length<count&&tries<count*6){tries++;
    const p=genOne(G,F,rng);
    if(forceUninsured)p.detail.保険=["なし"];
    assignPsych(G,p,rng);
    probReq.forEach(id=>p.psych[id]=true); // 仮説：その障壁を持つ人を集める
    p.id=out.length;out.push(p);}
  return out;
}

const api={mulberry32,genOne,assignPsych,genMatching,pick,pickOrdered};
if(typeof module!=="undefined")module.exports=api;
global.PersonaGen=api;
})(typeof window!=="undefined"?window:globalThis);
