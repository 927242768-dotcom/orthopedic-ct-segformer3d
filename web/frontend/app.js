const metricGrid=document.getElementById("metricGrid");const caption=document.getElementById("evalCaption");
const fmt=(v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):"—";
function metric(label,value,note){return '<div class="card metric"><small>'+label+'</small><strong>'+value+'</strong><em>'+note+'</em></div>'}
async function boot(){try{const r=await fetch("/api/research/evaluations");const d=await r.json();if(!r.ok)throw new Error(d.detail||"加载失败");
const v7=(d.evaluations||[]).find(x=>x.case_count===197&&String(x.display_name||"").includes("20260917_185941"))||(d.evaluations||[]).find(x=>x.case_count===197)||(d.evaluations||[])[0];
if(!v7){caption.textContent="当前未发现完整评估产物";metricGrid.innerHTML=metric("状态","暂无","不会使用虚构数据");return}
const m=v7.metrics||{};caption.textContent=v7.case_count+" 例 · "+(v7.split||"unknown")+" · 完成于 "+String(v7.evaluated_at||"").slice(0,10)+" · SegFormer3D HR";
metricGrid.innerHTML=[metric("Dice",fmt(m.dice?.mean,4),"197例均值"),metric("Precision",fmt(m.precision?.mean,4),"精确率"),metric("Recall",fmt(m.recall?.mean,4),"召回率"),metric("HD95",fmt(m.hd95_mm?.mean,2)+" mm","表面距离"),metric("ASSD",fmt(m.assd_mm?.mean,2)+" mm","平均对称表面距离")].join("")}
catch(e){caption.textContent="真实结果接口暂不可用";metricGrid.innerHTML=metric("状态","待启动","请使用一键启动 Demo")}}boot();