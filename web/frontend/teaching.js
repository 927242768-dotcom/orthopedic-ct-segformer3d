const $=id=>document.getElementById(id);
let evaluations=[],currentEval=null,currentCase=null,geometry=null,currentMode="raw",renderTimer=null,locateArmed=false,traceActive=false,traceStartZ=.5,traceMaxDelta=0;
const coord={x:.5,y:.5,z:.5};
const clamp=v=>Math.max(0,Math.min(1,v));
const pct=v=>Math.round(v*100)+"%";
async function json(url){const r=await fetch(url);const d=await r.json();if(!r.ok)throw new Error(d.detail||"加载失败");return d}
function option(v,t){const o=document.createElement("option");o.value=v;o.textContent=t;return o}
function bestEvaluation(){return evaluations.find(e=>e.case_count===197&&String(e.display_name||"").includes("20260917_185941"))||evaluations.find(e=>e.case_count===197)||evaluations[0]}
function baseUrl(mode,plane,pos){return "/api/research/evaluations/"+encodeURIComponent(currentEval.evaluation_id)+"/cases/"+encodeURIComponent(currentCase.case_id)+"/mpr?mode="+mode+"&plane="+plane+"&position="+pos.toFixed(4)+"&alpha=0.52&_="+Date.now()}
function imageRect(stage){
  const img=stage.querySelector("img"),sw=stage.clientWidth,sh=stage.clientHeight;
  const iw=img?.naturalWidth||sw,ih=img?.naturalHeight||sh;
  const scale=Math.min(sw/Math.max(iw,1),sh/Math.max(ih,1));
  const width=iw*scale,height=ih*scale;
  return {left:(sw-width)/2,top:(sh-height)/2,width,height};
}
function updateCrosshairs(){document.querySelectorAll(".mpr-stage").forEach(stage=>{
  const v=stage.querySelector(".cross-v"),h=stage.querySelector(".cross-h"),p=stage.dataset.plane,r=imageRect(stage);
  let u=.5,w=.5;
  if(p==="axial"){u=coord.x;w=coord.y}
  if(p==="coronal"){u=coord.x;w=1-coord.z}
  if(p==="sagittal"){u=coord.y;w=1-coord.z}
  v.style.left=(r.left+u*r.width)+"px";v.style.top=r.top+"px";v.style.height=r.height+"px";v.style.bottom="auto";
  h.style.top=(r.top+w*r.height)+"px";h.style.left=r.left+"px";h.style.width=r.width+"px";h.style.right="auto";
})}
function updateCoordUi(){["x","y","z"].forEach(a=>{$(a+"Slider").value=Math.round(coord[a]*1000);$(a+"Text").textContent=pct(coord[a])});if(geometry){const s=geometry.size_xyz||[0,0,0],sp=geometry.spacing_xyz_mm||[0,0,0],ix=Math.round(coord.x*Math.max(0,s[0]-1)),iy=Math.round(coord.y*Math.max(0,s[1]-1)),iz=Math.round(coord.z*Math.max(0,s[2]-1));$("coordIndex").textContent="X "+ix+"/"+Math.max(0,s[0]-1)+" · Y "+iy+"/"+Math.max(0,s[1]-1)+" · Z "+iz+"/"+Math.max(0,s[2]-1);$("coordMm").textContent="体素间距 "+sp.map(v=>Number(v).toFixed(2)).join(" × ")+" mm"}updateCrosshairs()}
function modeLabel(){return currentMode==="raw"?"原始 CT · 自主观察":currentMode==="prediction"?"AI 辅助定位 · 蓝色高亮":"专家参考标注 · 绿色区域"}
function setMode(mode){currentMode=mode;$("teachMode").textContent=modeLabel();$("rawBtn").classList.toggle("primary",mode==="raw");$("aiToggle").classList.toggle("primary",mode==="prediction");$("expertToggle").classList.toggle("primary",mode==="gt");scheduleRender()}
function scheduleRender(){clearTimeout(renderTimer);renderTimer=setTimeout(renderViews,70)}
function renderViews(){if(!currentEval||!currentCase)return;updateCoordUi();$("teachAxial").src=baseUrl(currentMode,"axial",coord.z);$("teachCoronal").src=baseUrl(currentMode,"coronal",coord.y);$("teachSagittal").src=baseUrl(currentMode,"sagittal",coord.x);$("axialIndex").textContent="Z "+pct(coord.z);$("coronalIndex").textContent="Y "+pct(coord.y);$("sagittalIndex").textContent="X "+pct(coord.x);if(traceActive){traceMaxDelta=Math.max(traceMaxDelta,Math.abs(coord.z-traceStartZ));$("traceFeedback").textContent=traceMaxDelta>=.2?"完成：已连续浏览 "+Math.round(traceMaxDelta*100)+"% 体数据范围，可比较结构出现、变化与消失。":"已移动 "+Math.round(traceMaxDelta*100)+"%，继续沿 Z 方向浏览至至少 20%。";if(traceMaxDelta>=.2){traceActive=false;$("traceFeedback").classList.add("success")}}}
async function loadCaseGeometry(){geometry=await json("/api/research/evaluations/"+encodeURIComponent(currentEval.evaluation_id)+"/cases/"+encodeURIComponent(currentCase.case_id)+"/geometry");updateCoordUi()}
async function loadEvaluation(meta){currentEval=await json("/api/research/evaluations/"+encodeURIComponent(meta.evaluation_id));$("teachCase").innerHTML="";(currentEval.cases||[]).forEach(c=>$("teachCase").append(option(c.case_id,c.case_id)));currentCase=(currentEval.cases||[])[0]||null;if(!currentCase)return;$("teachCase").value=currentCase.case_id;await loadCaseGeometry();$("teachSource").textContent=(currentEval.case_count||0)+" 例 "+(currentEval.split||"")+" · 当前病例 "+currentCase.case_id+" · 主模型 v7";update3dLink();renderViews()}
function update3dLink(){$("to3d").href="/research-3d?evaluation="+encodeURIComponent(currentEval.evaluation_id)+"&case="+encodeURIComponent(currentCase.case_id)+"&mode=teaching"}
function setCoord(axis,value){coord[axis]=clamp(value);scheduleRender()}
function clickToCoord(stage,event){
  const box=stage.getBoundingClientRect(),r=imageRect(stage);
  const lx=event.clientX-box.left-r.left,ly=event.clientY-box.top-r.top;
  if(lx<0||ly<0||lx>r.width||ly>r.height)return;
  const nx=clamp(lx/r.width),ny=clamp(ly/r.height),p=stage.dataset.plane;
  if(p==="axial"){coord.x=nx;coord.y=ny}
  if(p==="coronal"){coord.x=nx;coord.z=1-ny}
  if(p==="sagittal"){coord.y=nx;coord.z=1-ny}
  scheduleRender();if(locateArmed)checkTeachingHit()
}
async function checkTeachingHit(){locateArmed=false;try{const u="/api/research/evaluations/"+encodeURIComponent(currentEval.evaluation_id)+"/cases/"+encodeURIComponent(currentCase.case_id)+"/teaching-hit?x="+coord.x.toFixed(5)+"&y="+coord.y.toFixed(5)+"&z="+coord.z.toFixed(5),d=await json(u);$("locateFeedback").textContent=d.hit?"定位正确：该三维点命中专家参考结构。现在可以打开 AI 辅助和专家参考核对。":"未命中专家参考结构。先观察周围连续切片，必要时打开 AI 辅助后再次尝试。";$("locateFeedback").classList.toggle("success",!!d.hit);$("locateFeedback").classList.toggle("warning",!d.hit)}catch(e){$("locateFeedback").textContent="定位反馈失败："+e.message}}
document.querySelectorAll("[data-answer]").forEach(b=>b.addEventListener("click",()=>{const ok=b.dataset.answer==="axial";$("quizFeedback").textContent=ok?"正确：Axial 轴位由 Z 坐标控制。":"再想一下：哪一个平面的切片位置由 Z 坐标决定？";$("quizFeedback").className="task-feedback "+(ok?"success":"warning")}));
["x","y","z"].forEach(a=>$(a+"Slider").addEventListener("input",e=>setCoord(a,Number(e.target.value)/1000)));
document.querySelectorAll(".mpr-stage").forEach(s=>s.addEventListener("click",e=>clickToCoord(s,e)));
["teachAxial","teachCoronal","teachSagittal"].forEach(id=>$(id).addEventListener("load",updateCrosshairs));
window.addEventListener("resize",updateCrosshairs);
$("rawBtn").addEventListener("click",()=>setMode("raw"));$("aiToggle").addEventListener("click",()=>setMode("prediction"));$("expertToggle").addEventListener("click",()=>setMode("gt"));
$("locateStart").addEventListener("click",()=>{setMode("raw");locateArmed=true;$("locateFeedback").className="task-feedback";$("locateFeedback").textContent="请在任一 CT 平面点击你判断的目标骨性结构。"});
$("locateAi").addEventListener("click",()=>{setMode("prediction");locateArmed=true;$("locateFeedback").textContent="AI 辅助已打开。观察蓝色高亮后，再点击你认为正确的位置。"});
$("traceStart").addEventListener("click",()=>{setMode("raw");traceActive=true;traceStartZ=coord.z;traceMaxDelta=0;$("traceFeedback").className="task-feedback";$("traceFeedback").textContent="开始：沿 Z 方向移动轴位切片，目标至少 20%。"});
$("teachCase").addEventListener("change",async e=>{currentCase=(currentEval.cases||[]).find(c=>c.case_id===e.target.value);coord.x=coord.y=coord.z=.5;await loadCaseGeometry();update3dLink();renderViews()});
(async()=>{const d=await json("/api/research/evaluations");evaluations=d.evaluations||[];const meta=bestEvaluation();if(!meta)throw new Error("没有找到可用评估病例");await loadEvaluation(meta)})().catch(e=>$("teachSource").textContent="教学数据加载失败："+e.message);