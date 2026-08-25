const state = {
  caseId: null,
};

const $ = (id) => document.getElementById(id);

function setMessage(element, text, type = "") {
  element.textContent = text || "";
  element.className = `message ${type}`.trim();
}

function setHealth(data, ok) {
  const badge = $("healthBadge");
  if (ok) {
    badge.textContent = data.inference_ready ? "系统与模型就绪" : "Web 就绪 · 模型待训练";
    badge.className = data.inference_ready ? "badge badge-ok" : "badge badge-work";
  } else {
    badge.textContent = "后端不可用";
    badge.className = "badge badge-error";
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    setHealth(data, true);
  } catch (error) {
    setHealth({}, false);
    setMessage($("uploadMessage"), `无法连接后端：${error.message}`, "error");
  }
}

async function uploadCase() {
  const input = $("caseFiles");
  const files = Array.from(input.files || []);
  if (!files.length) {
    setMessage($("uploadMessage"), "请先选择 DICOM 文件或单个 NIfTI。", "error");
    return;
  }

  const form = new FormData();
  for (const file of files) form.append("files", file);

  const btn = $("uploadBtn");
  btn.disabled = true;
  setMessage($("uploadMessage"), `正在上传并建立病例（${files.length} 个文件）…`, "work");

  try {
    const response = await fetch("/api/cases/upload", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));

    state.caseId = data.case_id;
    $("caseId").textContent = data.case_id;
    setMessage($("uploadMessage"), `上传完成，共 ${data.file_count} 个文件。正在执行基础质控…`, "ok");
    await inspectCase();
    refreshPreview();
  } catch (error) {
    setMessage($("uploadMessage"), `上传失败：${error.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

async function inspectCase() {
  if (!state.caseId) return;
  const response = await fetch(`/api/cases/${encodeURIComponent(state.caseId)}/inspect`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || JSON.stringify(data));
  }

  $("inspectionEmpty").classList.add("hidden");
  const panel = $("inspectionJson");
  panel.classList.remove("hidden");
  panel.textContent = JSON.stringify(data, null, 2);
}

function refreshPreview() {
  if (!state.caseId) {
    setMessage($("uploadMessage"), "请先上传病例。", "error");
    return;
  }

  const center = Number($("windowCenter").value || 500);
  const width = Number($("windowWidth").value || 2000);
  const position = Number($("slicePosition").value || 50) / 100;
  if (!(width > 0)) {
    setMessage($("uploadMessage"), "窗宽必须大于 0。", "error");
    return;
  }

  const placeholder = $("previewPlaceholder");
  const grid = $("mprGrid");
  placeholder.textContent = "正在生成轴位 / 冠状位 / 矢状位预览…";
  placeholder.classList.remove("hidden");
  grid.classList.add("hidden");

  const planes = [
    ["axial", $("previewAxial")],
    ["coronal", $("previewCoronal")],
    ["sagittal", $("previewSagittal")],
  ];
  let loaded = 0;
  let failed = false;
  for (const [plane, img] of planes) {
    img.onload = () => {
      loaded += 1;
      if (loaded === planes.length && !failed) {
        placeholder.classList.add("hidden");
        grid.classList.remove("hidden");
      }
    };
    img.onerror = () => {
      failed = true;
      placeholder.textContent = `MPR ${plane} 预览失败，请检查病例空间信息。`;
    };
    img.src = `/api/cases/${encodeURIComponent(state.caseId)}/preview?center=${encodeURIComponent(center)}&width=${encodeURIComponent(width)}&plane=${plane}&position=${encodeURIComponent(position)}&t=${Date.now()}`;
  }
}

async function inferCase() {
  if (!state.caseId) {
    setMessage($("inferMessage"), "请先上传并检查病例。", "error");
    return;
  }

  setMessage($("inferMessage"), "正在检查模型推理接口…", "work");
  try {
    const response = await fetch(`/api/cases/${encodeURIComponent(state.caseId)}/infer`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      setMessage($("inferMessage"), data.message || data.detail || "模型尚未就绪。", "work");
      return;
    }
    setMessage($("inferMessage"), "模型推理完成。", "ok");
  } catch (error) {
    setMessage($("inferMessage"), `请求失败：${error.message}`, "error");
  }
}

function clearUi() {
  state.caseId = null;
  $("caseFiles").value = "";
  $("caseId").textContent = "尚未创建病例";
  $("inspectionEmpty").classList.remove("hidden");
  $("inspectionJson").classList.add("hidden");
  $("inspectionJson").textContent = "";
  $("mprGrid").classList.add("hidden");
  for (const id of ["previewAxial", "previewCoronal", "previewSagittal"]) {
    $(id).removeAttribute("src");
  }
  $("previewPlaceholder").classList.remove("hidden");
  $("previewPlaceholder").textContent = "等待病例上传";
  setMessage($("uploadMessage"), "");
  setMessage($("inferMessage"), "");
}

$("uploadBtn").addEventListener("click", uploadCase);
$("clearBtn").addEventListener("click", clearUi);
$("refreshPreviewBtn").addEventListener("click", refreshPreview);
$("slicePosition").addEventListener("change", refreshPreview);
$("inferBtn").addEventListener("click", inferCase);

checkHealth();
