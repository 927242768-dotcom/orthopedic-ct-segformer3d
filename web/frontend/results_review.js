const resultState = {
  evaluations: [],
  currentEvaluation: null,
  cases: [],
};

const el = (id) => document.getElementById(id);

function setMessage(text, type = "") {
  const node = el("resultMessage");
  node.textContent = text || "";
  node.className = `message ${type}`.trim();
}

function setEvaluationBadge(text, type = "work") {
  const node = el("evaluationBadge");
  node.textContent = text;
  node.className = `badge badge-${type}`;
}

function currentCase() {
  const id = el("caseSelect").value;
  return resultState.cases.find((item) => item.case_id === id) || null;
}

function metricText(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(4);
  }
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

function renderMetrics(item) {
  const grid = el("metricsGrid");
  grid.innerHTML = "";
  if (!item) {
    grid.innerHTML = '<div class="empty-state">等待评估结果</div>';
    return;
  }
  const ignored = new Set(["case_id", "prediction_available", "uncertainty_available"]);
  const entries = Object.entries(item).filter(([key]) => !ignored.has(key));
  if (!entries.length) {
    grid.innerHTML = '<div class="empty-state">该病例没有数值指标</div>';
    return;
  }
  for (const [key, value] of entries) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const label = document.createElement("small");
    label.textContent = key;
    const number = document.createElement("strong");
    number.textContent = metricText(value);
    card.append(label, number);
    grid.appendChild(card);
  }
}

function renderCurrentCase() {
  const item = currentCase();
  el("caseTitle").textContent = item?.case_id || "等待真实评估结果";
  renderMetrics(item);
  if (!item) {
    el("artifactBadge").textContent = "未加载";
    return;
  }
  const parts = [];
  if (item.prediction_available) parts.push("prediction");
  if (item.uncertainty_available) parts.push("uncertainty");
  el("artifactBadge").textContent = parts.length ? parts.join(" + ") : "无影像产物";
}

async function loadEvaluations() {
  setEvaluationBadge("扫描评估目录…", "work");
  try {
    const response = await fetch("/api/research/evaluations");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    resultState.evaluations = data.evaluations || [];
    const select = el("evaluationSelect");
    select.innerHTML = "";
    if (!resultState.evaluations.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "暂无真实 evaluate.py 输出";
      select.appendChild(option);
      el("caseSelect").innerHTML = '<option value="">无病例</option>';
      setEvaluationBadge("0 个真实评估", "work");
      setMessage("当前 experiments 下没有完整的 evaluate.py 输出；这是正常状态，不会用随机权重伪造结果。", "work");
      renderCurrentCase();
      return;
    }
    for (const item of resultState.evaluations) {
      const option = document.createElement("option");
      option.value = item.evaluation_id;
      const split = item.split || "unknown";
      option.textContent = `${item.evaluation_id} · ${split} · ${item.case_count} cases`;
      select.appendChild(option);
    }
    setEvaluationBadge(`${resultState.evaluations.length} 个评估`, "ok");
    await loadEvaluationDetail();
  } catch (error) {
    setEvaluationBadge("评估目录不可用", "error");
    setMessage(`加载失败：${error.message}`, "error");
  }
}

async function loadEvaluationDetail() {
  const evaluationId = el("evaluationSelect").value;
  if (!evaluationId) return;
  setMessage("正在读取逐病例指标…", "work");
  try {
    const response = await fetch(`/api/research/evaluations/${encodeURIComponent(evaluationId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    resultState.currentEvaluation = data;
    resultState.cases = data.cases || [];
    const select = el("caseSelect");
    select.innerHTML = "";
    for (const item of resultState.cases) {
      const option = document.createElement("option");
      option.value = item.case_id;
      option.textContent = item.case_id;
      select.appendChild(option);
    }
    if (!resultState.cases.length) {
      select.innerHTML = '<option value="">无病例</option>';
      setMessage("该评估没有逐病例指标。", "error");
      renderCurrentCase();
      return;
    }
    renderCurrentCase();
    setMessage(
      `已读取 ${resultState.cases.length} 例；split=${data.split || "unknown"}。Web 只展示磁盘结果，不重新计算指标。`,
      "ok",
    );
    refreshImage();
  } catch (error) {
    setMessage(`读取评估失败：${error.message}`, "error");
  }
}

function refreshImage() {
  const evaluationId = el("evaluationSelect").value;
  const item = currentCase();
  if (!evaluationId || !item) return;
  const mode = el("modeSelect").value;
  const available = mode === "prediction" ? item.prediction_available : item.uncertainty_available;
  const image = el("resultImage");
  const placeholder = el("imagePlaceholder");
  image.classList.add("hidden");
  placeholder.classList.remove("hidden");
  if (!available) {
    placeholder.textContent = `该病例没有 ${mode} NIfTI 产物。`;
    return;
  }
  const plane = el("planeSelect").value;
  const position = Number(el("positionRange").value) / 100;
  const alpha = Number(el("alphaRange").value) / 100;
  const params = new URLSearchParams({
    mode,
    plane,
    position: String(position),
    alpha: String(alpha),
    t: String(Date.now()),
  });
  placeholder.textContent = `正在加载 ${mode} · ${plane}…`;
  image.onload = () => {
    placeholder.classList.add("hidden");
    image.classList.remove("hidden");
  };
  image.onerror = () => {
    image.classList.add("hidden");
    placeholder.classList.remove("hidden");
    placeholder.textContent = "评估 overlay 加载失败，请检查 prediction/uncertainty 与处理后 CT 的物理空间。";
  };
  image.src = `/api/research/evaluations/${encodeURIComponent(evaluationId)}/cases/${encodeURIComponent(item.case_id)}/mpr?${params.toString()}`;
}

el("evaluationSelect").addEventListener("change", loadEvaluationDetail);
el("caseSelect").addEventListener("change", () => {
  renderCurrentCase();
  refreshImage();
});
el("modeSelect").addEventListener("change", refreshImage);
el("planeSelect").addEventListener("change", refreshImage);
el("positionRange").addEventListener("input", () => {
  el("positionText").textContent = `${el("positionRange").value}%`;
});
el("alphaRange").addEventListener("input", () => {
  el("alphaText").textContent = `${el("alphaRange").value}%`;
});
el("refreshBtn").addEventListener("click", refreshImage);

loadEvaluations();
