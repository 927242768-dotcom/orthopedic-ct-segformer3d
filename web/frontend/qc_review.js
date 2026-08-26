const qcState = {
  cases: [],
  index: 0,
  caseListCollapsed: false,
  mprObjectUrl: null,
  mprRequestId: 0,
  mprTimer: null,
};

const q = (id) => document.getElementById(id);

function setReviewMessage(text, type = "") {
  const element = q("reviewMessage");
  element.textContent = text || "";
  element.className = `message ${type}`.trim();
}

function statusLabel(status) {
  if (status === "pass") return "已通过";
  if (status === "needs_review") return "需复核";
  if (status === "fail") return "不通过";
  return "未审核";
}

function statusClass(status) {
  if (status === "pass") return "case-pass";
  if (status === "needs_review") return "case-review";
  if (status === "fail") return "case-fail";
  return "case-pending";
}

function updateProgress() {
  const reviewed = qcState.cases.filter((item) => item.review_status).length;
  const passed = qcState.cases.filter((item) => item.review_status === "pass").length;
  const badge = q("progressBadge");
  badge.textContent = `已审核 ${reviewed}/${qcState.cases.length} · 通过 ${passed}`;
  badge.className = reviewed === qcState.cases.length ? "badge badge-ok" : "badge badge-work";
}

function enterReviewArea({ smooth = true } = {}) {
  requestAnimationFrame(() => {
    q("qcMain").scrollIntoView({
      behavior: smooth ? "smooth" : "auto",
      block: "start",
    });
  });
}

function setCaseListCollapsed(collapsed, { enterReview = false, smooth = true } = {}) {
  const sidebar = q("caseSidebar");
  qcState.caseListCollapsed = collapsed;
  q("qcLayout").classList.toggle("case-list-collapsed", collapsed);
  sidebar.hidden = collapsed;
  sidebar.classList.toggle("is-collapsed", collapsed);
  sidebar.setAttribute("aria-hidden", String(collapsed));
  q("caseListToggleBtn").setAttribute("aria-expanded", String(!collapsed));
  q("caseListToggleBtn").textContent = collapsed ? "显示病例列表" : "收起病例列表";
  if (enterReview) enterReviewArea({ smooth });
}

function showCaseList() {
  setCaseListCollapsed(false);
  requestAnimationFrame(() => {
    q("caseSidebar").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function selectCase(index, { smooth = true } = {}) {
  if (index < 0 || index >= qcState.cases.length) return;
  qcState.index = index;
  setCaseListCollapsed(true);
  loadCurrentCase();
  enterReviewArea({ smooth });
}

function renderCaseList() {
  const list = q("caseList");
  list.innerHTML = "";
  qcState.cases.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-item ${statusClass(item.review_status)} ${index === qcState.index ? "active" : ""}`;
    button.innerHTML = `
      <span class="case-index">${String(index + 1).padStart(2, "0")}</span>
      <span class="case-copy"><strong>${item.case_id}</strong><small>${statusLabel(item.review_status)}</small></span>
      <span class="case-dot"></span>
    `;
    button.addEventListener("click", () => selectCase(index));
    list.appendChild(button);
  });
}

function setCheckbox(id, value) {
  q(id).checked = value === true;
}

async function refreshMpr() {
  if (!qcState.cases.length) return;
  const requestId = ++qcState.mprRequestId;
  const item = qcState.cases[qcState.index];
  const plane = q("mprPlane").value;
  const percent = Number(q("mprPosition").value || 50);
  const position = Math.max(0, Math.min(100, percent)) / 100;
  const overlay = q("mprOverlay").checked;
  q("mprPositionText").textContent = `${Math.round(position * 100)}%`;

  const image = q("mprImage");
  const placeholder = q("mprPlaceholder");
  image.classList.add("hidden");
  placeholder.classList.remove("hidden");
  placeholder.textContent = "正在生成交互式 MPR…";
  try {
    const params = new URLSearchParams({
      plane,
      position: String(position),
      overlay: String(overlay),
      alpha: "0.38",
    });
    const response = await fetch(
      `/api/research/qc/${encodeURIComponent(item.case_id)}/mpr?${params.toString()}`,
    );
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    if (requestId !== qcState.mprRequestId) return;
    if (qcState.mprObjectUrl) URL.revokeObjectURL(qcState.mprObjectUrl);
    qcState.mprObjectUrl = URL.createObjectURL(blob);
    image.onload = () => {
      placeholder.classList.add("hidden");
      image.classList.remove("hidden");
    };
    image.src = qcState.mprObjectUrl;
    q("mprIndexText").textContent = `${plane} · index=${response.headers.get("X-MPR-Index") || "?"} · overlay=${overlay ? "on" : "off"}`;
  } catch (error) {
    if (requestId !== qcState.mprRequestId) return;
    placeholder.textContent = `MPR 加载失败：${error.message}`;
    q("mprIndexText").textContent = "—";
  }
}

function scheduleMprRefresh() {
  if (qcState.mprTimer) clearTimeout(qcState.mprTimer);
  qcState.mprTimer = setTimeout(refreshMpr, 100);
}

function loadCurrentCase() {
  if (!qcState.cases.length) return;
  const item = qcState.cases[qcState.index];
  q("caseTitle").textContent = item.case_id;
  q("caseStatus").textContent = statusLabel(item.review_status);
  q("caseStatus").className = `tag ${statusClass(item.review_status)}`;
  q("labelValues").textContent = item.auto_label_display || item.auto_label_values || "—";

  setCheckbox("orientationOk", item.orientation_ok);
  setCheckbox("spacingOk", item.spacing_ok);
  setCheckbox("alignmentOk", item.label_alignment_ok);
  setCheckbox("boneWindowOk", item.bone_window_ok);
  q("reviewStatus").value = item.review_status || "";
  q("reviewer").value = item.reviewer || "";
  q("notes").value = item.notes || "";

  q("prevBtn").disabled = qcState.index <= 0;
  q("nextBtn").disabled = qcState.index >= qcState.cases.length - 1;

  const image = q("qcImage");
  const placeholder = q("imagePlaceholder");
  image.classList.add("hidden");
  placeholder.classList.remove("hidden");
  placeholder.textContent = "正在加载 QC contact sheet…";
  image.onload = () => {
    placeholder.classList.add("hidden");
    image.classList.remove("hidden");
  };
  image.onerror = () => {
    placeholder.textContent = "QC 图加载失败，请检查该病例 qc_contact_sheet.png。";
  };
  image.src = `/api/research/qc/${encodeURIComponent(item.case_id)}/image?t=${Date.now()}`;
  refreshMpr();

  setReviewMessage("");
  renderCaseList();
  updateProgress();
}

async function loadCases() {
  try {
    const response = await fetch("/api/research/qc");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    qcState.cases = data.cases || [];
    if (!qcState.cases.length) throw new Error("没有可审核病例");
    qcState.index = 0;
    loadCurrentCase();
  } catch (error) {
    q("caseList").innerHTML = `<div class="empty-state">${error.message}</div>`;
    q("imagePlaceholder").textContent = `加载失败：${error.message}`;
    q("progressBadge").textContent = "QC 数据不可用";
    q("progressBadge").className = "badge badge-error";
  }
}

async function saveReview(event) {
  event.preventDefault();
  if (!qcState.cases.length) return;
  const current = qcState.cases[qcState.index];
  const payload = {
    orientation_ok: q("orientationOk").checked,
    spacing_ok: q("spacingOk").checked,
    label_alignment_ok: q("alignmentOk").checked,
    bone_window_ok: q("boneWindowOk").checked,
    review_status: q("reviewStatus").value,
    reviewer: q("reviewer").value.trim(),
    notes: q("notes").value.trim(),
  };

  if (!payload.review_status) {
    setReviewMessage("请选择审核结论。", "error");
    return;
  }
  if (!payload.reviewer) {
    setReviewMessage("请输入审核人。", "error");
    return;
  }
  if (payload.review_status === "pass" && ![
    payload.orientation_ok,
    payload.spacing_ok,
    payload.label_alignment_ok,
    payload.bone_window_ok,
  ].every(Boolean)) {
    setReviewMessage("标记“通过”前，四项人工检查必须全部勾选。", "error");
    return;
  }

  const button = q("saveBtn");
  button.disabled = true;
  setReviewMessage("正在保存人工审核记录…", "work");
  try {
    const response = await fetch(`/api/research/qc/${encodeURIComponent(current.case_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    qcState.cases[qcState.index] = data.case;
    renderCaseList();
    updateProgress();
    q("caseStatus").textContent = statusLabel(data.case.review_status);
    q("caseStatus").className = `tag ${statusClass(data.case.review_status)}`;
    setReviewMessage("已保存到本地 manual_qc_review.csv。", "ok");
  } catch (error) {
    setReviewMessage(`保存失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

q("reviewForm").addEventListener("submit", saveReview);
q("caseListToggleBtn").addEventListener("click", () => {
  if (qcState.caseListCollapsed) {
    showCaseList();
  } else {
    setCaseListCollapsed(true, { enterReview: true });
  }
});
q("mprPlane").addEventListener("change", refreshMpr);
q("mprOverlay").addEventListener("change", refreshMpr);
q("mprPosition").addEventListener("input", () => {
  q("mprPositionText").textContent = `${q("mprPosition").value}%`;
  scheduleMprRefresh();
});
q("prevBtn").addEventListener("click", () => {
  if (qcState.index > 0) selectCase(qcState.index - 1);
});
q("nextBtn").addEventListener("click", () => {
  if (qcState.index < qcState.cases.length - 1) selectCase(qcState.index + 1);
});

document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === "ArrowLeft") q("prevBtn").click();
  if (event.key === "ArrowRight") q("nextBtn").click();
});

loadCases();
