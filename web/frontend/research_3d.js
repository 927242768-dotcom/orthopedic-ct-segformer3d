const viewerState = {
  cases: [],
  gl: null,
  program: null,
  positionBuffer: null,
  indexBuffer: null,
  indexCount: 0,
  yaw: -0.55,
  pitch: 0.28,
  distance: 3.25,
  dragging: false,
  lastX: 0,
  lastY: 0,
};

const el = (id) => document.getElementById(id);

function setViewerMessage(text, type = "") {
  const node = el("viewerMessage");
  node.textContent = text || "";
  node.className = `message ${type}`.trim();
}

function setViewerStatus(text, type = "neutral") {
  const node = el("viewerStatus");
  node.textContent = text;
  node.className = `badge badge-${type}`;
}

function selectedClassId() {
  const value = el("classSelect").value;
  return value === "foreground" ? null : Number(value);
}

function selectedSurface() {
  return el("surfaceSelect").value;
}

function selectedSdfSigmaMm() {
  return Number(el("sdfSigmaSelect").value);
}

function selectedSimplifyMm() {
  if (selectedSurface() === "sdf") return null;
  const value = el("simplifySelect").value;
  return value === "full" ? null : Number(value);
}

function meshQuery(classId, simplifyMm, surface, sdfSigmaMm) {
  const params = new URLSearchParams();
  if (classId !== null) params.set("class_id", String(classId));
  if (simplifyMm !== null) params.set("simplify_mm", String(simplifyMm));
  if (surface === "sdf") {
    params.set("surface", "sdf");
    params.set("sdf_sigma_mm", String(sdfSigmaMm));
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

function updateSurfaceControls() {
  const sdf = selectedSurface() === "sdf";
  el("sdfSigmaLabel").classList.toggle("hidden", !sdf);
  el("simplifySelect").disabled = sdf;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`WebGL shader 编译失败：${message}`);
  }
  return shader;
}

function createProgram(gl) {
  const vertexSource = `#version 300 es
    precision highp float;
    in vec3 aPosition;
    uniform mat4 uMV;
    uniform mat4 uMVP;
    out vec3 vViewPos;
    void main() {
      vec4 viewPos = uMV * vec4(aPosition, 1.0);
      vViewPos = viewPos.xyz;
      gl_Position = uMVP * vec4(aPosition, 1.0);
    }
  `;
  const fragmentSource = `#version 300 es
    precision highp float;
    in vec3 vViewPos;
    out vec4 outColor;
    void main() {
      vec3 dx = dFdx(vViewPos);
      vec3 dy = dFdy(vViewPos);
      vec3 normal = normalize(cross(dx, dy));
      if (!gl_FrontFacing) normal = -normal;
      vec3 lightDir = normalize(vec3(0.35, 0.60, 1.0));
      float diffuse = max(dot(normal, lightDir), 0.0);
      float rim = pow(1.0 - abs(normalize(vViewPos).z), 2.0);
      vec3 base = vec3(0.32, 0.68, 0.98);
      vec3 color = base * (0.27 + 0.73 * diffuse) + vec3(0.18, 0.35, 0.55) * rim * 0.32;
      outColor = vec4(color, 1.0);
    }
  `;
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`WebGL program 链接失败：${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

function mat4Identity() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row] * b[column * 4 + k];
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4Translation(x, y, z) {
  const out = mat4Identity();
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

function mat4RotationX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([
    1, 0, 0, 0,
    0, c, s, 0,
    0, -s, c, 0,
    0, 0, 0, 1,
  ]);
}

function mat4RotationY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([
    c, 0, -s, 0,
    0, 1, 0, 0,
    s, 0, c, 0,
    0, 0, 0, 1,
  ]);
}

function parseAsciiPly(text) {
  const marker = "end_header";
  const markerIndex = text.indexOf(marker);
  if (markerIndex < 0) throw new Error("PLY 缺少 end_header");
  const bodyStart = text.indexOf("\n", markerIndex);
  if (bodyStart < 0) throw new Error("PLY header 不完整");
  const header = text.slice(0, markerIndex).split(/\r?\n/);
  let vertexCount = 0;
  let faceCount = 0;
  for (const line of header) {
    const vertexMatch = line.match(/^element\s+vertex\s+(\d+)/);
    const faceMatch = line.match(/^element\s+face\s+(\d+)/);
    if (vertexMatch) vertexCount = Number(vertexMatch[1]);
    if (faceMatch) faceCount = Number(faceMatch[1]);
  }
  if (!(vertexCount > 0) || !(faceCount > 0)) throw new Error("PLY 顶点/面数量非法");

  const lines = text.slice(bodyStart + 1).trim().split(/\r?\n/);
  if (lines.length < vertexCount + faceCount) throw new Error("PLY 数据行数量不足");

  const vertices = new Float32Array(vertexCount * 3);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertexCount; i += 1) {
    const values = lines[i].trim().split(/\s+/);
    if (values.length < 3) throw new Error(`PLY vertex ${i} 格式错误`);
    for (let axis = 0; axis < 3; axis += 1) {
      const value = Number(values[axis]);
      if (!Number.isFinite(value)) throw new Error(`PLY vertex ${i} 包含非有限值`);
      vertices[i * 3 + axis] = value;
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    }
  }

  const indices = new Uint32Array(faceCount * 3);
  for (let i = 0; i < faceCount; i += 1) {
    const values = lines[vertexCount + i].trim().split(/\s+/).map(Number);
    if (values[0] !== 3 || values.length < 4) {
      throw new Error("当前本地查看器只接受三角面 PLY");
    }
    indices[i * 3] = values[1];
    indices[i * 3 + 1] = values[2];
    indices[i * 3 + 2] = values[3];
  }

  const center = min.map((value, axis) => (value + max[axis]) / 2);
  const extent = max.map((value, axis) => value - min[axis]);
  const scale = 2 / Math.max(...extent, 1e-6);
  for (let i = 0; i < vertexCount; i += 1) {
    vertices[i * 3] = (vertices[i * 3] - center[0]) * scale;
    vertices[i * 3 + 1] = (vertices[i * 3 + 1] - center[1]) * scale;
    vertices[i * 3 + 2] = (vertices[i * 3 + 2] - center[2]) * scale;
  }
  return { vertices, indices, vertexCount, faceCount };
}

function initWebGL() {
  const canvas = el("meshCanvas");
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("浏览器不支持 WebGL2");
  viewerState.gl = gl;
  viewerState.program = createProgram(gl);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.clearColor(0.005, 0.025, 0.055, 1.0);

  canvas.addEventListener("pointerdown", (event) => {
    viewerState.dragging = true;
    viewerState.lastX = event.clientX;
    viewerState.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", () => { viewerState.dragging = false; });
  canvas.addEventListener("pointercancel", () => { viewerState.dragging = false; });
  canvas.addEventListener("pointermove", (event) => {
    if (!viewerState.dragging) return;
    const dx = event.clientX - viewerState.lastX;
    const dy = event.clientY - viewerState.lastY;
    viewerState.lastX = event.clientX;
    viewerState.lastY = event.clientY;
    viewerState.yaw += dx * 0.009;
    viewerState.pitch = Math.max(-1.45, Math.min(1.45, viewerState.pitch + dy * 0.009));
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    viewerState.distance *= Math.exp(event.deltaY * 0.0012);
    viewerState.distance = Math.max(1.7, Math.min(9.0, viewerState.distance));
  }, { passive: false });

  requestAnimationFrame(drawFrame);
}

function uploadMesh(mesh) {
  const gl = viewerState.gl;
  if (!gl) throw new Error("WebGL 未初始化");
  if (viewerState.positionBuffer) gl.deleteBuffer(viewerState.positionBuffer);
  if (viewerState.indexBuffer) gl.deleteBuffer(viewerState.indexBuffer);

  viewerState.positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, viewerState.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);

  viewerState.indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, viewerState.indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
  viewerState.indexCount = mesh.indices.length;
}

function resizeCanvas(gl) {
  const canvas = gl.canvas;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
}

function drawFrame() {
  const gl = viewerState.gl;
  if (!gl) return;
  resizeCanvas(gl);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (viewerState.indexCount > 0 && viewerState.positionBuffer && viewerState.indexBuffer) {
    gl.useProgram(viewerState.program);
    const aspect = gl.canvas.width / Math.max(gl.canvas.height, 1);
    const projection = mat4Perspective(Math.PI / 4.2, aspect, 0.05, 50);
    const rotation = mat4Multiply(mat4RotationY(viewerState.yaw), mat4RotationX(viewerState.pitch));
    const mv = mat4Multiply(mat4Translation(0, 0, -viewerState.distance), rotation);
    const mvp = mat4Multiply(projection, mv);

    const positionLocation = gl.getAttribLocation(viewerState.program, "aPosition");
    gl.bindBuffer(gl.ARRAY_BUFFER, viewerState.positionBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, viewerState.indexBuffer);
    gl.uniformMatrix4fv(gl.getUniformLocation(viewerState.program, "uMV"), false, mv);
    gl.uniformMatrix4fv(gl.getUniformLocation(viewerState.program, "uMVP"), false, mvp);
    gl.drawElements(gl.TRIANGLES, viewerState.indexCount, gl.UNSIGNED_INT, 0);
  }
  requestAnimationFrame(drawFrame);
}

function populateClassSelect(caseItem) {
  const select = el("classSelect");
  select.innerHTML = "";
  const foreground = document.createElement("option");
  foreground.value = "foreground";
  foreground.textContent = "全部前景 (>0)";
  select.appendChild(foreground);
  const labelItems = Array.isArray(caseItem.label_items) && caseItem.label_items.length
    ? caseItem.label_items
    : (caseItem.label_values || [])
      .filter((value) => Number(value) > 0)
      .map((value) => ({ value, display: `标签 ${value}` }));
  for (const item of labelItems) {
    const option = document.createElement("option");
    option.value = String(item.value);
    option.textContent = item.display || `标签 ${item.value}`;
    select.appendChild(option);
  }
}

function selectedCase() {
  return viewerState.cases.find((item) => item.case_id === el("caseSelect").value) || null;
}

function updateMeshStats(summary) {
  const nodes = el("meshStats").querySelectorAll("strong");
  if (!summary) {
    nodes.forEach((node) => { node.textContent = "—"; });
    return;
  }
  const bounds = summary.bounds_xyz_mm || {};
  const min = bounds.min || [];
  const max = bounds.max || [];
  const boundsText = min.length === 3 && max.length === 3
    ? `X ${min[0].toFixed(1)}~${max[0].toFixed(1)} · Y ${min[1].toFixed(1)}~${max[1].toFixed(1)} · Z ${min[2].toFixed(1)}~${max[2].toFixed(1)} mm`
    : "—";
  nodes[0].textContent = selectedCase()?.case_id || "—";
  nodes[1].textContent = summary.selection || "—";
  nodes[2].textContent = Number(summary.vertex_count || 0).toLocaleString();
  nodes[3].textContent = Number(summary.face_count || 0).toLocaleString();
  nodes[4].textContent = `${Number(summary.surface_area_mm2 || 0).toLocaleString(undefined, { maximumFractionDigits: 1 })} mm²`;
  nodes[5].textContent = boundsText;
}

async function loadResearchCases() {
  setViewerStatus("加载病例…", "work");
  try {
    const response = await fetch("/api/research/cases");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    viewerState.cases = data.cases || [];
    const select = el("caseSelect");
    select.innerHTML = "";
    for (const item of viewerState.cases) {
      const option = document.createElement("option");
      option.value = item.case_id;
      option.textContent = `${item.case_id} · ${item.source_split}`;
      select.appendChild(option);
    }
    if (!viewerState.cases.length) throw new Error("没有可展示的标准化研究病例");
    populateClassSelect(viewerState.cases[0]);
    setViewerStatus(`${viewerState.cases.length} 例可用`, "ok");
    if (viewerState.cases[0].foreground_mesh_ready) {
      setViewerMessage("首例已存在真实 label mesh，可直接点击“加载 3D”。", "ok");
    }
  } catch (error) {
    setViewerStatus("病例不可用", "error");
    setViewerMessage(`加载病例失败：${error.message}`, "error");
  }
}

async function buildMesh() {
  const item = selectedCase();
  if (!item) return;
  const classId = selectedClassId();
  const surface = selectedSurface();
  const sdfSigmaMm = selectedSdfSigmaMm();
  const simplifyMm = selectedSimplifyMm();
  const button = el("buildBtn");
  button.disabled = true;
  setViewerStatus("正在生成 Mesh…", "work");
  setViewerMessage(
    surface === "sdf"
      ? `正在生成 SDF σ=${sdfSigmaMm} mm 物理表面；连通域变化将被后端拒绝…`
      : "正在从真值 NIfTI label 生成物理空间 Marching-Cubes 网格…",
    "work",
  );
  try {
    const response = await fetch(
      `/api/research/cases/${encodeURIComponent(item.case_id)}/mesh/build${meshQuery(classId, simplifyMm, surface, sdfSigmaMm)}`,
      { method: "POST" },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    updateMeshStats(data.summary);
    setViewerStatus("Mesh 已生成", "ok");
    setViewerMessage("Mesh 已生成，正在加载 WebGL…", "ok");
    await loadMesh();
  } catch (error) {
    setViewerStatus("Mesh 生成失败", "error");
    setViewerMessage(`生成失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function loadMesh() {
  const item = selectedCase();
  if (!item) return;
  const classId = selectedClassId();
  const surface = selectedSurface();
  const sdfSigmaMm = selectedSdfSigmaMm();
  const simplifyMm = selectedSimplifyMm();
  const button = el("loadBtn");
  button.disabled = true;
  setViewerStatus("加载 PLY…", "work");
  setViewerMessage("正在读取并解析本地 ASCII PLY，大网格可能需要数秒…", "work");
  try {
    const summaryResponse = await fetch(
      `/api/research/cases/${encodeURIComponent(item.case_id)}/mesh/summary${meshQuery(classId, simplifyMm, surface, sdfSigmaMm)}`,
    );
    if (!summaryResponse.ok) {
      const detail = await summaryResponse.json().catch(() => ({}));
      throw new Error(detail.detail || "Mesh 尚未生成，请先点击“生成 / 刷新 Mesh”");
    }
    const summary = await summaryResponse.json();

    const meshResponse = await fetch(
      `/api/research/cases/${encodeURIComponent(item.case_id)}/mesh${meshQuery(classId, simplifyMm, surface, sdfSigmaMm)}`,
    );
    if (!meshResponse.ok) {
      const detail = await meshResponse.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${meshResponse.status}`);
    }
    const text = await meshResponse.text();
    const mesh = parseAsciiPly(text);
    uploadMesh(mesh);
    updateMeshStats(summary);
    el("canvasPlaceholder").classList.add("hidden");
    setViewerStatus("3D 已加载", "ok");
    setViewerMessage(
      `已加载 ${mesh.vertexCount.toLocaleString()} 顶点 / ${mesh.faceCount.toLocaleString()} 三角面。拖拽旋转，滚轮缩放。`,
      "ok",
    );
  } catch (error) {
    setViewerStatus("3D 未加载", "error");
    setViewerMessage(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function resetView() {
  viewerState.yaw = -0.55;
  viewerState.pitch = 0.28;
  viewerState.distance = 3.25;
}

function readPhysicalPoint(prefix) {
  const point = {
    x: Number(el(`${prefix}x`).value),
    y: Number(el(`${prefix}y`).value),
    z: Number(el(`${prefix}z`).value),
  };
  if (![point.x, point.y, point.z].every(Number.isFinite)) {
    throw new Error(`点 ${prefix.toUpperCase()} 的 X/Y/Z 必须都是有效数字`);
  }
  return point;
}

async function calculateDistance() {
  const output = el("measurementResult");
  try {
    const pointA = readPhysicalPoint("a");
    const pointB = readPhysicalPoint("b");
    const response = await fetch("/api/research/measure/distance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_a: pointA, point_b: pointB }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    output.textContent = `A–B = ${Number(data.distance_mm).toFixed(3)} mm`;
  } catch (error) {
    output.textContent = `距离计算失败：${error.message}`;
  }
}

async function calculateAngle() {
  const output = el("measurementResult");
  try {
    const pointA = readPhysicalPoint("a");
    const vertexB = readPhysicalPoint("b");
    const pointC = readPhysicalPoint("c");
    const response = await fetch("/api/research/measure/angle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_a: pointA, vertex_b: vertexB, point_c: pointC }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    output.textContent = `∠ABC = ${Number(data.angle_degrees).toFixed(3)}°`;
  } catch (error) {
    output.textContent = `角度计算失败：${error.message}`;
  }
}

el("caseSelect").addEventListener("change", () => {
  const item = selectedCase();
  if (item) populateClassSelect(item);
  viewerState.indexCount = 0;
  el("canvasPlaceholder").classList.remove("hidden");
  updateMeshStats(null);
  setViewerMessage(item?.foreground_mesh_ready ? "该病例存在前景 mesh，可直接加载。" : "请选择标签并生成 Mesh。", "work");
});
el("classSelect").addEventListener("change", () => {
  viewerState.indexCount = 0;
  el("canvasPlaceholder").classList.remove("hidden");
  updateMeshStats(null);
});
el("surfaceSelect").addEventListener("change", () => {
  updateSurfaceControls();
  viewerState.indexCount = 0;
  el("canvasPlaceholder").classList.remove("hidden");
  updateMeshStats(null);
  setViewerMessage(
    selectedSurface() === "sdf"
      ? "已切换 SDF 物理平滑；默认 σ=0.4 mm，若连通域数量变化后端会拒绝导出。"
      : "已切换原始 Marching Cubes；可选择 Web 网格简化精度。",
    "work",
  );
});
el("sdfSigmaSelect").addEventListener("change", () => {
  viewerState.indexCount = 0;
  el("canvasPlaceholder").classList.remove("hidden");
  updateMeshStats(null);
  setViewerMessage("SDF σ 已切换，请重新生成对应表面。", "work");
});
el("simplifySelect").addEventListener("change", () => {
  viewerState.indexCount = 0;
  el("canvasPlaceholder").classList.remove("hidden");
  updateMeshStats(null);
  setViewerMessage("网格精度已切换，请生成或加载对应 Mesh。", "work");
});
el("buildBtn").addEventListener("click", buildMesh);
el("loadBtn").addEventListener("click", loadMesh);
el("resetViewBtn").addEventListener("click", resetView);
el("distanceBtn").addEventListener("click", calculateDistance);
el("angleBtn").addEventListener("click", calculateAngle);

try {
  updateSurfaceControls();
  initWebGL();
  loadResearchCases();
} catch (error) {
  setViewerStatus("WebGL2 不可用", "error");
  setViewerMessage(error.message, "error");
  el("canvasPlaceholder").innerHTML = `<strong>3D 初始化失败</strong><span>${error.message}</span>`;
}
