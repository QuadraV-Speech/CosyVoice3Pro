"use strict";

const SYNTHESIS_MODEL = "CosyVoice3Pro";
const REGISTRY_MODEL = "CosyVoice3ProSpeakerRegistry";

const state = {
  speakers: [],
  selectedAudioFile: null,
  audioUrl: null,
};

const elements = {
  sidebarStatusDot: document.querySelector("#sidebar-status-dot"),
  sidebarStatusText: document.querySelector("#sidebar-status-text"),
  gatewayOrb: document.querySelector("#gateway-orb"),
  gatewayStatus: document.querySelector("#gateway-status"),
  tritonOrb: document.querySelector("#triton-orb"),
  tritonStatus: document.querySelector("#triton-status"),
  modelOrb: document.querySelector("#model-orb"),
  modelStatus: document.querySelector("#model-status"),
  speakerCount: document.querySelector("#speaker-count"),
  refreshAll: document.querySelector("#refresh-all"),
  speakerSelect: document.querySelector("#speaker-select"),
  personaPreview: document.querySelector("#persona-preview p"),
  requestPrompt: document.querySelector("#request-prompt"),
  targetText: document.querySelector("#target-text"),
  textCount: document.querySelector("#text-count"),
  synthesisSpeed: document.querySelector("#synthesis-speed"),
  synthesisVolume: document.querySelector("#synthesis-volume"),
  synthesisFormat: document.querySelector("#synthesis-format"),
  synthesisMaxChars: document.querySelector("#synthesis-max-chars"),
  synthesisForm: document.querySelector("#synthesis-form"),
  generateButton: document.querySelector("#generate-button"),
  resultState: document.querySelector("#result-state"),
  emptyOutput: document.querySelector("#empty-output"),
  audioResult: document.querySelector("#audio-result"),
  audioPlayer: document.querySelector("#audio-player"),
  downloadAudio: document.querySelector("#download-audio"),
  resultDuration: document.querySelector("#result-duration"),
  resultFormat: document.querySelector("#result-format"),
  resultLatency: document.querySelector("#result-latency"),
  waveVisual: document.querySelector("#wave-visual"),
  waveBars: document.querySelector("#wave-bars"),
  speakerSearch: document.querySelector("#speaker-search"),
  speakerTableBody: document.querySelector("#speaker-table-body"),
  tableEmpty: document.querySelector("#table-empty"),
  registerDialog: document.querySelector("#register-dialog"),
  registerForm: document.querySelector("#register-form"),
  registerButton: document.querySelector("#register-button"),
  registerSpeakerId: document.querySelector("#register-speaker-id"),
  referenceAudio: document.querySelector("#reference-audio"),
  referenceAudioUrl: document.querySelector("#reference-audio-url"),
  audioUploadField: document.querySelector("#audio-upload-field"),
  audioUrlField: document.querySelector("#audio-url-field"),
  referenceText: document.querySelector("#reference-text"),
  defaultPrompt: document.querySelector("#default-prompt"),
  dropZone: document.querySelector("#drop-zone"),
  uploadTitle: document.querySelector("#upload-title"),
  uploadMeta: document.querySelector("#upload-meta"),
  toastRegion: document.querySelector("#toast-region"),
};

function tritonInput(name, shape, datatype, data) {
  return { name, shape, datatype, data };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { error: text };
    }
  }
  if (!response.ok) {
    throw new Error(
      payload.detail || payload.error || `HTTP ${response.status}`,
    );
  }
  return payload;
}

async function infer(modelName, inputs) {
  return requestJson(`/v2/models/${modelName}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inputs }),
  });
}

function outputsByName(payload) {
  return Object.fromEntries(
    (payload.outputs || []).map((item) => [item.name, item.data || []]),
  );
}

function registryMessage(payload) {
  const outputs = outputsByName(payload);
  const value = (outputs.message || [])[0] || "{}";
  try {
    return JSON.parse(value);
  } catch {
    return { message: value };
  }
}

function setStatus(orb, text, label, ready) {
  orb.classList.toggle("is-ready", ready === true);
  orb.classList.toggle("is-error", ready === false);
  text.textContent = label;
}

function toast(title, message, type = "success") {
  const node = document.createElement("div");
  node.className = `toast${type === "error" ? " is-error" : ""}`;
  node.innerHTML = `
    <span class="toast-marker"></span>
    <div>
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(message)}</p>
    </div>
    <button type="button" aria-label="关闭">×</button>
  `;
  node.querySelector("button").addEventListener("click", () => node.remove());
  elements.toastRegion.append(node);
  window.setTimeout(() => node.remove(), 5200);
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(unixSeconds) {
  if (!unixSeconds) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(unixSeconds * 1000));
}

function setButtonLoading(button, loading, loadingText, idleText) {
  button.disabled = loading;
  button.classList.toggle("is-loading", loading);
  const label = button.querySelector(".button-label");
  if (label) {
    label.textContent = loading ? loadingText : idleText;
  } else {
    button.textContent = loading ? loadingText : idleText;
  }
}

async function refreshHealth() {
  let gatewayReady = false;
  let tritonReady = false;
  let modelReady = false;

  try {
    const info = await requestJson("/admin/api/info");
    gatewayReady = true;
    tritonReady = Boolean(info.triton_ready);
  } catch {
    gatewayReady = false;
  }

  if (gatewayReady) {
    try {
      const response = await fetch(`/v2/models/${SYNTHESIS_MODEL}/ready`);
      modelReady = response.ok;
    } catch {
      modelReady = false;
    }
  }

  setStatus(
    elements.gatewayOrb,
    elements.gatewayStatus,
    gatewayReady ? "运行正常" : "连接失败",
    gatewayReady,
  );
  setStatus(
    elements.tritonOrb,
    elements.tritonStatus,
    tritonReady ? "Ready" : "Not Ready",
    tritonReady,
  );
  setStatus(
    elements.modelOrb,
    elements.modelStatus,
    modelReady ? "模型就绪" : "模型不可用",
    modelReady,
  );
  elements.sidebarStatusDot.classList.toggle(
    "is-ready",
    gatewayReady && tritonReady && modelReady,
  );
  elements.sidebarStatusDot.classList.toggle(
    "is-error",
    !gatewayReady || !tritonReady || !modelReady,
  );
  elements.sidebarStatusText.textContent =
    gatewayReady && tritonReady && modelReady ? "全部就绪" : "服务异常";
}

async function refreshSpeakers() {
  const payload = await infer(REGISTRY_MODEL, [
    tritonInput("operation", [1, 1], "BYTES", ["list"]),
  ]);
  const message = registryMessage(payload);
  state.speakers = Array.isArray(message.speakers) ? message.speakers : [];
  elements.speakerCount.textContent = String(state.speakers.length);
  renderSpeakerOptions();
  renderSpeakerTable();
}

function renderSpeakerOptions() {
  const current = elements.speakerSelect.value;
  elements.speakerSelect.innerHTML =
    '<option value="">请先注册或选择声纹</option>';
  for (const speaker of state.speakers) {
    const option = document.createElement("option");
    option.value = speaker.speaker_id;
    option.textContent = speaker.speaker_id;
    elements.speakerSelect.append(option);
  }
  if (state.speakers.some((speaker) => speaker.speaker_id === current)) {
    elements.speakerSelect.value = current;
  } else if (state.speakers.length === 1) {
    elements.speakerSelect.value = state.speakers[0].speaker_id;
  }
  updatePersonaPreview();
}

function renderSpeakerTable() {
  const query = elements.speakerSearch.value.trim().toLowerCase();
  const speakers = state.speakers.filter((speaker) =>
    speaker.speaker_id.toLowerCase().includes(query),
  );
  elements.speakerTableBody.replaceChildren();
  elements.tableEmpty.classList.toggle("is-visible", speakers.length === 0);

  for (const speaker of speakers) {
    const row = document.createElement("tr");
    const prompt = speaker.prompt || "默认中性画像";
    const duration =
      typeof speaker.duration_seconds === "number"
        ? `${speaker.duration_seconds.toFixed(2)} 秒`
        : "—";
    row.innerHTML = `
      <td>
        <div class="speaker-id-cell">
          <span class="speaker-avatar">${escapeHtml(speaker.speaker_id.slice(0, 1).toUpperCase())}</span>
          <span>
            <strong>${escapeHtml(speaker.speaker_id)}</strong>
            <small>TTS ready</small>
          </span>
        </div>
      </td>
      <td class="prompt-cell" title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</td>
      <td>${escapeHtml(duration)}</td>
      <td><span class="version-code">${escapeHtml(speaker.speaker_version || "—")}</span></td>
      <td>${escapeHtml(formatDate(speaker.registered_at))}</td>
      <td>
        <div class="row-actions">
          <button class="row-button use-speaker" type="button">使用</button>
          <button class="row-button is-danger delete-speaker" type="button">删除</button>
        </div>
      </td>
    `;
    row.querySelector(".use-speaker").addEventListener("click", () => {
      elements.speakerSelect.value = speaker.speaker_id;
      updatePersonaPreview();
      document.querySelector("#studio").scrollIntoView({ behavior: "smooth" });
    });
    row.querySelector(".delete-speaker").addEventListener("click", () =>
      deleteSpeaker(speaker.speaker_id),
    );
    elements.speakerTableBody.append(row);
  }
}

function updatePersonaPreview() {
  const speaker = state.speakers.find(
    (item) => item.speaker_id === elements.speakerSelect.value,
  );
  if (!speaker) {
    elements.personaPreview.textContent = "选择声纹后显示";
    return;
  }
  elements.personaPreview.textContent =
    speaker.prompt || "未设置自定义画像，将使用模型默认指令。";
}

async function deleteSpeaker(speakerId) {
  const confirmed = window.confirm(
    `确定删除声纹“${speakerId}”吗？此操作会删除注册特征。`,
  );
  if (!confirmed) return;
  try {
    await infer(REGISTRY_MODEL, [
      tritonInput("operation", [1, 1], "BYTES", ["delete"]),
      tritonInput("speaker_id", [1, 1], "BYTES", [speakerId]),
    ]);
    toast("声纹已删除", speakerId);
    await refreshSpeakers();
  } catch (error) {
    toast("删除失败", error.message, "error");
  }
}

function buildWaveBars() {
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < 58; index += 1) {
    const bar = document.createElement("i");
    const height =
      12 + Math.abs(Math.sin(index * 0.63) * 48) + ((index * 17) % 20);
    bar.style.height = `${height}px`;
    bar.style.animationDelay = `${(index % 13) * 35}ms`;
    fragment.append(bar);
  }
  elements.waveBars.append(fragment);
}

async function synthesize(event) {
  event.preventDefault();
  const speakerId = elements.speakerSelect.value;
  const targetText = elements.targetText.value.trim();
  const prompt = elements.requestPrompt.value.trim();
  const speed = elements.synthesisSpeed.value;
  const volume = elements.synthesisVolume.value;
  const outputFormat = elements.synthesisFormat.value;
  const maxChars = elements.synthesisMaxChars.value;
  if (!speakerId || !targetText) return;

  setButtonLoading(elements.generateButton, true, "正在生成…", "生成语音");
  elements.resultState.textContent = "推理中";
  elements.resultState.classList.remove("is-ready");
  const startedAt = performance.now();

  try {
    const formData = new FormData();
    formData.append("text", targetText);
    formData.append("speakerId", speakerId);
    formData.append("prompt", prompt);
    formData.append("speed", speed);
    formData.append("volume", volume);
    formData.append("output_format", outputFormat);
    formData.append("max_chars", maxChars);

    const response = await fetch("/tts/", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const responseText = await response.text();
      let detail = responseText || `HTTP ${response.status}`;
      try {
        const payload = JSON.parse(responseText);
        detail = payload.detail || payload.error || detail;
      } catch {
        // Keep the plain-text response.
      }
      throw new Error(detail);
    }

    const audioBlob = await response.blob();
    if (!audioBlob.size) {
      throw new Error("服务返回了空音频");
    }
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(audioBlob);
    elements.audioPlayer.onerror = () => {
      elements.resultDuration.textContent = "不可试听";
    };
    elements.audioPlayer.onloadedmetadata = () => {
      const duration = elements.audioPlayer.duration;
      elements.resultDuration.textContent = Number.isFinite(duration)
        ? `${duration.toFixed(2)} s`
        : "—";
    };
    elements.audioPlayer.src = state.audioUrl;
    elements.downloadAudio.href = state.audioUrl;
    elements.downloadAudio.download =
      `${speakerId}-${Date.now()}.${outputFormat}`;
    elements.downloadAudio.textContent = `下载 ${outputFormat.toUpperCase()}`;
    elements.resultDuration.textContent = "读取中";
    elements.resultFormat.textContent =
      `${outputFormat.toUpperCase()} · 16 kHz`;
    elements.resultLatency.textContent = `${((performance.now() - startedAt) / 1000).toFixed(2)} s`;
    elements.emptyOutput.classList.add("is-hidden");
    elements.audioResult.classList.remove("is-hidden");
    elements.resultState.textContent = "生成完成";
    elements.resultState.classList.add("is-ready");
    toast(
      "语音生成完成",
      `${speakerId} · ${outputFormat.toUpperCase()} · ${speed} · ${volume}`,
    );
  } catch (error) {
    elements.resultState.textContent = "生成失败";
    toast("语音生成失败", error.message, "error");
  } finally {
    setButtonLoading(elements.generateButton, false, "正在生成…", "生成语音");
  }
}

function setSelectedFile(file) {
  state.selectedAudioFile = file || null;
  if (!file) {
    elements.uploadTitle.textContent = "点击选择或拖入音频";
    elements.uploadMeta.textContent =
      "支持 WAV、MP3、M4A；建议 3～10 秒清晰单人声";
    return;
  }
  elements.uploadTitle.textContent = file.name;
  elements.uploadMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 注册时自动转为 16kHz 单声道`;
}

function registerAudioSource() {
  const selected = document.querySelector(
    'input[name="register-audio-source"]:checked',
  );
  return selected ? selected.value : "upload";
}

function updateRegisterAudioSource() {
  const source = registerAudioSource();
  const useUrl = source === "url";
  elements.audioUploadField.classList.toggle("is-hidden", useUrl);
  elements.audioUrlField.classList.toggle("is-hidden", !useUrl);
  if (useUrl) {
    window.setTimeout(() => elements.referenceAudioUrl.focus(), 40);
  }
}

async function registerSpeaker(event) {
  event.preventDefault();
  const speakerId = elements.registerSpeakerId.value.trim();
  const referenceText = elements.referenceText.value.trim();
  const prompt = elements.defaultPrompt.value.trim();
  const source = registerAudioSource();
  const file = state.selectedAudioFile || elements.referenceAudio.files[0];
  const audioUrl = elements.referenceAudioUrl.value.trim();
  if (!speakerId || !referenceText) return;
  if (source === "upload" && !file) {
    toast("请选择提示音频", "支持 WAV、MP3、M4A 等浏览器可解码格式。", "error");
    return;
  }
  if (source === "url" && !audioUrl) {
    toast("请输入音频 URL", "URL 必须可以由服务端公开访问。", "error");
    return;
  }

  setButtonLoading(
    elements.registerButton,
    true,
    "正在处理并提取特征…",
    "提取特征并注册",
  );

  try {
    const formData = new FormData();
    formData.append("speakerId", speakerId);
    formData.append("reference_text", referenceText);
    formData.append("prompt", prompt);
    if (source === "url") {
      formData.append("audio_url", audioUrl);
    } else {
      formData.append("audio", file);
    }

    const payload = await requestJson("/register", {
      method: "POST",
      body: formData,
    });
    const duration = payload.metadata
      ? payload.metadata.duration_seconds
      : null;
    toast(
      "声纹注册成功",
      `${speakerId} · ${duration || "—"} 秒 · ${source === "url" ? "URL" : "上传"}`,
    );
    elements.registerDialog.close();
    elements.registerForm.reset();
    setSelectedFile(null);
    updateRegisterAudioSource();
    await refreshSpeakers();
    elements.speakerSelect.value = speakerId;
    updatePersonaPreview();
  } catch (error) {
    toast("声纹注册失败", error.message, "error");
  } finally {
    setButtonLoading(
      elements.registerButton,
      false,
      "正在处理并提取特征…",
      "提取特征并注册",
    );
  }
}

function openRegisterDialog() {
  elements.registerDialog.showModal();
  window.setTimeout(() => elements.registerSpeakerId.focus(), 80);
}

async function refreshAll() {
  elements.refreshAll.disabled = true;
  try {
    await Promise.all([refreshHealth(), refreshSpeakers()]);
  } catch (error) {
    toast("刷新失败", error.message, "error");
  } finally {
    elements.refreshAll.disabled = false;
  }
}

function bindEvents() {
  document.querySelectorAll("[data-scroll]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) =>
        item.classList.toggle("is-active", item === button),
      );
      document
        .querySelector(`#${button.dataset.scroll}`)
        .scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  document.querySelectorAll("#open-register, #open-register-secondary").forEach(
    (button) => button.addEventListener("click", openRegisterDialog),
  );
  document.querySelector("#close-register").addEventListener("click", () =>
    elements.registerDialog.close(),
  );
  document.querySelector("#cancel-register").addEventListener("click", () =>
    elements.registerDialog.close(),
  );

  elements.refreshAll.addEventListener("click", refreshAll);
  elements.speakerSelect.addEventListener("change", updatePersonaPreview);
  elements.speakerSearch.addEventListener("input", renderSpeakerTable);
  elements.synthesisForm.addEventListener("submit", synthesize);
  elements.registerForm.addEventListener("submit", registerSpeaker);
  document
    .querySelectorAll('input[name="register-audio-source"]')
    .forEach((input) =>
      input.addEventListener("change", updateRegisterAudioSource),
    );
  elements.targetText.addEventListener("input", () => {
    elements.textCount.textContent = String(elements.targetText.value.length);
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.requestPrompt.value = button.dataset.prompt;
      elements.requestPrompt.focus();
    });
  });

  elements.referenceAudio.addEventListener("change", () =>
    setSelectedFile(elements.referenceAudio.files[0]),
  );
  for (const eventName of ["dragenter", "dragover"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    });
  }
  elements.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
      document.querySelector(
        'input[name="register-audio-source"][value="upload"]',
      ).checked = true;
      updateRegisterAudioSource();
      setSelectedFile(file);
    }
  });

  elements.audioPlayer.addEventListener("play", () =>
    elements.waveVisual.classList.add("is-playing"),
  );
  for (const eventName of ["pause", "ended"]) {
    elements.audioPlayer.addEventListener(eventName, () =>
      elements.waveVisual.classList.remove("is-playing"),
    );
  }
}

async function initialize() {
  buildWaveBars();
  bindEvents();
  updateRegisterAudioSource();
  await refreshAll();
  window.setInterval(refreshHealth, 30000);
}

initialize().catch((error) => {
  toast("初始化失败", error.message, "error");
});
