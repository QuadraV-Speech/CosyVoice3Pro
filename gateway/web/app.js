"use strict";

const state = {
  speakers: [],
  selectedAudioFile: null,
  audioUrl: null,
  streamSession: null,
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
  streamButton: document.querySelector("#stream-button"),
  resultState: document.querySelector("#result-state"),
  emptyOutput: document.querySelector("#empty-output"),
  audioResult: document.querySelector("#audio-result"),
  audioPlayer: document.querySelector("#audio-player"),
  downloadAudio: document.querySelector("#download-audio"),
  resultDuration: document.querySelector("#result-duration"),
  resultFormat: document.querySelector("#result-format"),
  resultLatency: document.querySelector("#result-latency"),
  outputSampleRate: document.querySelector("#output-sample-rate"),
  outputTransport: document.querySelector("#output-transport"),
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
    const response = await fetch("/health");
    const info = await response.json();
    gatewayReady = true;
    tritonReady = Boolean(info.tritonReady);
    modelReady = Boolean(
      info.models
      && info.models.ttsReady
      && info.models.streamingTtsReady
      && info.models.speakerRegistryReady,
    );
  } catch {
    gatewayReady = false;
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
  const payload = await requestJson("/speakers");
  state.speakers = Array.isArray(payload.speakers) ? payload.speakers : [];
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
    option.value = speaker.speakerId;
    option.textContent = speaker.speakerId;
    elements.speakerSelect.append(option);
  }
  if (state.speakers.some((speaker) => speaker.speakerId === current)) {
    elements.speakerSelect.value = current;
  } else if (state.speakers.length === 1) {
    elements.speakerSelect.value = state.speakers[0].speakerId;
  }
  updatePersonaPreview();
}

function renderSpeakerTable() {
  const query = elements.speakerSearch.value.trim().toLowerCase();
  const speakers = state.speakers.filter((speaker) =>
    speaker.speakerId.toLowerCase().includes(query),
  );
  elements.speakerTableBody.replaceChildren();
  elements.tableEmpty.classList.toggle("is-visible", speakers.length === 0);

  for (const speaker of speakers) {
    const row = document.createElement("tr");
    const prompt = speaker.prompt || "默认中性画像";
    const duration =
      typeof speaker.durationSeconds === "number"
        ? `${speaker.durationSeconds.toFixed(2)} 秒`
        : "—";
    row.innerHTML = `
      <td>
        <div class="speaker-id-cell">
          <span class="speaker-avatar">${escapeHtml(speaker.speakerId.slice(0, 1).toUpperCase())}</span>
          <span>
            <strong>${escapeHtml(speaker.speakerId)}</strong>
            <small>TTS ready</small>
          </span>
        </div>
      </td>
      <td class="prompt-cell" title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</td>
      <td>${escapeHtml(duration)}</td>
      <td><span class="version-code">${escapeHtml(speaker.speakerVersion || "—")}</span></td>
      <td>${escapeHtml(formatDate(speaker.registeredAt))}</td>
      <td>
        <div class="row-actions">
          <button class="row-button use-speaker" type="button">使用</button>
          <button class="row-button is-danger delete-speaker" type="button">删除</button>
        </div>
      </td>
    `;
    row.querySelector(".use-speaker").addEventListener("click", () => {
      elements.speakerSelect.value = speaker.speakerId;
      updatePersonaPreview();
      document.querySelector("#studio").scrollIntoView({ behavior: "smooth" });
    });
    row.querySelector(".delete-speaker").addEventListener("click", () =>
      deleteSpeaker(speaker.speakerId),
    );
    elements.speakerTableBody.append(row);
  }
}

function updatePersonaPreview() {
  const speaker = state.speakers.find(
    (item) => item.speakerId === elements.speakerSelect.value,
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
    await requestJson(`/speakers/${encodeURIComponent(speakerId)}`, {
      method: "DELETE",
    });
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

function decodeBase64Pcm(encoded) {
  const binary = window.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("服务返回了不完整的 PCM 采样");
  }
  return bytes;
}

function createWavBlob(pcmChunks, sampleRate) {
  const dataLength = pcmChunks.reduce(
    (total, chunk) => total + chunk.byteLength,
    0,
  );
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const writeText = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataLength, true);
  return new Blob([header, ...pcmChunks], { type: "audio/wav" });
}

async function consumeSse(body, onEvent) {
  if (!body) throw new Error("浏览器未收到可读取的 SSE 响应体");
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = async (block) => {
    if (!block || block.startsWith(":")) return;
    let eventName = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) return;
    let payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      throw new Error(`无法解析 SSE ${eventName} 事件`);
    }
    await onEvent(eventName, payload);
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary).replaceAll("\r", "");
        buffer = buffer.slice(boundary + 2);
        await dispatch(block);
        boundary = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) await dispatch(buffer.trim());
  } finally {
    reader.releaseLock();
  }
}

function schedulePcm(session, pcmBytes) {
  const sampleCount = pcmBytes.byteLength / 2;
  const audioBuffer = session.audioContext.createBuffer(
    1,
    sampleCount,
    session.sampleRate,
  );
  const channel = audioBuffer.getChannelData(0);
  const pcmView = new DataView(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    pcmBytes.byteLength,
  );
  for (let index = 0; index < sampleCount; index += 1) {
    channel[index] = pcmView.getInt16(index * 2, true) / 32768;
  }

  const source = session.audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(session.audioContext.destination);
  const startAt = Math.max(
    session.nextPlaybackAt,
    session.audioContext.currentTime + 0.055,
  );
  source.start(startAt);
  session.nextPlaybackAt = startAt + audioBuffer.duration;
  session.sources.add(source);
  source.addEventListener("ended", () => session.sources.delete(source), {
    once: true,
  });
}

function setStreamingButton(active) {
  elements.streamButton.classList.toggle("is-streaming", active);
  elements.streamButton.querySelector(".button-label").textContent = active
    ? "停止在线播报"
    : "流式在线播报";
}

function cancelStreaming(showStatus = true) {
  const session = state.streamSession;
  if (!session || session.cancelled) return;
  session.cancelled = true;
  session.controller.abort();
  for (const source of session.sources) {
    try {
      source.stop();
    } catch {
      // The source may already have ended.
    }
  }
  session.sources.clear();
  session.audioContext.close().catch(() => {});
  elements.waveVisual.classList.remove("is-playing");
  if (showStatus) {
    elements.resultState.textContent = "播报已停止";
    toast("在线播报已停止", "已取消服务端推理和浏览器播放队列。", "error");
  }
}

async function streamSynthesize() {
  if (state.streamSession) {
    cancelStreaming(true);
    return;
  }
  if (!elements.synthesisForm.reportValidity()) return;

  const speakerId = elements.speakerSelect.value;
  const targetText = elements.targetText.value.trim();
  const prompt = elements.requestPrompt.value.trim();
  if (!speakerId || !targetText) {
    toast("缺少合成参数", "请选择声纹并输入合成文本。", "error");
    return;
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    toast("浏览器不支持在线播报", "当前浏览器缺少 Web Audio API。", "error");
    return;
  }

  const session = {
    controller: new AbortController(),
    audioContext: new AudioContextClass(),
    sources: new Set(),
    chunks: [],
    sampleRate: 16000,
    nextPlaybackAt: 0,
    cancelled: false,
    done: null,
    firstAudioAt: null,
  };
  state.streamSession = session;
  try {
    await session.audioContext.resume();
  } catch (error) {
    state.streamSession = null;
    await session.audioContext.close().catch(() => {});
    toast("无法启动音频设备", error.message, "error");
    return;
  }
  session.nextPlaybackAt = session.audioContext.currentTime;
  setStreamingButton(true);
  elements.generateButton.disabled = true;
  elements.resultState.textContent = "连接流式服务";
  elements.resultState.classList.remove("is-ready");
  elements.audioPlayer.pause();
  elements.outputSampleRate.textContent = "16 kHz";
  elements.outputTransport.textContent = "SSE · PCM";
  const startedAt = performance.now();

  try {
    const formData = new FormData();
    formData.append("text", targetText);
    formData.append("speakerId", speakerId);
    formData.append("prompt", prompt);
    formData.append("speed", elements.synthesisSpeed.value);
    formData.append("volume", elements.synthesisVolume.value);

    const response = await fetch("/tts/stream", {
      method: "POST",
      body: formData,
      signal: session.controller.signal,
      headers: { Accept: "text/event-stream" },
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

    await consumeSse(response.body, async (eventName, payload) => {
      if (eventName === "meta") {
        session.sampleRate = Number(payload.sampleRate) || 16000;
        elements.outputSampleRate.textContent =
          `${session.sampleRate / 1000} kHz`;
        elements.resultState.textContent = "等待首段音频";
        return;
      }
      if (eventName === "audio") {
        const pcmBytes = decodeBase64Pcm(payload.audio || "");
        if (!pcmBytes.byteLength) return;
        if (session.firstAudioAt == null) {
          session.firstAudioAt = performance.now();
          elements.emptyOutput.classList.add("is-hidden");
          elements.resultState.textContent = "在线播报中";
          elements.waveVisual.classList.add("is-playing");
        }
        session.chunks.push(pcmBytes);
        schedulePcm(session, pcmBytes);
        return;
      }
      if (eventName === "done") {
        session.done = payload;
        return;
      }
      if (eventName === "error") {
        throw new Error(payload.detail || "流式推理失败");
      }
    });

    if (!session.done || !session.chunks.length) {
      throw new Error("SSE 连接结束，但没有收到完整音频");
    }

    const wavBlob = createWavBlob(session.chunks, session.sampleRate);
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(wavBlob);
    elements.audioPlayer.src = state.audioUrl;
    elements.downloadAudio.href = state.audioUrl;
    elements.downloadAudio.download = `${speakerId}-${Date.now()}-stream.wav`;
    elements.downloadAudio.textContent = "下载流式 WAV";
    elements.resultDuration.textContent =
      `${Number(session.done.durationSeconds).toFixed(2)} s`;
    elements.resultFormat.textContent = "WAV · SSE PCM · 16 kHz";
    const localFirstAudio = session.firstAudioAt - startedAt;
    const firstAudioMs = Number(session.done.firstAudioMs) || localFirstAudio;
    const totalMs = Number(session.done.totalMs) ||
      (performance.now() - startedAt);
    elements.resultLatency.textContent =
      `首包 ${(firstAudioMs / 1000).toFixed(2)}s · 总 ${(totalMs / 1000).toFixed(2)}s`;
    elements.audioResult.classList.remove("is-hidden");

    while (
      !session.cancelled
      && session.audioContext.currentTime + 0.03 < session.nextPlaybackAt
    ) {
      await new Promise((resolve) => window.setTimeout(resolve, 100));
    }
    if (!session.cancelled) {
      elements.resultState.textContent = "播报完成";
      elements.resultState.classList.add("is-ready");
      elements.waveVisual.classList.remove("is-playing");
      toast(
        "在线播报完成",
        `${speakerId} · 首包 ${(firstAudioMs / 1000).toFixed(2)} 秒`,
      );
    }
  } catch (error) {
    if (error.name !== "AbortError" && !session.cancelled) {
      elements.resultState.textContent = "播报失败";
      elements.waveVisual.classList.remove("is-playing");
      toast("在线播报失败", error.message, "error");
    }
  } finally {
    if (!session.cancelled) {
      await session.audioContext.close().catch(() => {});
    }
    if (state.streamSession === session) state.streamSession = null;
    setStreamingButton(false);
    elements.generateButton.disabled = false;
  }
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
  if (!speakerId || !targetText || state.streamSession) return;

  setButtonLoading(
    elements.generateButton,
    true,
    "正在生成…",
    "生成完整音频",
  );
  elements.streamButton.disabled = true;
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
    elements.outputSampleRate.textContent = "16 kHz";
    elements.outputTransport.textContent = "Server encoded";
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
    setButtonLoading(
      elements.generateButton,
      false,
      "正在生成…",
      "生成完整音频",
    );
    elements.streamButton.disabled = false;
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
    const duration = payload.speaker
      ? payload.speaker.durationSeconds
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
  elements.streamButton.addEventListener("click", streamSynthesize);
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
