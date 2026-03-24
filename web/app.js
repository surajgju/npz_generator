import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const statusEl = document.getElementById("status");
const bufferSecEl = document.getElementById("bufferSec");
const queueLenEl = document.getElementById("queueLen");
const inFpsEl = document.getElementById("inFps");
const outFpsEl = document.getElementById("outFps");
const playFpsEl = document.getElementById("playFps");
const streamFpsEl = document.getElementById("streamFps");
const pipelineModeEl = document.getElementById("pipelineMode");
const audioStatusEl = document.getElementById("audioStatus");
const audioBufferEl = document.getElementById("audioBuffer");
const playStateEl = document.getElementById("playState");
const lodLevelEl = document.getElementById("lodLevel");
const bufferFillEl = document.getElementById("bufferFill");
const enableAudioBtn = document.getElementById("enableAudio");
const togglePlayBtn = document.getElementById("togglePlay");
const clearBufferBtn = document.getElementById("clearBuffer");
const resetCamBtn = document.getElementById("resetCam");
const canvas = document.getElementById("canvas");
const log = (...args) => console.log("[Viewer]", ...args);
const warn = (...args) => console.warn("[Viewer]", ...args);
const error = (...args) => console.error("[Viewer]", ...args);
const DEBUG = true;
const APP_VERSION = "audio9";
let lastDebugSummary = 0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const AUTO_CENTER = true;
const AUTO_SCALE = true;
const TARGET_HEIGHT = 1.7;

const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.01, 50);
camera.position.set(1.2, 1.4, 2.5);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.0, 0);
controls.update();

const ambient = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(2, 3, 2);
scene.add(dir);

let mesh = null;
let geometry = null;
let positions = null;
let faces = null;
let lodFaces = null;
let currentLod = 0;
let frameCount = 0;
let cameraFitted = false;

const NORMAL_EVERY = 15;
let streamFps = 20;
const maxBufferSeconds = 10;

let currentFrameIndex = 0;
let manualPaused = false;
let playCount = 0;
let lastRateTime = performance.now();
let currentPlayFps = streamFps;

let audioCtx = null;
let audioWs = null;
let audioEnabled = false;
let audioStarted = false;
let audioStartTime = null;
let audioScheduledTime = 0;
let audioQueue = [];
let audioQueuedSec = 0;
const AUDIO_JITTER_SEC = 0.05;
const AUDIO_LOW_BUFFER_SEC = 0.2;
const workerEnabled = typeof Worker !== "undefined";
let workerAlive = false;
let worker = null;
let workerFrame = null;
let workerBounds = null;
let workerFrameIndex = -1;
let workerQueueLen = 0;
let workerDropped = 0;
let workerInFps = 0;
let workerPlaybackFps = 0;
let frameDirty = false;
let workerReady = false;
let workerFallbackTimer = null;
let lastFrameUpdate = 0;
let workerMaxIndex = -1;
let workerMinIndex = -1;
let driftStart = null;
const DRIFT_THRESHOLD_FRAMES = 20;
const RESYNC_GRACE_MS = 1500;

async function loadFaces() {
  try {
    const res = await fetch("./faces.json");
    faces = await res.json();
    lodFaces = [
      faces,
      faces.filter((_, idx) => idx % 2 === 0),
      faces.filter((_, idx) => idx % 4 === 0),
    ];
    log("Loaded faces:", faces.length);
  } catch (err) {
    error("Failed to load faces.json", err);
  }
}

function initMesh(nverts) {
  if (!lodFaces) {
    warn("initMesh called before faces loaded");
    return;
  }
  geometry = new THREE.BufferGeometry();
  const index = new Uint32Array(lodFaces[0].flat());
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  positions = new Float32Array(nverts * 3);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({
    color: 0xd3d3d3,
    side: THREE.DoubleSide,
  });
  mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  scene.add(mesh);
  log("Mesh initialized:", { nverts, faces: lodFaces[0].length });
}

function applyLod(level) {
  if (!geometry || !lodFaces) return;
  if (level === currentLod) return;
  const index = new Uint32Array(lodFaces[level].flat());
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  currentLod = level;
  lodLevelEl.textContent = `LOD${level}`;
  log(`LOD switched to LOD${level}`);
}

function updateVertices(floatArray, bounds) {
  if (!geometry || !positions) return;
  if (floatArray.length !== positions.length) {
    warn("Vertex size mismatch:", floatArray.length, "expected", positions.length);
    return;
  }
  positions.set(floatArray);
  geometry.attributes.position.needsUpdate = true;
  if (frameCount % NORMAL_EVERY === 0) {
    geometry.computeVertexNormals();
  }
  const shouldRecalcBounds = frameCount === 0 || frameCount % 30 === 0;
  if (mesh && (AUTO_CENTER || AUTO_SCALE) && (bounds || shouldRecalcBounds)) {
    let minX = bounds ? bounds.min[0] : Infinity;
    let minY = bounds ? bounds.min[1] : Infinity;
    let minZ = bounds ? bounds.min[2] : Infinity;
    let maxX = bounds ? bounds.max[0] : -Infinity;
    let maxY = bounds ? bounds.max[1] : -Infinity;
    let maxZ = bounds ? bounds.max[2] : -Infinity;
    if (!bounds) {
      for (let i = 0; i < floatArray.length; i += 3) {
        const x = floatArray[i];
        const y = floatArray[i + 1];
        const z = floatArray[i + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
          warn("Non-finite vertex detected, skipping frame");
          return;
        }
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }
    }
    if (AUTO_CENTER) {
      const cx = (minX + maxX) * 0.5;
      const cy = (minY + maxY) * 0.5;
      const cz = (minZ + maxZ) * 0.5;
      mesh.position.set(-cx, -cy, -cz);
      controls.target.set(0, 0, 0);
    }
    if (AUTO_SCALE) {
      const height = Math.max(1e-6, maxY - minY);
      let scale = TARGET_HEIGHT / height;
      scale = Math.min(20, Math.max(0.05, scale));
      mesh.scale.setScalar(scale);
    }
    if (!cameraFitted) {
      const dx = maxX - minX;
      const dy = maxY - minY;
      const dz = maxZ - minZ;
      const size = Math.max(dx, dy, dz) * (mesh.scale.x || 1.0);
      const dist = Math.max(1.0, size * 2.2);
      camera.position.set(0, 0, dist);
      camera.near = Math.max(0.01, dist / 100);
      camera.far = dist * 10;
      camera.updateProjectionMatrix();
      controls.update();
      cameraFitted = true;
      log("Camera fitted:", { size, dist });
    }
  }
  frameCount += 1;
}

function animate() {
  requestAnimationFrame(animate);
  updatePlaybackFrame();
  controls.update();
  renderer.render(scene, camera);
}

function initWorker() {
  if (!workerEnabled) return;
  pipelineModeEl.textContent = "Worker";
  worker = new Worker(`./stream_worker.js?v=${APP_VERSION}`);
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "frame") {
      streamFps = msg.streamFps || streamFps;
      streamFpsEl.textContent = `${streamFps}`;
      workerPlaybackFps = msg.playbackFps || workerPlaybackFps;
      workerQueueLen = msg.queueLen ?? workerQueueLen;
      workerDropped = msg.dropped ?? workerDropped;
      workerFrameIndex = msg.frameIndex ?? workerFrameIndex;
      workerBounds = msg.bounds || null;
      if (!mesh) {
        const nverts = msg.nverts || Math.floor((msg.buffer?.byteLength || 0) / 12);
        if (nverts > 0) {
          initMesh(nverts);
        }
      }
      workerFrame = new Float32Array(msg.buffer);
      frameDirty = true;
    } else if (msg.type === "status") {
      if (msg.status === "connected") {
        statusEl.textContent = "Connected";
        workerReady = true;
        workerAlive = true;
        if (workerFallbackTimer) {
          clearTimeout(workerFallbackTimer);
          workerFallbackTimer = null;
        }
      } else if (msg.status === "error" || msg.status === "closed") {
        warn("Worker status:", msg);
        pipelineModeEl.textContent = "Error";
        statusEl.textContent = "Disconnected";
        workerAlive = false;
        if (audioCtx) {
          audioCtx.suspend();
        }
      }
    } else if (msg.type === "stats") {
      workerInFps = msg.inFps ?? workerInFps;
      workerQueueLen = msg.queueLen ?? workerQueueLen;
      workerDropped = msg.dropped ?? workerDropped;
      streamFps = msg.streamFps || streamFps;
      workerPlaybackFps = msg.playbackFps || workerPlaybackFps;
      workerMaxIndex = Number.isFinite(msg.maxIndex) ? msg.maxIndex : workerMaxIndex;
      workerMinIndex = Number.isFinite(msg.minIndexEstimate) ? msg.minIndexEstimate : workerMinIndex;
      if (DEBUG) {
        log("Worker stats", {
          inFps: workerInFps,
          queueLen: workerQueueLen,
          dropped: workerDropped,
          playbackFps: workerPlaybackFps,
          baseFrame: msg.baseFrame,
          maxIndex: msg.maxIndex,
          minIndex: msg.minIndexEstimate,
          streamFps,
        });
      }
    }
  };
  worker.onerror = () => {
    warn("Worker failed to start");
    pipelineModeEl.textContent = "Error";
    statusEl.textContent = "Disconnected";
    workerAlive = false;
  };
  statusEl.textContent = "Connecting...";
  worker.postMessage({ type: "init", host: window.location.host });
  workerFallbackTimer = setTimeout(() => {
    if (!workerReady) {
      warn("Worker not ready");
      pipelineModeEl.textContent = "Error";
      statusEl.textContent = "Disconnected";
      workerAlive = false;
    }
  }, 2000);
}

function connectAudioSocket() {
  if (audioWs) return;
  audioWs = new WebSocket(`ws://${window.location.host}/ws/audio_out`);
  audioWs.binaryType = "arraybuffer";
  let audioHeader = null;

  audioWs.onopen = () => {
    audioStatusEl.textContent = "connected";
    log("Connected to /ws/audio_out");
  };
  audioWs.onclose = () => {
    audioStatusEl.textContent = "disconnected";
    warn("Audio WS disconnected");
    audioWs = null;
  };
  audioWs.onerror = () => {
    audioStatusEl.textContent = "error";
    error("Audio WS error");
  };
  audioWs.onmessage = (event) => {
    if (typeof event.data === "string") {
      try {
        audioHeader = JSON.parse(event.data);
      } catch {
        audioHeader = null;
      }
      return;
    }
    if (!audioHeader || !audioCtx) return;
    if (audioHeader.type !== "audio") {
      audioHeader = null;
      return;
    }
    const sr = audioHeader.sr || 16000;
    const pcm = new Int16Array(event.data);
    const floats = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      floats[i] = pcm[i] / 32768;
    }
    const buffer = audioCtx.createBuffer(1, floats.length, sr);
    buffer.copyToChannel(floats, 0);
    const duration = buffer.duration;
    if (!audioStarted) {
      audioQueue.push({ buffer, duration });
      audioQueuedSec += duration;
    } else {
      scheduleAudioBuffer(buffer, duration);
    }
    audioHeader = null;
  };
}

function scheduleAudioBuffer(buffer, duration) {
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  if (audioScheduledTime < audioCtx.currentTime + AUDIO_JITTER_SEC) {
    if (DEBUG) {
      warn("Audio jitter bump", { from: audioScheduledTime, to: audioCtx.currentTime + AUDIO_JITTER_SEC });
    }
    audioScheduledTime = audioCtx.currentTime + AUDIO_JITTER_SEC;
  }
  src.start(audioScheduledTime);
  audioScheduledTime += duration;
}

function audioBufferedSeconds() {
  if (!audioCtx) return 0;
  if (!audioStarted) return audioQueuedSec;
  return Math.max(0, audioScheduledTime - audioCtx.currentTime);
}

function meshBufferedSeconds() {
  return workerQueueLen / streamFps;
}

function computePlaybackFps(bufferSec) {
  const minFps = 18;
  const maxFps = streamFps;
  const t = Math.min(1, Math.max(0, (bufferSec - 0.5) / 3.5));
  return minFps + t * (maxFps - minFps);
}

function tryStartPlayback() {
  if (audioStarted || !audioEnabled || !audioCtx) return;
  const meshBuf = Math.max(workerQueueLen / streamFps, (workerMaxIndex + 1) / streamFps);
  const audioBuf = audioQueuedSec;
  if (meshBuf >= 1.0 && audioBuf >= 1.0) {
    audioStartTime = audioCtx.currentTime + 0.1;
    audioScheduledTime = audioStartTime;
    for (const item of audioQueue) {
      scheduleAudioBuffer(item.buffer, item.duration);
    }
    audioQueue = [];
    audioQueuedSec = 0;
    audioStarted = true;
    log("Audio started:", { audioStartTime });
  } else if (DEBUG) {
    log("Audio wait", { meshBuf, audioBuf, workerQueueLen, workerMaxIndex });
  }
}

function updatePlaybackFrame() {
  updatePlaybackFrameWorker();
}

function updatePlaybackFrameWorker() {
  if (!workerEnabled) {
    pipelineModeEl.textContent = "Unsupported";
    playStateEl.textContent = "holding";
    if (audioCtx) {
      audioCtx.suspend();
    }
    return;
  }
  if (!workerAlive) {
    playStateEl.textContent = "holding";
    if (audioCtx) {
      audioCtx.suspend();
    }
    return;
  }
  if (manualPaused) {
    playStateEl.textContent = "paused";
    return;
  }
  tryStartPlayback();
  if (!audioStarted || !audioCtx || audioStartTime === null) {
    playStateEl.textContent = "buffering";
    return;
  }
  if (audioBufferedSeconds() < AUDIO_LOW_BUFFER_SEC) {
    playStateEl.textContent = "holding";
    return;
  }
  const elapsed = audioCtx.currentTime - audioStartTime;
  if (elapsed < 0) {
    playStateEl.textContent = "buffering";
    return;
  }
  if (worker) {
    worker.postMessage({ type: "tick", elapsed });
  }
  if (frameDirty && workerFrame) {
    updateVertices(workerFrame, workerBounds);
    frameDirty = false;
    playCount += 1;
    playStateEl.textContent = "playing";
    lastFrameUpdate = performance.now();
  } else {
    playStateEl.textContent = "holding";
  }
  if (workerFrameIndex >= 0) {
    currentFrameIndex = workerFrameIndex;
  }
  const bufferSec = meshBufferedSeconds();
  currentPlayFps = computePlaybackFps(bufferSec);
  if (workerMaxIndex >= 0) {
    const audioFrame = Math.floor(elapsed * streamFps);
    const drift = audioFrame - workerMaxIndex;
    const allowResync = bufferSec < 2.0;
    if (allowResync && drift > DRIFT_THRESHOLD_FRAMES) {
      if (driftStart === null) {
        driftStart = performance.now();
      } else if (performance.now() - driftStart > RESYNC_GRACE_MS) {
        const oldStart = audioStartTime;
        audioStartTime = audioCtx.currentTime - (workerMaxIndex / streamFps);
        driftStart = null;
        warn("Resync drift", { drift, oldStart, newStart: audioStartTime, maxIndex: workerMaxIndex });
      }
    } else {
      driftStart = null;
    }
  }
  if (DEBUG && audioStarted) {
    const now = performance.now();
    if (now - lastFrameUpdate > 2000) {
      warn("No frames applied for >2s", { workerQueueLen, workerReady, streamFps });
      lastFrameUpdate = now;
    }
  }
}

function updateHud() {
  const now = performance.now();
  const dt = (now - lastRateTime) / 1000;
  if (dt > 0.4) {
    const inFps = workerInFps ? workerInFps : 0;
    const outFps = Math.round(playCount / dt);
    inFpsEl.textContent = `${inFps}`;
    outFpsEl.textContent = `${outFps}`;
    playCount = 0;
    lastRateTime = now;
  }
  const bufferSec = meshBufferedSeconds();
  bufferSecEl.textContent = `${bufferSec.toFixed(1)}s`;
  queueLenEl.textContent = `${workerQueueLen}`;
  const displayPlayFps = workerPlaybackFps ? workerPlaybackFps : currentPlayFps;
  playFpsEl.textContent = `${Math.round(displayPlayFps)}`;
  streamFpsEl.textContent = `${streamFps}`;
  audioBufferEl.textContent = `${audioBufferedSeconds().toFixed(1)}s`;
  bufferFillEl.style.width = `${Math.min(100, (bufferSec / maxBufferSeconds) * 100)}%`;
  let lod = 0;
  if (bufferSec < 2) {
    lod = 2;
  } else if (bufferSec < 4) {
    lod = 1;
  } else {
    lod = 0;
  }
  applyLod(lod);
  if (DEBUG) {
    const now = performance.now();
    if (now - lastDebugSummary > 1000) {
      lastDebugSummary = now;
      log("HUD", {
        status: statusEl.textContent,
        pipeline: pipelineModeEl.textContent,
        bufferSec: bufferSec.toFixed(2),
        queueLen: workerQueueLen,
        inFps: inFpsEl.textContent,
        outFps: outFpsEl.textContent,
        streamFps,
        audioBuf: audioBufferedSeconds().toFixed(2),
        audioStarted,
        workerReady,
      });
    }
  }
  requestAnimationFrame(updateHud);
}

togglePlayBtn.addEventListener("click", () => {
  manualPaused = !manualPaused;
  togglePlayBtn.textContent = manualPaused ? "Resume" : "Pause";
  if (audioCtx) {
    if (manualPaused) {
      audioCtx.suspend();
    } else {
      audioCtx.resume();
    }
  }
});

clearBufferBtn.addEventListener("click", () => {
  currentFrameIndex = 0;
  workerQueueLen = 0;
  workerMaxIndex = -1;
  workerMinIndex = -1;
  workerFrameIndex = -1;
  workerFrame = null;
  frameDirty = false;
  if (worker) {
    worker.postMessage({ type: "reset" });
  }
});

resetCamBtn.addEventListener("click", () => {
  camera.position.set(1.2, 1.4, 2.5);
  controls.target.set(0, 1.0, 0);
  controls.update();
});

enableAudioBtn.addEventListener("click", async () => {
  if (audioEnabled) return;
  audioEnabled = true;
  audioStatusEl.textContent = "enabled";
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  connectAudioSocket();
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);
});

await loadFaces();
if (workerEnabled) {
  initWorker();
} else {
  pipelineModeEl.textContent = "Unsupported";
  statusEl.textContent = "Disconnected";
}
animate();
updateHud();
