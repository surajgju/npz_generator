let ws = null;
let wsHost = null;
let streamFps = 20;
let playbackFps = 20;
let baseFrame = null;
let baseStreamFps = null;
let nverts = null;

let capacity = 1;
let frameSlots = [];
let frameMarks = null;
let storedCount = 0;
let dropped = 0;

let recvCount = 0;
let lastStatsTime = performance.now();
let maxIndex = -1;
let lastRecvTime = null;
let effectiveFps = 0;

const canShared = typeof SharedArrayBuffer !== "undefined";
let sharedBuffers = null;
let sharedIndex = 0;
let interpBuffer = null;

function resetBuffer() {
  frameSlots = new Array(capacity);
  frameMarks = new Int32Array(capacity);
  frameMarks.fill(-1);
  storedCount = 0;
  dropped = 0;
  maxIndex = -1;
  lastRecvTime = null;
  effectiveFps = 0;
}

function ensureCapacity() {
  capacity = Math.max(1, Math.ceil(streamFps * 10));
  resetBuffer();
}

function initSharedBuffers() {
  if (!canShared || !nverts) return;
  const bytes = nverts * 3 * 4;
  sharedBuffers = [new SharedArrayBuffer(bytes), new SharedArrayBuffer(bytes)];
}

function storeFrame(idx, data) {
  if (!frameMarks) {
    ensureCapacity();
  }
  const slot = idx % capacity;
  if (frameMarks[slot] !== -1 && frameMarks[slot] !== idx) {
    dropped += 1;
  } else if (frameMarks[slot] === -1) {
    storedCount += 1;
  }
  frameSlots[slot] = data;
  frameMarks[slot] = idx;
  if (idx > maxIndex) maxIndex = idx;
}

function getFrame(idx) {
  const slot = idx % capacity;
  if (frameMarks[slot] === idx) return frameSlots[slot];
  return null;
}

function computeBounds(arr) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < arr.length; i += 3) {
    const x = arr[i];
    const y = arr[i + 1];
    const z = arr[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function emitFrame(frameIndex, data, bounds) {
  if (canShared && sharedBuffers) {
    const sab = sharedBuffers[sharedIndex];
    new Float32Array(sab).set(data);
    postMessage(
      {
        type: "frame",
        buffer: sab,
        shared: true,
        bounds,
        frameIndex,
        streamFps,
        playbackFps,
        nverts,
        queueLen: storedCount,
        dropped,
      }
    );
    sharedIndex = 1 - sharedIndex;
  } else {
    const out = new Float32Array(data.length);
    out.set(data);
    postMessage(
      {
        type: "frame",
        buffer: out.buffer,
        shared: false,
        bounds,
        frameIndex,
        streamFps,
        playbackFps,
        nverts,
        queueLen: storedCount,
        dropped,
      },
      [out.buffer]
    );
  }
}

function updateEffectiveFps() {
  const now = performance.now();
  if (lastRecvTime !== null) {
    const dt = (now - lastRecvTime) / 1000;
    if (dt > 0) {
      const inst = 1 / dt;
      if (Number.isFinite(inst)) {
        effectiveFps = effectiveFps ? (effectiveFps * 0.9 + inst * 0.1) : inst;
      }
    }
  }
  lastRecvTime = now;
  playbackFps = effectiveFps > 0 ? effectiveFps : streamFps;
}

function maybeSendStats() {
  const now = performance.now();
  const dt = (now - lastStatsTime) / 1000;
  if (dt < 0.4) return;
  const inFps = Math.round(recvCount / dt);
  const minIndexEstimate = maxIndex - capacity + 1;
  postMessage({
    type: "stats",
    queueLen: storedCount,
    dropped,
    inFps,
    streamFps,
    playbackFps,
    baseFrame,
    maxIndex,
    minIndexEstimate,
  });
  recvCount = 0;
  lastStatsTime = now;
}

function onTick(elapsed) {
  if (baseFrame === null) return;
  const fps = streamFps;
  const frameFloat = elapsed * fps;
  let i0 = Math.floor(frameFloat);
  let i1 = i0 + 1;
  if (maxIndex >= 0 && i0 > maxIndex) {
    i0 = maxIndex;
    i1 = i0;
  }
  const alpha = frameFloat - i0;
  let a = getFrame(i0);
  let b = getFrame(i1);
  if (!a && maxIndex >= 0) {
    for (let j = 1; j <= 3; j++) {
      const cand = getFrame(i0 - j);
      if (cand) {
        a = cand;
        break;
      }
    }
  }
  if (!b && maxIndex >= 0) {
    for (let j = 1; j <= 3; j++) {
      const cand = getFrame(i1 - j);
      if (cand) {
        b = cand;
        break;
      }
    }
  }
  if (a && b) {
    if (!interpBuffer || interpBuffer.length !== a.length) {
      interpBuffer = new Float32Array(a.length);
    }
    for (let i = 0; i < a.length; i++) {
      interpBuffer[i] = a[i] * (1 - alpha) + b[i] * alpha;
    }
    emitFrame(i0, interpBuffer, computeBounds(interpBuffer));
  } else if (a) {
    emitFrame(i0, a, computeBounds(a));
  }
  maybeSendStats();
}

function connectVerts() {
  const host = wsHost || (self.location ? self.location.host : "127.0.0.1:8000");
  ws = new WebSocket(`ws://${host}/ws/verts`);
  ws.binaryType = "arraybuffer";
  let header = null;
  ws.onopen = () => {
    postMessage({ type: "status", status: "connected" });
  };
  ws.onclose = () => {
    postMessage({ type: "status", status: "closed" });
  };
  ws.onerror = () => {
    postMessage({ type: "status", status: "error" });
  };
  ws.onmessage = (event) => {
    try {
      if (typeof event.data === "string") {
        try {
          header = JSON.parse(event.data);
        } catch {
          header = null;
        }
        return;
      }
      if (!header) return;
      if (Number.isFinite(header.fps)) {
        streamFps = header.fps;
        if (baseStreamFps !== null && baseStreamFps !== streamFps) {
          baseFrame = null;
          baseStreamFps = streamFps;
          ensureCapacity();
        }
        if (baseStreamFps === null) {
          baseStreamFps = streamFps;
          ensureCapacity();
        }
      }
      const rawFrame = Number.isFinite(header.frame) ? header.frame : maxIndex + 1;
      if (baseFrame === null) {
        baseFrame = rawFrame;
      } else if (rawFrame < baseFrame) {
        baseFrame = rawFrame;
        ensureCapacity();
      }
      const idx = rawFrame - baseFrame;
      let arr;
      if (header.dtype === "int16" && header.quant === "minmax") {
        const q = new Int16Array(event.data);
        const min = header.min || [0, 0, 0];
        const scale = header.scale || [1, 1, 1];
        const out = new Float32Array(q.length);
        for (let i = 0; i < q.length; i++) {
          const axis = i % 3;
          out[i] = (q[i] + 32768) * scale[axis] + min[axis];
        }
        arr = out;
      } else {
        arr = new Float32Array(event.data);
      }
      if (!nverts) {
        nverts = header.nverts;
        initSharedBuffers();
      }
      storeFrame(idx, arr);
      updateEffectiveFps();
      recvCount += 1;
      maybeSendStats();
      header = null;
    } catch (err) {
      postMessage({ type: "status", status: "error", detail: String(err) });
    }
  };
}

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    wsHost = msg.host || null;
    ensureCapacity();
    connectVerts();
    return;
  }
  if (msg.type === "reset") {
    baseFrame = null;
    resetBuffer();
    return;
  }
  if (msg.type === "tick") {
    onTick(msg.elapsed);
  }
};
