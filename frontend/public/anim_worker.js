let ws = null;
let wsHost = null;
let streamFps = 20;
let bones = [];
let morphs = [];
let nbones = 0;
let nmorphs = 0;
let bufferSeconds = 10;
let tickCount = 0;
const DEBUG = true;

let blinkConfig = null;
let blinkLeftIndices = [];
let blinkRightIndices = [];
let blinkNextAt = null;
let blinkStartAt = null;
let mouthConfig = null;
let mouthMorphIndices = [];
let jawBoneIndex = -1;

let saccadeConfig = { intervalSec: [1.0, 3.0], yawDeg: 2.0, pitchDeg: 1.0 };
let saccadeNextAt = null;
let saccadeYaw = 0;
let saccadePitch = 0;
let saccadeTargetYaw = 0;
let saccadeTargetPitch = 0;
let eyeBoneIndices = [];
let morphSmoothAlpha = null;
let prevMorphs = null;

let baseFrame = null;
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
  prevMorphs = null;
}

function ensureCapacity() {
  capacity = Math.max(1, Math.ceil(streamFps * bufferSeconds));
  resetBuffer();
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

function slerpQuat(ax, ay, az, aw, bx, by, bz, bw, t) {
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) {
    cos = -cos;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  if (1.0 - cos < 1e-6) {
    const k0 = 1 - t;
    const k1 = t;
    return [ax * k0 + bx * k1, ay * k0 + by * k1, az * k0 + bz * k1, aw * k0 + bw * k1];
  }
  const theta = Math.acos(cos);
  const sinT = Math.sin(theta);
  const k0 = Math.sin((1 - t) * theta) / sinT;
  const k1 = Math.sin(t * theta) / sinT;
  return [ax * k0 + bx * k1, ay * k0 + by * k1, az * k0 + bz * k1, aw * k0 + bw * k1];
}

function randRange(min, max) {
  return min + Math.random() * (max - min);
}

function quatMul(ax, ay, az, aw, bx, by, bz, bw) {
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function quatFromYawPitch(yaw, pitch) {
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);
  const cx = Math.cos(pitch * 0.5);
  const sx = Math.sin(pitch * 0.5);
  const qy = [0, sy, 0, cy];
  const qx = [sx, 0, 0, cx];
  const q = quatMul(qy[0], qy[1], qy[2], qy[3], qx[0], qx[1], qx[2], qx[3]);
  return q;
}

function updateBlink(elapsed) {
  if (!blinkConfig) return 0;
  const interval = blinkConfig.intervalSec || [3.0, 6.0];
  const duration = blinkConfig.durationSec || 0.12;
  if (blinkNextAt === null) {
    blinkNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (blinkStartAt === null && elapsed >= blinkNextAt) {
    blinkStartAt = elapsed;
    blinkNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (blinkStartAt !== null) {
    const t = (elapsed - blinkStartAt) / duration;
    if (t >= 1) {
      blinkStartAt = null;
      return 0;
    }
    const tri = t < 0.5 ? t * 2 : (1 - t) * 2;
    return tri * (blinkConfig.strength || 0.6);
  }
  return 0;
}

function updateSaccade(elapsed) {
  const interval = saccadeConfig.intervalSec || [1.0, 3.0];
  if (saccadeNextAt === null) {
    saccadeNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (elapsed >= saccadeNextAt) {
    const yaw = (saccadeConfig.yawDeg || 2.0) * (Math.PI / 180);
    const pitch = (saccadeConfig.pitchDeg || 1.0) * (Math.PI / 180);
    saccadeTargetYaw = randRange(-yaw, yaw);
    saccadeTargetPitch = randRange(-pitch, pitch);
    saccadeNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  saccadeYaw += (saccadeTargetYaw - saccadeYaw) * 0.15;
  saccadePitch += (saccadeTargetPitch - saccadePitch) * 0.15;
  return { yaw: saccadeYaw, pitch: saccadePitch };
}

function applyExtras(buffer, elapsed) {
  if (!buffer) return;
  if (blinkConfig && (blinkLeftIndices.length || blinkRightIndices.length)) {
    const val = updateBlink(elapsed);
    if (val > 0) {
      const morphOffset = 3 + nbones * 4;
      for (const idx of blinkLeftIndices) {
        const i = morphOffset + idx;
        buffer[i] = Math.max(-1, Math.min(1, buffer[i] + val));
      }
      for (const idx of blinkRightIndices) {
        const i = morphOffset + idx;
        buffer[i] = Math.max(-1, Math.min(1, buffer[i] + val));
      }
    }
  }
  if (eyeBoneIndices.length) {
    const { yaw, pitch } = updateSaccade(elapsed);
    const dq = quatFromYawPitch(yaw, pitch);
    for (const boneIndex of eyeBoneIndices) {
      const qOffset = 3 + boneIndex * 4;
      const qx = buffer[qOffset];
      const qy = buffer[qOffset + 1];
      const qz = buffer[qOffset + 2];
      const qw = buffer[qOffset + 3];
      const nq = quatMul(dq[0], dq[1], dq[2], dq[3], qx, qy, qz, qw);
      buffer[qOffset] = nq[0];
      buffer[qOffset + 1] = nq[1];
      buffer[qOffset + 2] = nq[2];
      buffer[qOffset + 3] = nq[3];
    }
  }
}


function applyMorphSmoothing(buffer) {
  if (!buffer || !morphSmoothAlpha || morphSmoothAlpha >= 1 || nmorphs <= 0) {
    return;
  }
  const offset = 3 + nbones * 4;
  if (!prevMorphs || prevMorphs.length !== nmorphs) {
    prevMorphs = new Float32Array(nmorphs);
    for (let i = 0; i < nmorphs; i++) {
      prevMorphs[i] = buffer[offset + i];
    }
    return;
  }
  const alpha = morphSmoothAlpha;
  for (let i = 0; i < nmorphs; i++) {
    const v = buffer[offset + i];
    const next = prevMorphs[i] + (v - prevMorphs[i]) * alpha;
    prevMorphs[i] = next;
    buffer[offset + i] = next;
  }
}

function debugMorphs(buffer) {
  if (!buffer || nmorphs <= 0) return;
  if (tickCount % 100 !== 0) return;
  const offset = 3 + nbones * 4;
  const end = offset + nmorphs;
  let maxVal = -Infinity;
  for (let i = offset; i < end; i++) {
    const v = buffer[i];
    if (v > maxVal) maxVal = v;
  }
  const first = [];
  for (let i = 0; i < Math.min(10, nmorphs); i++) {
    first.push(buffer[offset + i]);
  }
  console.log("[Debug] Morph Index:", offset, "Max Value in Frame:", maxVal.toFixed(4));
  console.log("[Debug] First 10 Morphs:", first);
}

function emitFrame(frameIndex, data) {
  const out = new Float32Array(data.length);
  out.set(data);
  debugMorphs(out);
  postMessage(
    {
      type: "frame",
      buffer: out.buffer,
      frameIndex,
      streamFps,
      queueLen: storedCount,
      dropped,
      nbones,
      nmorphs,
    },
    [out.buffer]
  );
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
    baseFrame,
    maxIndex,
    minIndexEstimate,
  });
  recvCount = 0;
  lastStatsTime = now;
}

function onTick(elapsed) {
  if (baseFrame === null) return;
  const frameFloat = elapsed * streamFps;
  let i0 = Math.floor(frameFloat);
  let i1 = i0 + 1;
  const alpha = Math.max(0, Math.min(1, frameFloat - i0));
  if (maxIndex >= 0 && i0 > maxIndex) {
    i0 = maxIndex;
    i1 = i0;
  }
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
    // root
    interpBuffer[0] = a[0] * (1 - alpha) + b[0] * alpha;
    interpBuffer[1] = a[1] * (1 - alpha) + b[1] * alpha;
    interpBuffer[2] = a[2] * (1 - alpha) + b[2] * alpha;
    let offset = 3;
    for (let i = 0; i < nbones; i++) {
      const ax = a[offset];
      const ay = a[offset + 1];
      const az = a[offset + 2];
      const aw = a[offset + 3];
      const bx = b[offset];
      const by = b[offset + 1];
      const bz = b[offset + 2];
      const bw = b[offset + 3];
      const q = slerpQuat(ax, ay, az, aw, bx, by, bz, bw, alpha);
      interpBuffer[offset] = q[0];
      interpBuffer[offset + 1] = q[1];
      interpBuffer[offset + 2] = q[2];
      interpBuffer[offset + 3] = q[3];
      offset += 4;
    }
    for (let i = 0; i < nmorphs; i++) {
      interpBuffer[offset + i] = a[offset + i] * (1 - alpha) + b[offset + i] * alpha;
    }
    applyExtras(interpBuffer, elapsed);
    applyMorphSmoothing(interpBuffer);
    if (tickCount % 60 === 0) console.log("[Worker] Emitting interpolated frame", { i0, alpha: alpha.toFixed(2), queueLen: storedCount });
    emitFrame(i0, interpBuffer);
  } else if (a) {
    const out = new Float32Array(a.length);
    out.set(a);
    applyExtras(out, elapsed);
    applyMorphSmoothing(out);
    if (tickCount % 60 === 0) console.log("[Worker] Emitting direct frame", { i0, queueLen: storedCount });
    emitFrame(i0, out);
  }
  maybeSendStats();
}

function connectAnim() {
  const host = wsHost || (self.location ? self.location.host : "127.0.0.1:8000");
  ws = new WebSocket(`ws://${host}/ws/anim`);
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
    if (typeof event.data === "string") {
      try {
        header = JSON.parse(event.data);
      } catch {
        header = null;
      }
      if (header && header.type === "anim_init") {
        streamFps = header.fps || streamFps;
        bones = header.bones || [];
        morphs = header.morphs || [];
        if (Number.isFinite(header.bufferSeconds)) {
          bufferSeconds = header.bufferSeconds;
        }
        blinkConfig = header.blink || null;
        blinkLeftIndices = [];
        blinkRightIndices = [];
        if (blinkConfig) {
          const left = blinkConfig.left || [];
          const right = blinkConfig.right || [];
          for (const name of left) {
            const idx = morphs.indexOf(name);
            if (idx >= 0) blinkLeftIndices.push(idx);
          }
          for (const name of right) {
            const idx = morphs.indexOf(name);
            if (idx >= 0) blinkRightIndices.push(idx);
          }
        }
        mouthConfig = header.mouth || null;
        mouthMorphIndices = [];
        if (mouthConfig && mouthConfig.morphs) {
          for (const name of mouthConfig.morphs) {
            const idx = morphs.indexOf(name);
            if (idx >= 0) mouthMorphIndices.push(idx);
          }
        }
        morphSmoothAlpha =
          typeof header.morphSmoothAlpha === "number" ? header.morphSmoothAlpha : null;
        if (header.saccade) {
          saccadeConfig = header.saccade;
        }
        nbones = bones.length;
        nmorphs = morphs.length;
        eyeBoneIndices = [];
        jawBoneIndex = -1;
        for (let i = 0; i < bones.length; i++) {
          const name = (bones[i] || "").toLowerCase();
          if (name.includes("eye")) {
            eyeBoneIndices.push(i);
          }
          if (jawBoneIndex < 0 && name.includes("jaw")) {
            jawBoneIndex = i;
          }
        }
        baseFrame = null;
        ensureCapacity();
        postMessage({
          type: "init",
          streamFps,
          bones,
          morphs,
          bufferSeconds,
          blink: blinkConfig,
          saccade: saccadeConfig,
          mouth: mouthConfig,
        });
      }
      return;
    }
    if (event.data instanceof ArrayBuffer) {
      if (header && header.type === "anim") {
        const frame = header.frame ?? 0;
        const arr = new Float32Array(event.data);
        if (!Number.isFinite(baseFrame) || frame < baseFrame) {
          baseFrame = frame;
          resetBuffer();
        }
        const rel = frame - baseFrame;
        storeFrame(rel, arr);
        recvCount += 1;
        updateEffectiveFps();
        maybeSendStats();
        if (tickCount % 200 === 0) {
          console.log("[Worker] Frame stored in buffer:", { frame, rel, queueLen: storedCount });
        }
        header = null; // Clear header after processing binary data
      } else {
        console.warn("[Worker] Received binary data without matching 'anim' header. Current header type:", header ? header.type : "null");
      }
    }
  };
}

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    wsHost = msg.host || wsHost;
    connectAnim();
    onTick(msg.elapsed || 0);
    tickCount++;
  } else if (msg.type === "tick") {
    onTick(msg.elapsed);
    tickCount++;
  } else if (msg.type === "reset") {
    baseFrame = null;
    resetBuffer();
  }
};
