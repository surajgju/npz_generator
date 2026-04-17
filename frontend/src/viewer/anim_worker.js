let ws = null;
let wsHost = null;
let streamFps = 20;
let bones = [];
let morphs = [];
let nbones = 0;
let nmorphs = 0;
let bufferSeconds = 10;
const DEBUG = false;
const PROTOCOL_VERSION = 2;

let buildId = "dev";
let knownBootId = null;
let knownSessionId = null;
let knownServerClockId = null;
let knownLastAppliedFrame = -1;
let currentSessionId = null;
let serverBootId = null;
let serverClockId = null;

let blinkConfig = null;
let blinkLeftIndices = [];
let blinkRightIndices = [];
let blinkNextAt = null;
let blinkStartAt = null;

let saccadeConfig = { intervalSec: [1.0, 3.0], yawDeg: 2.0, pitchDeg: 1.0 };
let saccadeNextAt = null;
let saccadeYaw = 0;
let saccadePitch = 0;
let saccadeTargetYaw = 0;
let saccadeTargetPitch = 0;
let eyeBoneIndices = [];
let morphSmoothAlpha = null;
let prevMorphs = null;

let capacity = 1;
let frameSlots = [];
let frameMarks = null;
let framePhase = null;
let storedCount = 0;
let snapshotDropCount = 0;
let liveDropCount = 0;
let resyncSkipped = 0;
let recvCount = 0;
let lastStatsTime = performance.now();
let maxIndex = -1;
let lastRecvTime = null;
let effectiveFps = 0;
let interpBuffer = null;
let tickCount = 0;
let lastAppliedFrame = -1;

let serverOffsetMs = 0;
let hasServerOffset = false;

let playbackState = "live_playing"; // snapshot_loading | tail_lock_align | live_playing | resyncing
let targetLiveFrame = null;
let tailLockStartMs = 0;
let seenLiveAtOrBeyondTarget = false;
let lastResyncRequestMs = 0;
const MAX_TAIL_LOCK_MS = 1200;
const MAX_DRIFT_MS = 100;
function resetTimelineState() {
  targetLiveFrame = null;
  tailLockStartMs = 0;
  seenLiveAtOrBeyondTarget = false;
  lastAppliedFrame = -1;
  if (!frameMarks) ensureCapacity();
  else resetBuffer();
}

function switchSession(nextSessionId, mode = "live_playing", reason = "session_switch") {
  if (!nextSessionId || nextSessionId === currentSessionId) return false;
  currentSessionId = nextSessionId;
  knownSessionId = nextSessionId;
  resetTimelineState();
  playbackState = mode;
  postMessage({
    type: "session_switch",
    sessionId: currentSessionId,
    playbackState,
    reason,
  });
  return true;
}

function debugLog(...args) {
  if (DEBUG) console.log("[Worker]", ...args);
}

function resetBuffer() {
  frameSlots = new Array(capacity);
  frameMarks = new Int32Array(capacity);
  framePhase = new Uint8Array(capacity);
  frameMarks.fill(-1);
  framePhase.fill(0);
  storedCount = 0;
  snapshotDropCount = 0;
  liveDropCount = 0;
  resyncSkipped = 0;
  maxIndex = -1;
  lastRecvTime = null;
  effectiveFps = 0;
  prevMorphs = null;
}

function ensureCapacity() {
  capacity = Math.max(1, Math.ceil(streamFps * bufferSeconds));
  resetBuffer();
}

function observeServerTime(serverTimeMs) {
  if (!Number.isFinite(serverTimeMs)) return;
  const raw = serverTimeMs - performance.now();
  if (!hasServerOffset) {
    serverOffsetMs = raw;
    hasServerOffset = true;
  } else {
    // EMA smoothing
    serverOffsetMs = serverOffsetMs * 0.9 + raw * 0.1;
    // If drift exceeds threshold, snap immediately
    if (Math.abs(serverOffsetMs - raw) > MAX_DRIFT_MS) {
      console.warn(`[Worker] Large drift detected: ${Math.abs(serverOffsetMs - raw).toFixed(2)}ms, snapping.`);
      serverOffsetMs = raw;
    }
  }
}

function serverNowEstimateMs() {
  if (!hasServerOffset) return performance.now();
  return performance.now() + serverOffsetMs;
}

function storeFrame(idx, data, phaseName) {
  if (!frameMarks) ensureCapacity();
  const slot = idx % capacity;
  const phase = phaseName === "snapshot" ? 1 : 2;
  if (frameMarks[slot] !== -1 && frameMarks[slot] !== idx) {
    if (phase === 1) snapshotDropCount += 1;
    else liveDropCount += 1;
  } else if (frameMarks[slot] === -1) {
    storedCount += 1;
  }
  frameSlots[slot] = data;
  frameMarks[slot] = idx;
  framePhase[slot] = phase;
  if (idx > maxIndex) maxIndex = idx;
}

function getFrame(idx) {
  if (!frameMarks) return null;
  const slot = idx % capacity;
  if (frameMarks[slot] === idx) return frameSlots[slot];
  return null;
}

function getNearestFrameAtOrBefore(idx) {
  for (let back = 0; back < capacity; back++) {
    const k = idx - back;
    if (k < 0) break;
    const fr = getFrame(k);
    if (fr) return { frame: k, data: fr };
  }
  return null;
}

function getLatestFrame() {
  if (maxIndex < 0) return null;
  return getNearestFrameAtOrBefore(maxIndex);
}

function slerpQuat(ax, ay, az, aw, bx, by, bz, bw, t) {
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) {
    cos = -cos;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
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
  return quatMul(qy[0], qy[1], qy[2], qy[3], qx[0], qx[1], qx[2], qx[3]);
}

function updateBlink(elapsed) {
  if (!blinkConfig) return 0;
  const interval = blinkConfig.intervalSec || [3.0, 6.0];
  const duration = blinkConfig.durationSec || 0.12;
  if (blinkNextAt === null) blinkNextAt = elapsed + randRange(interval[0], interval[1]);
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
  if (saccadeNextAt === null) saccadeNextAt = elapsed + randRange(interval[0], interval[1]);
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
  if (!buffer || !morphSmoothAlpha || morphSmoothAlpha >= 1 || nmorphs <= 0) return;
  const offset = 3 + nbones * 4;
  if (!prevMorphs || prevMorphs.length !== nmorphs) {
    prevMorphs = new Float32Array(nmorphs);
    for (let i = 0; i < nmorphs; i++) prevMorphs[i] = buffer[offset + i];
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

function emitFrame(frameIndex, data) {
  if (frameIndex < lastAppliedFrame && DEBUG) {
    debugLog("emitFrame backward index", { frameIndex, lastAppliedFrame, sessionId: currentSessionId });
  }
  const out = new Float32Array(data.length);
  out.set(data);
  postMessage(
    {
      type: "frame",
      buffer: out.buffer,
      frameIndex,
      streamFps,
      queueLen: storedCount,
      snapshotDropCount,
      liveDropCount,
      resyncSkipped,
      nbones,
      nmorphs,
      sessionId: currentSessionId,
      serverBootId,
      serverClockId,
      protocolVersion: PROTOCOL_VERSION,
    },
    [out.buffer]
  );
  lastAppliedFrame = Math.max(lastAppliedFrame, frameIndex);
}

function updateEffectiveFps() {
  const now = performance.now();
  if (lastRecvTime !== null) {
    const dt = (now - lastRecvTime) / 1000;
    if (dt > 0) {
      const inst = 1 / dt;
      if (Number.isFinite(inst)) effectiveFps = effectiveFps ? effectiveFps * 0.9 + inst * 0.1 : inst;
    }
  }
  lastRecvTime = now;
}

function maybeSendStats() {
  const now = performance.now();
  const dt = (now - lastStatsTime) / 1000;
  if (dt < 0.4) return;
  const inFps = Math.round(recvCount / dt);
  const minIndexEstimate = maxIndex >= 0 ? Math.max(0, maxIndex - capacity + 1) : -1;
  postMessage({
    type: "stats",
    queueLen: storedCount,
    snapshotDropCount,
    liveDropCount,
    resyncSkipped,
    inFps,
    streamFps,
    maxIndex,
    minIndexEstimate,
    playbackState,
    targetLiveFrame,
    sessionId: currentSessionId,
  });
  recvCount = 0;
  lastStatsTime = now;
}

function sendResyncRequest(reason) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !currentSessionId) return;
  const now = performance.now();
  if (now - lastResyncRequestMs < 500) return;
  lastResyncRequestMs = now;
  resyncSkipped += 1;
  ws.send(
    JSON.stringify({
      type: "resync_request",
      stream_session_id: currentSessionId,
      session_id: currentSessionId,
      reason,
    })
  );
  postMessage({ type: "resync", reason, sessionId: currentSessionId });
}

function onTick(elapsed) {
  if (playbackState === "snapshot_loading") {
    maybeSendStats();
    return;
  }
  if (playbackState === "tail_lock_align") {
    const target = Number.isFinite(targetLiveFrame) ? targetLiveFrame : maxIndex;
    const best = getNearestFrameAtOrBefore(target) || getLatestFrame();
    if (best) {
      const out = new Float32Array(best.data.length);
      out.set(best.data);
      applyExtras(out, elapsed);
      applyMorphSmoothing(out);
      // During tail lock we always emit the hold frame so viewer frameDirty
      // stays true even when frame index does not advance.
      const holdOut = new Float32Array(out.length);
      holdOut.set(out);
      postMessage(
        {
          type: "frame",
          buffer: holdOut.buffer,
          frameIndex: best.frame,
          streamFps,
          queueLen: storedCount,
          snapshotDropCount,
          liveDropCount,
          resyncSkipped,
          nbones,
          nmorphs,
          sessionId: currentSessionId,
          serverBootId,
          serverClockId,
          protocolVersion: PROTOCOL_VERSION,
        },
        [holdOut.buffer]
      );
    }
    if (seenLiveAtOrBeyondTarget) {
      playbackState = "live_playing";
    } else if (performance.now() - tailLockStartMs > MAX_TAIL_LOCK_MS) {
      playbackState = "resyncing";
      sendResyncRequest("tail_lock_timeout");
    }
    maybeSendStats();
    return;
  }
  if (playbackState === "resyncing") {
    maybeSendStats();
    return;
  }
  if (maxIndex < 0) {
    maybeSendStats();
    return;
  }
  const frameFloat = elapsed * streamFps;
  let i0 = Math.floor(frameFloat);
  let i1 = i0 + 1;
  const alpha = Math.max(0, Math.min(1, frameFloat - i0));
  if (i0 > maxIndex) {
    i0 = maxIndex;
    i1 = i0;
  }
  let a = getFrame(i0);
  let b = getFrame(i1);
  if (!a) {
    for (let j = 1; j <= 3; j++) {
      const cand = getFrame(i0 - j);
      if (cand) {
        a = cand;
        i0 = i0 - j;
        break;
      }
    }
  }
  if (!b) {
    for (let j = 1; j <= 3; j++) {
      const cand = getFrame(i1 - j);
      if (cand) {
        b = cand;
        break;
      }
    }
  }
  if (a && b) {
    if (!interpBuffer || interpBuffer.length !== a.length) interpBuffer = new Float32Array(a.length);
    interpBuffer[0] = a[0] * (1 - alpha) + b[0] * alpha;
    interpBuffer[1] = a[1] * (1 - alpha) + b[1] * alpha;
    interpBuffer[2] = a[2] * (1 - alpha) + b[2] * alpha;
    let offset = 3;
    for (let i = 0; i < nbones; i++) {
      const q = slerpQuat(
        a[offset],
        a[offset + 1],
        a[offset + 2],
        a[offset + 3],
        b[offset],
        b[offset + 1],
        b[offset + 2],
        b[offset + 3],
        alpha
      );
      interpBuffer[offset] = q[0];
      interpBuffer[offset + 1] = q[1];
      interpBuffer[offset + 2] = q[2];
      interpBuffer[offset + 3] = q[3];
      offset += 4;
    }
    for (let i = 0; i < nmorphs; i++) interpBuffer[offset + i] = a[offset + i] * (1 - alpha) + b[offset + i] * alpha;
    applyExtras(interpBuffer, elapsed);
    applyMorphSmoothing(interpBuffer);
    emitFrame(i0, interpBuffer);
  } else if (a) {
    const out = new Float32Array(a.length);
    out.set(a);
    applyExtras(out, elapsed);
    applyMorphSmoothing(out);
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
    ws.send(
      JSON.stringify({
        type: "anim_subscribe",
        protocol_version: PROTOCOL_VERSION,
        known_boot_id: knownBootId,
        known_server_clock_id: knownServerClockId,
        known_stream_session_id: knownSessionId,
        known_session_id: knownSessionId,
        last_applied_frame: Number.isFinite(knownLastAppliedFrame) ? knownLastAppliedFrame : -1,
      })
    );
  };
  ws.onclose = () => postMessage({ type: "status", status: "closed" });
  ws.onerror = () => postMessage({ type: "status", status: "error" });
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      let msg = null;
      try {
        msg = JSON.parse(event.data);
      } catch {
        msg = null;
      }
      header = msg;
      if (!msg) return;
      if (msg.type === "anim_session_switch") {
        const sid = msg.stream_session_id || msg.session_id || currentSessionId;
        switchSession(sid, "live_playing", "server_session_switch");
        currentSessionId = sid;
        if (Number.isFinite(msg.server_time_ms)) observeServerTime(msg.server_time_ms);
        return;
      }
      if (msg.type === "anim_subscribe_ack") {
        serverBootId = msg.server_boot_id || serverBootId;
        serverClockId = msg.server_clock_id || serverClockId;
        if (knownServerClockId && serverClockId && knownServerClockId !== serverClockId) {
          postMessage({
            type: "status",
            status: "reset_required",
            reason: "server_clock_mismatch",
            serverBootId,
            serverClockId,
            sessionId: currentSessionId,
            protocolVersion: PROTOCOL_VERSION,
          });
          ws.close();
          return;
        }
        const ackSessionId = msg.stream_session_id || msg.session_id || currentSessionId;
        const modeFromAck = msg.mode === "resume" ? "snapshot_loading" : "live_playing";
        switchSession(ackSessionId, modeFromAck, "subscribe_ack");
        currentSessionId = ackSessionId;
        streamFps = msg.stream_fps || streamFps;
        observeServerTime(msg.server_time_ms);
        if (msg.mode === "reset_required") {
          postMessage({
            type: "status",
            status: "reset_required",
            serverBootId,
            serverClockId,
            sessionId: currentSessionId,
            protocolVersion: PROTOCOL_VERSION,
          });
          ws.close();
          return;
        }
        playbackState = modeFromAck;
        postMessage({
          type: "handshake",
          mode: msg.mode,
          protocolVersion: PROTOCOL_VERSION,
          serverBootId,
          serverClockId,
          sessionId: currentSessionId,
          buildId,
        });
        return;
      }
      if (msg.type === "anim_snapshot_start") {
        const sid = msg.stream_session_id || msg.session_id || currentSessionId;
        switchSession(sid, "snapshot_loading", "snapshot_start");
        playbackState = "snapshot_loading";
        targetLiveFrame = null;
        seenLiveAtOrBeyondTarget = false;
        return;
      }
      if (msg.type === "anim_snapshot_end") {
        const sid = msg.stream_session_id || msg.session_id || currentSessionId;
        if (msg.server_clock_id && serverClockId && msg.server_clock_id !== serverClockId) {
          playbackState = "resyncing";
          sendResyncRequest("clock_mismatch");
          return;
        }
        switchSession(sid, "snapshot_loading", "snapshot_end");
        currentSessionId = sid;
        observeServerTime(msg.live_head_server_time_ms);
        const liveHeadFrame = Number.isFinite(msg.live_head_frame) ? msg.live_head_frame : -1;
        const liveHeadServerTimeMs = Number.isFinite(msg.live_head_server_time_ms)
          ? msg.live_head_server_time_ms
          : serverNowEstimateMs();
        const audioLiveEdgeFrame = Number.isFinite(msg.audio_live_edge_frame) ? msg.audio_live_edge_frame : -1;
        const expectedFrame =
          liveHeadFrame >= 0
            ? liveHeadFrame + Math.round(((serverNowEstimateMs() - liveHeadServerTimeMs) * streamFps) / 1000)
            : -1;
        targetLiveFrame = Math.max(expectedFrame, audioLiveEdgeFrame, liveHeadFrame);
        playbackState = "tail_lock_align";
        tailLockStartMs = performance.now();
        seenLiveAtOrBeyondTarget = false;
        postMessage({
          type: "snapshot_anchor",
          sessionId: currentSessionId,
          targetLiveFrame,
          liveHeadFrame,
          audioLiveEdgeFrame,
        });
        return;
      }
      if (msg.type === "anim_init") {
        streamFps = msg.fps || streamFps;
        bones = msg.bones || [];
        morphs = msg.morphs || [];
        if (Number.isFinite(msg.bufferSeconds)) bufferSeconds = msg.bufferSeconds;
        blinkConfig = msg.blink || null;
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
        morphSmoothAlpha = typeof msg.morphSmoothAlpha === "number" ? msg.morphSmoothAlpha : null;
        if (msg.saccade) saccadeConfig = msg.saccade;
        nbones = bones.length;
        nmorphs = morphs.length;
        eyeBoneIndices = [];
        for (let i = 0; i < bones.length; i++) {
          const name = (bones[i] || "").toLowerCase();
          if (name.includes("eye")) eyeBoneIndices.push(i);
        }
        ensureCapacity();
        postMessage({
          type: "init",
          streamFps,
          bones,
          morphs,
          bufferSeconds,
          blink: blinkConfig,
          saccade: saccadeConfig,
          protocolVersion: PROTOCOL_VERSION,
          serverBootId,
          sessionId: currentSessionId,
          buildId,
        });
        return;
      }
      if (msg.type === "anim") {
        const sid = msg.stream_session_id || msg.session_id || null;
        if (sid && sid !== currentSessionId) {
          const nextMode = msg.phase === "snapshot" ? "snapshot_loading" : "live_playing";
          switchSession(sid, nextMode, "anim_header");
        }
        if (Number.isFinite(msg.server_time_ms)) observeServerTime(msg.server_time_ms);
        if (sid) currentSessionId = sid;
      }
      return;
    }
    if (!(event.data instanceof ArrayBuffer)) return;
    if (!header || header.type !== "anim") return;
    const frame = Number.isFinite(header.frame) ? header.frame : 0;
    const phaseName = header.phase || "live";
    currentSessionId = header.stream_session_id || header.session_id || currentSessionId;
    if (!Number.isFinite(nbones) || nbones <= 0) nbones = header.nbones || nbones;
    if (!Number.isFinite(nmorphs) || nmorphs <= 0) nmorphs = header.nmorphs || nmorphs;
    if (!frameMarks) ensureCapacity();
    const arr = new Float32Array(event.data);
    storeFrame(frame, arr, phaseName);
    recvCount += 1;
    updateEffectiveFps();
    if (phaseName === "live" && Number.isFinite(targetLiveFrame) && frame >= targetLiveFrame) {
      seenLiveAtOrBeyondTarget = true;
    }
    maybeSendStats();
    header = null;
  };
}

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    wsHost = msg.host || wsHost;
    knownBootId = msg.knownBootId || null;
    knownSessionId = msg.knownSessionId || null;
    knownServerClockId = msg.knownServerClockId || null;
    knownLastAppliedFrame = Number.isFinite(msg.lastAppliedFrame) ? msg.lastAppliedFrame : -1;
    buildId = msg.buildId || buildId;
    connectAnim();
    onTick(msg.elapsed || 0);
    tickCount += 1;
  } else if (msg.type === "tick") {
    onTick(msg.elapsed);
    tickCount += 1;
  } else if (msg.type === "reset") {
    resetTimelineState();
    playbackState = "live_playing";
  }
};
