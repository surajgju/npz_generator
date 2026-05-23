import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { ConversationClient } from "./ConversationClient.js";
import { ensureMicCapture, teardownMicCapture } from "./AudioCapture.js";

let statusEl = null;
let bufferSecEl = null;
let queueLenEl = null;
let inFpsEl = null;
let outFpsEl = null;
let playFpsEl = null;
let streamFpsEl = null;
let pipelineModeEl = null;
let audioStatusEl = null;
let audioBufferEl = null;
let conversationStatusEl = null;
let conversationStateEl = null;
let conversationSessionEl = null;
let playStateEl = null;
let lodLevelEl = null;
let bufferFillEl = null;
let transportAgeEl = null;
let inputWaitEl = null;
let inferMsEl = null;
let resampleMsEl = null;
let retargetMsEl = null;
let outputWaitEl = null;
let flushReasonEl = null;
let fitViewBtn = null;
let viewFaceBtn = null;
let viewFrontBtn = null;
let viewBackBtn = null;
let viewLeftBtn = null;
let viewRightBtn = null;
let viewTopBtn = null;
let viewIsoBtn = null;
let faceOffsetEl = null;
let faceOffsetValEl = null;
let toggleGridEl = null;
let toggleAxesEl = null;
let toggleWireframeEl = null;
let toggleAutoRotateEl = null;
let toggleTranslateEl = null;
let enableAudioBtn = null;
let connectConversationBtn = null;
let pttButton = null;
let disconnectMicButton = null;
let interruptReplyBtn = null;
let togglePlayBtn = null;
let clearBufferBtn = null;
let resetCamBtn = null;
let expVal0El = null;
let expVal1El = null;
let expVal2El = null;
let canvas = null;
let initialized = false;
let boundUi = false;
let resizeHandler = null;
const log = (...args) => console.log("[Viewer]", ...args);
const warn = (...args) => console.warn("[Viewer]", ...args);
const error = (...args) => console.error("[Viewer]", ...args);
const DEBUG = false;
const BUILD_ID = import.meta.env.VITE_BUILD_ID || import.meta.env.MODE || "dev";
const ENABLE_MORPH_DEBUGGER = import.meta.env.VITE_ENABLE_MORPH_DEBUGGER === "1";
const IDLE_START_MS = 2000;
let lastDebugSummary = 0;
let lastHudLog = null;
let lastStatsLog = null;

const idleConfig = {
  blink: { intervalSec: [2.5, 5.0], durationSec: 0.15, strength: 0.6 },
  saccade: { intervalSec: [1.0, 3.0], yawDeg: 2.0, pitchDeg: 1.0 },
  headSway: { yawDeg: 3.0, pitchDeg: 1.5, rollDeg: 0.5, periodSec: [5.0, 7.0] },
  // Discrete neck look-around: random glance target every 2-3s, very slow blend
  neckLook: { intervalSec: [2.0, 3.5], yawDeg: 60.0, pitchDeg: 8.0, blendSpeed: 0.008 },
  neckScale: 0.5,
};
let idleActive = false;
let idleStartSec = 0;
let idleBlinkNextAt = null;
let idleBlinkStartAt = null;
let idleSaccadeNextAt = null;
let idleSaccadeYaw = 0;
let idleSaccadePitch = 0;
let idleSaccadeTargetYaw = 0;
let idleSaccadeTargetPitch = 0;
let idleHeadPeriodSec = 6.0;
let idleHeadPhase = 0;
let idleBlinkPreferredNames = [];
let idleBlinkAutoNames = [];
let idleBlinkTargets = [];
let idleBlinkStrengthScale = 1.0;
// Neck look-around state (discrete glance, separate from sinusoidal sway)
let idleNeckLookNextAt = null;
let idleNeckLookTargetYaw = 0;
let idleNeckLookTargetPitch = 0;
let idleNeckLookCurrentYaw = 0;
let idleNeckLookCurrentPitch = 0;
let lastBlinkNames = "";
let lastBlinkAutoNames = "";
let debugMorphNames = {
  mouthMorphs: [],
  mouthUp: [],
  mouthDown: [],
  mouthSafe: [],
  browMorphs: [],
  browUp: [],
  browDown: [],
};
let debugMouthPreferredNames = [];
let debugMorphTargets = {
  mouthMorphs: [],
  mouthUp: [],
  mouthDown: [],
  mouthSafe: [],
  browMorphs: [],
  browUp: [],
  browDown: [],
  blink: [],
};
let debugManualMode = false;
let manualOverride = {
  active: false,
  type: null,
  startSec: 0,
  durationSec: 0,
  payload: null,
};
const EMOTION_PRESETS = {
  smile: { Exp001: 1.2, Exp002: 0.2 },
  happy: { Exp001: 1.8, Exp002: 0.5, Exp000: -0.2 },
  sad: { Exp001: -1.2, Exp004: 1.2, Exp002: 0.5 },
  anger: { Exp002: -1.8, Exp003: 0.8 },
  nervous: { Exp001: -0.6, Exp002: -0.4, Exp004: 0.6 },
  curious: { Exp002: 1.5, Exp003: -0.5 },
};
let idleBones = {
  head: null,
  neck: null,
  leftEye: null,
  rightEye: null,
  jaw: null,
  leftCollar: null,
  rightCollar: null,
  leftShoulder: null,
  rightShoulder: null,
  leftElbow: null,
  rightElbow: null,
  spine1: null,
  spine2: null,
  spine3: null,
};
let idleRestQuats = {
  head: null,
  neck: null,
  leftEye: null,
  rightEye: null,
  jaw: null,
  leftCollar: null,
  rightCollar: null,
  leftShoulder: null,
  rightShoulder: null,
  leftElbow: null,
  rightElbow: null,
  spine1: null,
  spine2: null,
  spine3: null,
};
let mismatchStartMs = null;

function logDelta(label, curr, prev) {
  const changes = [];
  for (const [key, value] of Object.entries(curr)) {
    const prevVal = prev ? prev[key] : undefined;
    if (prevVal !== value) {
      changes.push(`${key}=${value}`);
    }
  }
  if (changes.length) {
    log(`${label} ${changes.join(" ")}`);
  }
  return curr;
}

function randRange(min, max) {
  return min + Math.random() * (max - min);
}

function findBoneByName(name) {
  const lower = name.toLowerCase();
  for (const [key, bone] of bonesByName.entries()) {
    if (key.toLowerCase() === lower) return bone;
  }
  return null;
}

function findBoneByIncludes(matchers) {
  for (const [key, bone] of bonesByName.entries()) {
    const lower = key.toLowerCase();
    for (const m of matchers) {
      if (lower.includes(m)) return bone;
    }
  }
  return null;
}

let idleRestQuatsCached = false;
function cacheIdleBones() {
  if (!avatarLoaded || idleRestQuatsCached) return;
  idleBones.head = findBoneByName("head") || findBoneByIncludes(["head"]);
  idleBones.neck = findBoneByName("neck") || findBoneByIncludes(["neck"]);
  idleBones.leftEye =
    findBoneByName("left_eye_smplhf") || findBoneByIncludes(["left_eye", "leye"]);
  idleBones.rightEye =
    findBoneByName("right_eye_smplhf") || findBoneByIncludes(["right_eye", "reye"]);
  idleBones.jaw = findBoneByName("jaw") || findBoneByIncludes(["jaw"]);
  idleBones.leftCollar =
    findBoneByName("left_collar") || findBoneByIncludes(["left_collar", "l_collar"]);
  idleBones.rightCollar =
    findBoneByName("right_collar") || findBoneByIncludes(["right_collar", "r_collar"]);
  idleBones.leftShoulder =
    findBoneByName("left_shoulder") || findBoneByIncludes(["left_shoulder", "l_shoulder"]);
  idleBones.rightShoulder =
    findBoneByName("right_shoulder") || findBoneByIncludes(["right_shoulder", "r_shoulder"]);
  idleBones.leftElbow =
    findBoneByName("left_elbow") || findBoneByIncludes(["left_elbow", "l_elbow"]);
  idleBones.rightElbow =
    findBoneByName("right_elbow") || findBoneByIncludes(["right_elbow", "r_elbow"]);
  idleBones.spine1 = findBoneByName("spine1") || findBoneByIncludes(["spine1"]);
  idleBones.spine2 = findBoneByName("spine2") || findBoneByIncludes(["spine2"]);
  idleBones.spine3 = findBoneByName("spine3") || findBoneByIncludes(["spine3"]);
  idleRestQuats.head = idleBones.head ? idleBones.head.quaternion.clone() : null;
  idleRestQuats.neck = idleBones.neck ? idleBones.neck.quaternion.clone() : null;
  idleRestQuats.leftEye = idleBones.leftEye ? idleBones.leftEye.quaternion.clone() : null;
  idleRestQuats.rightEye = idleBones.rightEye ? idleBones.rightEye.quaternion.clone() : null;
  idleRestQuats.jaw = idleBones.jaw ? idleBones.jaw.quaternion.clone() : null;
  idleRestQuats.leftCollar = idleBones.leftCollar
    ? idleBones.leftCollar.quaternion.clone()
    : null;
  idleRestQuats.rightCollar = idleBones.rightCollar
    ? idleBones.rightCollar.quaternion.clone()
    : null;
  idleRestQuats.leftShoulder = idleBones.leftShoulder
    ? idleBones.leftShoulder.quaternion.clone()
    : null;
  idleRestQuats.rightShoulder = idleBones.rightShoulder
    ? idleBones.rightShoulder.quaternion.clone()
    : null;
  idleRestQuats.leftElbow = idleBones.leftElbow ? idleBones.leftElbow.quaternion.clone() : null;
  idleRestQuats.rightElbow = idleBones.rightElbow ? idleBones.rightElbow.quaternion.clone() : null;
  idleRestQuats.spine1 = idleBones.spine1 ? idleBones.spine1.quaternion.clone() : null;
  idleRestQuats.spine2 = idleBones.spine2 ? idleBones.spine2.quaternion.clone() : null;
  idleRestQuats.spine3 = idleBones.spine3 ? idleBones.spine3.quaternion.clone() : null;
  idleRestQuatsCached = true;
}

function refreshIdleBlinkTargets() {
  idleBlinkTargets = [];
  if (!avatarLoaded || morphTargetMap.size === 0) return;
  let names = idleBlinkPreferredNames.filter((n) => morphTargetMap.has(n));
  if (names.length === 0 && idleBlinkAutoNames.length) {
    names = idleBlinkAutoNames.filter((n) => morphTargetMap.has(n));
  }
  names = filterBlinkNames(names);
  if (names.length === 0) {
    const fallback = ["Exp000", "Exp001"].filter((n) => morphTargetMap.has(n));
    if (fallback.length) {
      names = filterBlinkNames(fallback);
    } else {
      names = [];
      for (const name of morphTargetMap.keys()) {
        const lower = name.toLowerCase();
        if (lower.includes("blink") || lower.includes("eye")) {
          names.push(name);
        }
        if (names.length >= 2) break;
      }
      names = filterBlinkNames(names);
    }
  }
  if (DEBUG) {
    const key = names.join(",");
    if (key !== lastBlinkNames) {
      lastBlinkNames = key;
      log("Blink targets:", names, "scale:", idleBlinkStrengthScale);
    }
  }
  for (const name of names) {
    const targets = morphTargetMap.get(name);
    if (!targets) continue;
    for (const t of targets) {
      idleBlinkTargets.push(t);
    }
  }
}

function filterBlinkNames(names) {
  if (!names.length) return names;
  const avoid = new Set([
    ...(debugMorphNames.mouthMorphs || []),
    ...(debugMorphNames.mouthSafe || []),
    ...(debugMorphNames.mouthUp || []),
    ...(debugMorphNames.mouthDown || []),
  ]);
  if (!avoid.size) {
    idleBlinkStrengthScale = 1.0;
    return names;
  }
  const filtered = names.filter((n) => !avoid.has(n));
  if (!filtered.length && names.length) {
    // SMPL-X global morphs will always overlap — use reduced strength silently
    idleBlinkStrengthScale = 0.4;
    return names;
  }
  idleBlinkStrengthScale = 1.0;
  return filtered;
}

function resetAllMorphsToBase() {
  for (const targets of morphTargetMap.values()) {
    for (const t of targets) {
      const base = t.base ?? 0;
      t.mesh.morphTargetInfluences[t.index] = base;
    }
  }
}

function refreshSpeechMorphIndices() {
  mouthMorphIndices = debugMouthPreferredNames
    .map((name) => morphNames.indexOf(name))
    .filter((idx) => idx >= 0);
  mouthMorphIndexSet = new Set(mouthMorphIndices);
}

function resetLivePlaybackFilters() {
  stabilizedRootOffset.set(0, 0, 0);
  rootStabilizerReady = false;
  speechEnergySmoothed = 0;
  speechBodyBlend = 0;
  speechOverlayLastSec = 0;
  idleTailArmed = false;
  idleTailStartMs = null;
  serverSilentSinceMs = null;
}

function stabilizeRootTranslation(rootX, rootY, rootZ, advanceRootSmoothing) {
  if (!rootStabilizerReady) {
    stabilizedRootOffset.set(rootX, rootY, rootZ);
    rootStabilizerReady = true;
    return stabilizedRootOffset;
  }
  if (advanceRootSmoothing) {
    for (const axis of ["x", "z"]) {
      const raw = axis === "x" ? rootX : rootZ;
      const prev = stabilizedRootOffset[axis];
      const delta = raw - prev;
      if (Math.abs(delta) <= ROOT_XZ_DEADBAND_M) {
        continue;
      }
      stabilizedRootOffset[axis] =
        prev +
        Math.max(
          -ROOT_XZ_MAX_STEP_M,
          Math.min(ROOT_XZ_MAX_STEP_M, delta * ROOT_XZ_EMA_ALPHA)
        );
    }
  }
  stabilizedRootOffset.y = rootY;
  return stabilizedRootOffset;
}

function resetIdlePose() {
  if (idleRestQuats.head && idleBones.head) idleBones.head.quaternion.copy(idleRestQuats.head);
  if (idleRestQuats.neck && idleBones.neck) idleBones.neck.quaternion.copy(idleRestQuats.neck);
  if (idleRestQuats.leftEye && idleBones.leftEye)
    idleBones.leftEye.quaternion.copy(idleRestQuats.leftEye);
  if (idleRestQuats.rightEye && idleBones.rightEye)
    idleBones.rightEye.quaternion.copy(idleRestQuats.rightEye);
  if (idleRestQuats.jaw && idleBones.jaw) idleBones.jaw.quaternion.copy(idleRestQuats.jaw);
  if (idleRestQuats.leftCollar && idleBones.leftCollar)
    idleBones.leftCollar.quaternion.copy(idleRestQuats.leftCollar);
  if (idleRestQuats.rightCollar && idleBones.rightCollar)
    idleBones.rightCollar.quaternion.copy(idleRestQuats.rightCollar);
  if (idleRestQuats.leftShoulder && idleBones.leftShoulder)
    idleBones.leftShoulder.quaternion.copy(idleRestQuats.leftShoulder);
  if (idleRestQuats.rightShoulder && idleBones.rightShoulder)
    idleBones.rightShoulder.quaternion.copy(idleRestQuats.rightShoulder);
  if (idleRestQuats.leftElbow && idleBones.leftElbow)
    idleBones.leftElbow.quaternion.copy(idleRestQuats.leftElbow);
  if (idleRestQuats.rightElbow && idleBones.rightElbow)
    idleBones.rightElbow.quaternion.copy(idleRestQuats.rightElbow);
  if (idleRestQuats.spine1 && idleBones.spine1) idleBones.spine1.quaternion.copy(idleRestQuats.spine1);
  if (idleRestQuats.spine2 && idleBones.spine2) idleBones.spine2.quaternion.copy(idleRestQuats.spine2);
  if (idleRestQuats.spine3 && idleBones.spine3) idleBones.spine3.quaternion.copy(idleRestQuats.spine3);
  for (const t of idleBlinkTargets) {
    const base = t.base ?? 0;
    t.mesh.morphTargetInfluences[t.index] = base;
  }
}

function updateIdleBlink(elapsed) {
  const interval = idleConfig.blink.intervalSec;
  const duration = idleConfig.blink.durationSec;
  if (idleBlinkNextAt === null) {
    idleBlinkNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (idleBlinkStartAt === null && elapsed >= idleBlinkNextAt) {
    idleBlinkStartAt = elapsed;
    idleBlinkNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (idleBlinkStartAt !== null) {
    const t = (elapsed - idleBlinkStartAt) / duration;
    if (t >= 1) {
      idleBlinkStartAt = null;
      return 0;
    }
    return t < 0.5 ? t * 2 : (1 - t) * 2;
  }
  return 0;
}

function updateIdleSaccade(elapsed) {
  const interval = idleConfig.saccade.intervalSec;
  if (idleSaccadeNextAt === null) {
    idleSaccadeNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (elapsed >= idleSaccadeNextAt) {
    const yaw = (idleConfig.saccade.yawDeg || 2.0) * (Math.PI / 180);
    const pitch = (idleConfig.saccade.pitchDeg || 1.0) * (Math.PI / 180);
    idleSaccadeTargetYaw = randRange(-yaw, yaw);
    idleSaccadeTargetPitch = randRange(-pitch, pitch);
    idleSaccadeNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  idleSaccadeYaw += (idleSaccadeTargetYaw - idleSaccadeYaw) * 0.15;
  idleSaccadePitch += (idleSaccadeTargetPitch - idleSaccadePitch) * 0.15;
  return { yaw: idleSaccadeYaw, pitch: idleSaccadePitch };
}

function autoDetectBlinkMorphs() {
  idleBlinkAutoNames = [];
  if (!avatarLoaded) return;
  const mesh = morphMeshes.find(
    (m) => m && m.geometry && m.geometry.morphAttributes && m.geometry.morphAttributes.position
  );
  if (!mesh) return;
  const geom = mesh.geometry;
  const morphAttrs = geom.morphAttributes.position || [];
  if (!morphAttrs.length) return;
  if (!idleBones.leftEye || !idleBones.rightEye) {
    cacheIdleBones();
  }
  if (!idleBones.leftEye || !idleBones.rightEye) return;
  const leftWorld = new THREE.Vector3();
  const rightWorld = new THREE.Vector3();
  idleBones.leftEye.getWorldPosition(leftWorld);
  idleBones.rightEye.getWorldPosition(rightWorld);
  const leftLocal = mesh.worldToLocal(leftWorld.clone());
  const rightLocal = mesh.worldToLocal(rightWorld.clone());
  const baseAttr = geom.attributes.position;
  if (!baseAttr) return;
  const baseArray = baseAttr.array;
  const vertCount = baseAttr.count;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < vertCount; i++) {
    const y = baseArray[i * 3 + 1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const height = Math.max(1e-6, maxY - minY);
  const leftMask = new Uint8Array(vertCount);
  const rightMask = new Uint8Array(vertCount);
  const mouthMask = new Uint8Array(vertCount);
  const browMask = new Uint8Array(vertCount);
  const buildEyeMasks = (r, upBand, downBand) => {
    leftMask.fill(0);
    rightMask.fill(0);
    const rSq = r * r;
    for (let i = 0; i < vertCount; i++) {
      const y = baseArray[i * 3 + 1];
      const ix = baseArray[i * 3] - leftLocal.x;
      const iy = y - leftLocal.y;
      const iz = baseArray[i * 3 + 2] - leftLocal.z;
      if (
        ix * ix + iy * iy + iz * iz <= rSq &&
        y >= leftLocal.y - downBand &&
        y <= leftLocal.y + upBand
      ) {
        leftMask[i] = 1;
      }
      const jx = baseArray[i * 3] - rightLocal.x;
      const jy = y - rightLocal.y;
      const jz = baseArray[i * 3 + 2] - rightLocal.z;
      if (
        jx * jx + jy * jy + jz * jz <= rSq &&
        y >= rightLocal.y - downBand &&
        y <= rightLocal.y + upBand
      ) {
        rightMask[i] = 1;
      }
    }
  };
  // Tighter radius — keep close to actual eyelid, away from the eyebrow
  let radius = Math.max(0.02, height * 0.05);
  let bandUp = height * 0.04;   // enough to catch upper eyelid above eye center
  let bandDown = height * 0.03; // lower eyelid
  buildEyeMasks(radius, bandUp, bandDown);
  let mouthCenter = new THREE.Vector3(0, minY + height * 0.25, 0);
  if (!idleBones.jaw) {
    cacheIdleBones();
  }
  if (idleBones.jaw) {
    mouthCenter = mesh.worldToLocal(idleBones.jaw.getWorldPosition(new THREE.Vector3()));
  }
  const mouthRadius = Math.max(0.03, height * 0.08);
  const mouthRadiusSq = mouthRadius * mouthRadius;
  const browCenter = leftLocal.clone().add(rightLocal).multiplyScalar(0.5);
  browCenter.y += height * 0.08;
  const browRadius = Math.max(0.03, height * 0.06);
  const browRadiusSq = browRadius * browRadius;
  for (let i = 0; i < vertCount; i++) {
    const mx = baseArray[i * 3] - mouthCenter.x;
    const my = baseArray[i * 3 + 1] - mouthCenter.y;
    const mz = baseArray[i * 3 + 2] - mouthCenter.z;
    if (mx * mx + my * my + mz * mz <= mouthRadiusSq) mouthMask[i] = 1;
    const bx = baseArray[i * 3] - browCenter.x;
    const by = baseArray[i * 3 + 1] - browCenter.y;
    const bz = baseArray[i * 3 + 2] - browCenter.z;
    if (bx * bx + by * by + bz * bz <= browRadiusSq) browMask[i] = 1;
  }
  let leftCount = 0;
  let rightCount = 0;
  let mouthCount = 0;
  let browCount = 0;
  for (let i = 0; i < vertCount; i++) {
    if (leftMask[i]) leftCount += 1;
    if (rightMask[i]) rightCount += 1;
    if (mouthMask[i]) mouthCount += 1;
    if (browMask[i]) browCount += 1;
  }
  if (leftCount < 15 || rightCount < 15) {
    radius = Math.max(0.03, height * 0.065);
    bandUp = height * 0.05;
    bandDown = height * 0.04;
    buildEyeMasks(radius, bandUp, bandDown);
    leftCount = 0;
    rightCount = 0;
    for (let i = 0; i < vertCount; i++) {
      if (leftMask[i]) leftCount += 1;
      if (rightMask[i]) rightCount += 1;
    }
  }
  if (!leftCount || !rightCount) return;
  const indexToName = [];
  if (mesh.morphTargetDictionary) {
    for (const [name, idx] of Object.entries(mesh.morphTargetDictionary)) {
      indexToName[idx] = name;
    }
  }
  const relative = mesh.morphTargetsRelative || geom.morphTargetsRelative;
  const leftScores = [];
  const rightScores = [];
  for (let m = 0; m < morphAttrs.length; m++) {
    const name = indexToName[m];
    if (!name) continue;
    const attr = morphAttrs[m];
    const arr = attr.array;
    let leftSum = 0;
    let rightSum = 0;
    let mouthSum = 0;
    let browSum = 0;
    for (let i = 0; i < vertCount; i++) {
      if (!leftMask[i] && !rightMask[i] && !mouthMask[i] && !browMask[i]) continue;
      const bx = baseArray[i * 3];
      const by = baseArray[i * 3 + 1];
      const bz = baseArray[i * 3 + 2];
      let dx = arr[i * 3];
      let dy = arr[i * 3 + 1];
      let dz = arr[i * 3 + 2];
      if (!relative) {
        dx -= bx;
        dy -= by;
        dz -= bz;
      }
      const mag = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (leftMask[i]) leftSum += mag;
      if (rightMask[i]) rightSum += mag;
      if (mouthMask[i]) mouthSum += mag;
      if (browMask[i]) browSum += mag;
    }
    const leftMean = leftSum / leftCount;
    const rightMean = rightSum / rightCount;
    const mouthMean = mouthCount ? mouthSum / mouthCount : 0;
    const browMean = browCount ? browSum / browCount : 0;
    leftScores.push([name, leftMean, mouthMean, browMean]);
    rightScores.push([name, rightMean, mouthMean, browMean]);
  }
  // For SMPL-X global morphs, no morph is region-pure. Use relative ratio ranking:
  // score each morph by   eyelid / (eyelid + brow + mouth)  — picks whatever
  // moves the eyelid MOST relative to the rest of the face.
  const addRatio = (arr) => arr.map((v) => {
    const eye = v[1];
    const total = eye + v[2] + v[3];
    return [...v, total > 0 ? eye / total : 0]; // v[4] = ratio
  });
  const leftRatios = addRatio(leftScores).sort((a, b) => b[4] - a[4]);
  const rightRatios = addRatio(rightScores).sort((a, b) => b[4] - a[4]);

  // Pick top 2 by ratio; no hard rejection threshold needed
  const leftTop = leftRatios.slice(0, 2).map((v) => v[0]);
  const rightTop = rightRatios.slice(0, 2).map((v) => v[0]);
  idleBlinkAutoNames = Array.from(new Set([...leftTop, ...rightTop]));

  // Remove selected blink names from mouth/brow lists so filterBlinkNames
  // doesn't flag them as overlapping (SMPL-X morphs are always global).
  const blinkSet = new Set(idleBlinkAutoNames);
  for (const key of ["mouthMorphs", "mouthUp", "mouthDown", "mouthSafe", "browMorphs", "browUp", "browDown"]) {
    debugMorphNames[key] = (debugMorphNames[key] || []).filter((n) => !blinkSet.has(n));
  }

  if (DEBUG) {
    const key = idleBlinkAutoNames.join(",");
    if (key !== lastBlinkAutoNames) {
      lastBlinkAutoNames = key;
      const bestLeft = leftRatios[0] ? `${leftRatios[0][0]}(ratio=${leftRatios[0][4].toFixed(2)})` : "none";
      const bestRight = rightRatios[0] ? `${rightRatios[0][0]}(ratio=${rightRatios[0][4].toFixed(2)})` : "none";
      log("Auto blink morphs:", { leftTop, rightTop, bestLeft, bestRight, selected: idleBlinkAutoNames });
    }
  }
}

function resolveMorphTargets(names) {
  const out = [];
  for (const name of names) {
    const targets = morphTargetMap.get(name);
    if (!targets) continue;
    for (const t of targets) out.push(t);
  }
  return out;
}

function refreshDebugMorphTargets() {
  debugMorphTargets.blink = [...idleBlinkTargets];
  debugMorphTargets.mouthMorphs = resolveMorphTargets(debugMorphNames.mouthMorphs);
  debugMorphTargets.mouthUp = resolveMorphTargets(debugMorphNames.mouthUp);
  debugMorphTargets.mouthDown = resolveMorphTargets(debugMorphNames.mouthDown);
  debugMorphTargets.mouthSafe = resolveMorphTargets(debugMorphNames.mouthSafe);
  debugMorphTargets.browMorphs = resolveMorphTargets(debugMorphNames.browMorphs);
  debugMorphTargets.browUp = resolveMorphTargets(debugMorphNames.browUp);
  debugMorphTargets.browDown = resolveMorphTargets(debugMorphNames.browDown);
}

function autoDetectFaceRegions() {
  debugMorphNames = {
    mouthMorphs: [],
    mouthUp: [],
    mouthDown: [],
    mouthSafe: [],
    browMorphs: [],
    browUp: [],
    browDown: [],
  };
  if (!avatarLoaded) return;
  const mesh = morphMeshes.find(
    (m) => m && m.geometry && m.geometry.morphAttributes && m.geometry.morphAttributes.position
  );
  if (!mesh) {
    warn("Debug detection: no morph mesh with position deltas.");
    // No morph fallback needed when dynamic detection fails
    return;
  }
  if (!idleBones.leftEye || !idleBones.rightEye || !idleBones.jaw) {
    cacheIdleBones();
  }
  if (!idleBones.leftEye || !idleBones.rightEye) {
    warn("Debug detection: missing eye bones.");
  }
  if (!idleBones.jaw) {
    warn("Debug detection: missing jaw bone.");
  }
  const geom = mesh.geometry;
  const morphAttrs = geom.morphAttributes.position || [];
  if (!morphAttrs.length) {
    warn("Debug detection: morphAttributes.position empty.");
    // No morph fallback needed when dynamic detection fails
    return;
  }
  if (!mesh.morphTargetDictionary) {
    warn("Debug detection: morphTargetDictionary missing.");
  }
  const baseAttr = geom.attributes.position;
  if (!baseAttr) return;
  const baseArray = baseAttr.array;
  const vertCount = baseAttr.count;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < vertCount; i++) {
    const y = baseArray[i * 3 + 1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const height = Math.max(1e-6, maxY - minY);
  // Moderate radius: big enough to detect expression morphs, small enough to not swallow blink region
  const mouthRadius = Math.max(0.04, height * 0.13);
  const browRadius = Math.max(0.04, height * 0.12);
  const mouthRadiusSq = mouthRadius * mouthRadius;
  const browRadiusSq = browRadius * browRadius;
  const jawCenter = idleBones.jaw
    ? mesh.worldToLocal(idleBones.jaw.getWorldPosition(new THREE.Vector3()))
    : new THREE.Vector3(0, minY + height * 0.25, 0);
  let browCenter = new THREE.Vector3(0, minY + height * 0.75, 0);
  if (idleBones.leftEye && idleBones.rightEye) {
    const leftWorld = idleBones.leftEye.getWorldPosition(new THREE.Vector3());
    const rightWorld = idleBones.rightEye.getWorldPosition(new THREE.Vector3());
    const leftLocal = mesh.worldToLocal(leftWorld.clone());
    const rightLocal = mesh.worldToLocal(rightWorld.clone());
    browCenter = leftLocal.add(rightLocal).multiplyScalar(0.5);
    browCenter.y += height * 0.06;
  }
  const mouthMask = new Uint8Array(vertCount);
  const browMask = new Uint8Array(vertCount);
  for (let i = 0; i < vertCount; i++) {
    const mx = baseArray[i * 3] - jawCenter.x;
    const my = baseArray[i * 3 + 1] - jawCenter.y;
    const mz = baseArray[i * 3 + 2] - jawCenter.z;
    if (mx * mx + my * my + mz * mz <= mouthRadiusSq) mouthMask[i] = 1;
    const bx = baseArray[i * 3] - browCenter.x;
    const by = baseArray[i * 3 + 1] - browCenter.y;
    const bz = baseArray[i * 3 + 2] - browCenter.z;
    if (bx * bx + by * by + bz * bz <= browRadiusSq) browMask[i] = 1;
  }
  let mouthCount = 0;
  let browCount = 0;
  for (let i = 0; i < vertCount; i++) {
    if (mouthMask[i]) mouthCount += 1;
    if (browMask[i]) browCount += 1;
  }
  if (!mouthCount) warn("Debug detection: mouth region empty.");
  if (!browCount) warn("Debug detection: brow region empty.");
  const indexToName = [];
  if (mesh.morphTargetDictionary) {
    for (const [name, idx] of Object.entries(mesh.morphTargetDictionary)) {
      indexToName[idx] = name;
    }
  }
  const relative = mesh.morphTargetsRelative || geom.morphTargetsRelative;
  const mouthScores = [];
  const browScores = [];
  for (let m = 0; m < morphAttrs.length; m++) {
    const name = indexToName[m];
    if (!name) continue;
    const attr = morphAttrs[m];
    const arr = attr.array;
    let mouthSum = 0;
    let mouthAbs = 0;
    let browSum = 0;
    let browAbs = 0;
    let globalAbs = 0;
    // Also track overall displacement magnitude (not just Y)
    let mouthMag = 0;
    let browMag = 0;
    for (let i = 0; i < vertCount; i++) {
      const bxv = baseArray[i * 3];
      const byv = baseArray[i * 3 + 1];
      const bzv = baseArray[i * 3 + 2];
      let dx = arr[i * 3];
      let dy = arr[i * 3 + 1];
      let dz = arr[i * 3 + 2];
      if (!relative) { dx -= bxv; dy -= byv; dz -= bzv; }
      const mag3 = Math.sqrt(dx * dx + dy * dy + dz * dz);
      globalAbs += Math.abs(dy);
      if (!mouthMask[i] && !browMask[i]) continue;
      if (mouthMask[i]) {
        mouthSum += dy;
        mouthAbs += Math.abs(dy);
        mouthMag += mag3;
      }
      if (browMask[i]) {
        browSum += dy;
        browAbs += Math.abs(dy);
        browMag += mag3;
      }
    }
    if (mouthCount) {
      const mouthMeanAbs = Math.max(mouthAbs, mouthMag * 0.5) / mouthCount;
      const mouthMean = mouthSum / mouthCount;
      const globalMeanAbs = globalAbs / vertCount;
      mouthScores.push([name, mouthMeanAbs, mouthMean, globalMeanAbs]);
    }
    if (browCount) {
      const browMeanAbs = Math.max(browAbs, browMag * 0.5) / browCount;
      browScores.push([name, browMeanAbs, browSum / browCount]);
    }
  }
  mouthScores.sort((a, b) => b[1] - a[1]);
  browScores.sort((a, b) => b[1] - a[1]);
  const mouthTop = mouthScores.slice(0, 8).map((v) => v[0]);
  const browTop = browScores.slice(0, 8).map((v) => v[0]);
  const mouthUp = mouthScores.filter((v) => v[2] > 0).slice(0, 6).map((v) => v[0]);
  const mouthDown = mouthScores.filter((v) => v[2] < 0).slice(0, 6).map((v) => v[0]);
  const mouthSafe = mouthScores
    .filter((v) => v[1] > 0)
    .slice(0, 6)
    .map((v) => v[0]);
  const browUp = browScores.filter((v) => v[2] > 0).slice(0, 6).map((v) => v[0]);
  const browDown = browScores.filter((v) => v[2] < 0).slice(0, 6).map((v) => v[0]);
  debugMorphNames.mouthMorphs = mouthTop;
  debugMorphNames.mouthUp = mouthUp;
  debugMorphNames.mouthDown = mouthDown;
  debugMorphNames.mouthSafe = mouthSafe;
  debugMorphNames.browMorphs = browTop;
  debugMorphNames.browUp = browUp;
  debugMorphNames.browDown = browDown;
  if (debugMouthPreferredNames.length) {
    debugMorphNames.mouthMorphs = debugMouthPreferredNames.slice();
    debugMorphNames.mouthSafe = debugMouthPreferredNames.slice();
  }
  // If geometric detection still yielded nothing for mouth/brow, use index-based fallback
  const mouthEmpty = !mouthUp.length && !mouthDown.length;
  const browEmpty = !browUp.length && !browDown.length;
  if (mouthEmpty || browEmpty) {
    if (DEBUG) warn("Debug detection: geometric detection found no mouth/brow morphs, applying fallback.");
    // Dynamic detection only; removal of fallback paths
  } else {
    if (DEBUG) log("Debug detection: mouth/brow morphs detected.", { mouthUp, mouthDown, browUp, browDown });
  }
}


function updateIdleNeckLook(elapsed) {
  const cfg = idleConfig.neckLook;
  const interval = cfg.intervalSec || [2.0, 3.5];
  if (idleNeckLookNextAt === null) {
    idleNeckLookNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  if (elapsed >= idleNeckLookNextAt) {
    const yawRad = (cfg.yawDeg || 12.0) * (Math.PI / 180);
    const pitchRad = (cfg.pitchDeg || 5.0) * (Math.PI / 180);
    idleNeckLookTargetYaw = randRange(-yawRad, yawRad);
    idleNeckLookTargetPitch = randRange(-pitchRad, pitchRad);
    idleNeckLookNextAt = elapsed + randRange(interval[0], interval[1]);
  }
  const speed = cfg.blendSpeed || 0.06;
  idleNeckLookCurrentYaw += (idleNeckLookTargetYaw - idleNeckLookCurrentYaw) * speed;
  idleNeckLookCurrentPitch += (idleNeckLookTargetPitch - idleNeckLookCurrentPitch) * speed;
  return { yaw: idleNeckLookCurrentYaw, pitch: idleNeckLookCurrentPitch };
}

function applyIdlePose(nowSec) {
  if (!idleActive || !avatarLoaded) return;
  if (!idleRestQuats.head && idleBones.head) {
    cacheIdleBones();
  }
  const elapsed = nowSec - idleStartSec;
  const headCfg = idleConfig.headSway;
  const basePeriod = idleHeadPeriodSec || 6.0;
  const swayYawAmp = (headCfg.yawDeg || 3.0) * (Math.PI / 180);
  const swayPitchAmp = (headCfg.pitchDeg || 1.5) * (Math.PI / 180);
  const rollAmp = (headCfg.rollDeg || 0.5) * (Math.PI / 180);
  const swayYaw = Math.sin((elapsed / basePeriod) * Math.PI * 2 + idleHeadPhase) * swayYawAmp;
  const swayPitch =
    Math.sin((elapsed / (basePeriod * 1.3)) * Math.PI * 2 + idleHeadPhase * 0.7) * swayPitchAmp;
  const roll =
    Math.sin((elapsed / (basePeriod * 1.7)) * Math.PI * 2 + idleHeadPhase * 1.3) * rollAmp;

  // Discrete look-around: head glances every 2-3s at human-like speed
  const { yaw: lookYaw, pitch: lookPitch } = updateIdleNeckLook(elapsed);
  const totalYaw = swayYaw + lookYaw;
  const totalPitch = swayPitch + lookPitch;

  if (idleBones.head && idleRestQuats.head) {
    tempEuler.set(totalPitch, totalYaw, roll, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.head.quaternion.copy(idleRestQuats.head).multiply(tempQuat2);
  }
  if (idleBones.neck && idleRestQuats.neck) {
    const scale = idleConfig.neckScale ?? 0.5;
    tempEuler.set(totalPitch * scale, totalYaw * scale, roll * scale, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.neck.quaternion.copy(idleRestQuats.neck).multiply(tempQuat2);
  }
  // Subtle spine sway following the head.
  const spineYaw = swayYaw * 0.5;
  const spinePitch = swayPitch * 0.4;
  const spineBones = [
    [idleBones.spine1, idleRestQuats.spine1, 0.4],
    [idleBones.spine2, idleRestQuats.spine2, 0.35],
    [idleBones.spine3, idleRestQuats.spine3, 0.25],
  ];
  for (const [bone, rest, scale] of spineBones) {
    if (!bone || !rest) continue;
    tempEuler.set(spinePitch * scale, spineYaw * scale, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    bone.quaternion.copy(rest).multiply(tempQuat2);
  }
  // ── Arm rest pose ──
  // In SMPL-X GLB the collar (clavicle) bone's LOCAL space:
  //   Z-axis = elevation (negative Z = depress/lower the shoulder girdle)
  // The shoulder (upper arm) LOCAL space:
  //   Z-axis = abduction/adduction (negative Z = adduct = arm toward body)
  //   X-axis = flexion/extension (negative X = forward lean)
  // Drive collars to depress the shoulder girdle, then adduct via shoulder Z.
  const collarDepress = (-20 * Math.PI) / 180; // depress clavicle slightly
  if (idleBones.leftCollar && idleRestQuats.leftCollar) {
    tempEuler.set(0, 0, collarDepress, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftCollar.quaternion.copy(idleRestQuats.leftCollar).multiply(tempQuat2);
  }
  if (idleBones.rightCollar && idleRestQuats.rightCollar) {
    tempEuler.set(0, 0, collarDepress, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.rightCollar.quaternion.copy(idleRestQuats.rightCollar).multiply(tempQuat2);
  }
  // Shoulder: adduct (bring arm in from T-pose) via Z, plus slight forward X lean.
  const shoulderAdductLeft  = ( 55 * Math.PI) / 180; // left shoulder adduct
  const shoulderAdductRight = (-55 * Math.PI) / 180; // right shoulder adduct
  const shoulderForward     = (-10 * Math.PI) / 180; // slight forward lean
  if (idleBones.leftShoulder && idleRestQuats.leftShoulder) {
    tempEuler.set(shoulderForward, 0, shoulderAdductLeft, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftShoulder.quaternion.copy(idleRestQuats.leftShoulder).multiply(tempQuat2);
  }
  if (idleBones.rightShoulder && idleRestQuats.rightShoulder) {
    tempEuler.set(shoulderForward, 0, shoulderAdductRight, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.rightShoulder.quaternion.copy(idleRestQuats.rightShoulder).multiply(tempQuat2);
  }
  const elbowBend = (-10 * Math.PI) / 180;
  if (idleBones.leftElbow && idleRestQuats.leftElbow) {
    tempEuler.set(elbowBend, 0, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftElbow.quaternion.copy(idleRestQuats.leftElbow).multiply(tempQuat2);
  }
  if (idleBones.rightElbow && idleRestQuats.rightElbow) {
    tempEuler.set(elbowBend, 0, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.rightElbow.quaternion.copy(idleRestQuats.rightElbow).multiply(tempQuat2);
  }
  // ── Eye saccade + bone-driven blink ──
  // Blink: rotate the eye bones downward on X (local "look down" closes
  // the upper eyelid in SMPL-X). This avoids touching Exp morphs which
  // are face-global and affect the nasolabial fold.
  const blinkVal = updateIdleBlink(elapsed) * (idleConfig.blink.strength || 0.6);
  // Max downward rotation for a full blink: ~30°
  const blinkMaxRad = (30 * Math.PI) / 180;
  const blinkRot = blinkVal * blinkMaxRad;
  if (idleBones.leftEye && idleBones.rightEye && idleRestQuats.leftEye && idleRestQuats.rightEye) {
    const { yaw: eyeYaw, pitch: eyePitch } = updateIdleSaccade(elapsed);
    // Combine saccade + blink on X axis (blink = additional downward pitch)
    tempEuler.set(eyePitch + blinkRot, eyeYaw, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftEye.quaternion.copy(idleRestQuats.leftEye).multiply(tempQuat2);
    idleBones.rightEye.quaternion.copy(idleRestQuats.rightEye).multiply(tempQuat2);
  } else {
    // Fallback: eyes not found, skip
  }
}

function beginLagIdleOverlay(nowSec) {
  idleStartSec = nowSec;
  idleBlinkNextAt = null;
  idleBlinkStartAt = null;
  idleSaccadeNextAt = null;
  idleSaccadeYaw = 0;
  idleSaccadePitch = 0;
  idleSaccadeTargetYaw = 0;
  idleSaccadeTargetPitch = 0;
  idleNeckLookNextAt = null;
  idleNeckLookTargetYaw = 0;
  idleNeckLookTargetPitch = 0;
  idleNeckLookCurrentYaw = 0;
  idleNeckLookCurrentPitch = 0;
  idleHeadPeriodSec = randRange(idleConfig.headSway.periodSec[0], idleConfig.headSway.periodSec[1]);
  idleHeadPhase = randRange(0, Math.PI * 2);
  cacheIdleBones();
  autoDetectFaceRegions();
  autoDetectBlinkMorphs();
  refreshIdleBlinkTargets();
}

function slerpBoneTowardIdle(bone, restQuat, pitch, yaw, roll, blend) {
  if (!bone || !restQuat) return;
  tempEuler.set(pitch, yaw, roll, "YXZ");
  tempQuat2.setFromEuler(tempEuler);
  tempQuat.copy(restQuat).multiply(tempQuat2);
  bone.quaternion.slerp(tempQuat, blend);
}

function applyIdlePoseAdditive(nowSec, blend) {
  if (!avatarLoaded || blend <= 0) return;
  if (!idleRestQuats.head && idleBones.head) {
    cacheIdleBones();
  }
  const elapsed = nowSec - idleStartSec;
  const headCfg = idleConfig.headSway;
  const basePeriod = idleHeadPeriodSec || 6.0;
  const yawAmp = (headCfg.yawDeg || 2.0) * (Math.PI / 180);
  const pitchAmp = (headCfg.pitchDeg || 1.0) * (Math.PI / 180);
  const rollAmp = (headCfg.rollDeg || 0.5) * (Math.PI / 180);
  const yaw = Math.sin((elapsed / basePeriod) * Math.PI * 2 + idleHeadPhase) * yawAmp;
  const pitch =
    Math.sin((elapsed / (basePeriod * 1.3)) * Math.PI * 2 + idleHeadPhase * 0.7) * pitchAmp;
  const roll =
    Math.sin((elapsed / (basePeriod * 1.7)) * Math.PI * 2 + idleHeadPhase * 1.3) * rollAmp;

  slerpBoneTowardIdle(idleBones.head, idleRestQuats.head, pitch, yaw, roll, blend);

  const neckScale = idleConfig.neckScale ?? 0.5;
  slerpBoneTowardIdle(
    idleBones.neck,
    idleRestQuats.neck,
    pitch * neckScale,
    yaw * neckScale,
    roll * neckScale,
    blend
  );

  const spineYaw = yaw * 0.5;
  const spinePitch = pitch * 0.4;
  slerpBoneTowardIdle(idleBones.spine1, idleRestQuats.spine1, spinePitch * 0.4, spineYaw * 0.4, 0, blend);
  slerpBoneTowardIdle(idleBones.spine2, idleRestQuats.spine2, spinePitch * 0.35, spineYaw * 0.35, 0, blend);
  slerpBoneTowardIdle(idleBones.spine3, idleRestQuats.spine3, spinePitch * 0.25, spineYaw * 0.25, 0, blend);

  // NOTE: shoulders excluded — server drives them during live playback.
  const elbowBend = (-5 * Math.PI) / 180;
  slerpBoneTowardIdle(idleBones.leftElbow, idleRestQuats.leftElbow, elbowBend, 0, 0, blend);
  slerpBoneTowardIdle(idleBones.rightElbow, idleRestQuats.rightElbow, elbowBend, 0, 0, blend);

  if (idleBones.leftEye && idleBones.rightEye && idleRestQuats.leftEye && idleRestQuats.rightEye) {
    const { yaw: eyeYaw, pitch: eyePitch } = updateIdleSaccade(elapsed);
    slerpBoneTowardIdle(idleBones.leftEye, idleRestQuats.leftEye, eyePitch, eyeYaw, 0, blend);
    slerpBoneTowardIdle(idleBones.rightEye, idleRestQuats.rightEye, eyePitch, eyeYaw, 0, blend);
  }

  if (idleBlinkTargets.length) {
    const blinkVal = updateIdleBlink(elapsed) * (idleConfig.blink.strength || 0.6) * idleBlinkStrengthScale;
    for (const t of idleBlinkTargets) {
      const base = t.base ?? 0;
      const current = t.mesh.morphTargetInfluences[t.index] ?? base;
      const target = Math.max(-1, Math.min(1, base + blinkVal));
      t.mesh.morphTargetInfluences[t.index] = current + (target - current) * blend;
    }
  }
}

function updateIdleState() {
  const now = performance.now();
  if (manualOverride.active) {
    if (idleActive) {
      idleActive = false;
      resetIdlePose();
    }
    return;
  }
  // Idle fires when NO fresh frames have arrived from the worker for IDLE_START_MS.
  // lastWorkerFrameAt is updated as soon as a frame arrives from the server.
  // This avoids falsely triggering idle during audio buffering when frame rendering is paused.
  const noFrames = lastWorkerFrameAt === null || now - lastWorkerFrameAt > IDLE_START_MS;
  const disconnected =
    !workerReady || statusEl.textContent === "Disconnected" || pipelineModeEl.textContent === "Error";
  const shouldIdle = !manualPaused && (noFrames || disconnected);
  if (shouldIdle === idleActive) return;
  idleActive = shouldIdle;
  if (idleActive) {
    idleStartSec = now / 1000;
    idleBlinkNextAt = null;
    idleBlinkStartAt = null;
    idleSaccadeNextAt = null;
    idleSaccadeYaw = 0;
    idleSaccadePitch = 0;
    idleSaccadeTargetYaw = 0;
    idleSaccadeTargetPitch = 0;
    idleNeckLookNextAt = null;
    idleNeckLookTargetYaw = 0;
    idleNeckLookTargetPitch = 0;
    idleNeckLookCurrentYaw = 0;
    idleNeckLookCurrentPitch = 0;
    idleHeadPeriodSec = randRange(
      idleConfig.headSway.periodSec[0],
      idleConfig.headSway.periodSec[1]
    );
    idleHeadPhase = randRange(0, Math.PI * 2);
    cacheIdleBones();
    autoDetectFaceRegions();
    autoDetectBlinkMorphs();
    refreshIdleBlinkTargets();
    resetAllMorphsToBase();
  } else {
    resetIdlePose();
  }
}

function easeInOut(t) {
  return 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, t)));
}

function updateSpeechBodyBlend(nowSec, targetBlend) {
  const target = Math.max(0, Math.min(1, targetBlend));
  if (!speechOverlayLastSec) {
    speechOverlayLastSec = nowSec;
    speechBodyBlend = target;
    return speechBodyBlend;
  }
  const dt = Math.max(0, Math.min(0.1, nowSec - speechOverlayLastSec));
  speechOverlayLastSec = nowSec;
  const duration = target > speechBodyBlend ? SPEECH_BODY_FADE_IN_SEC : SPEECH_BODY_FADE_OUT_SEC;
  const step = duration > 0 ? Math.min(1, dt / duration) : 1;
  speechBodyBlend += (target - speechBodyBlend) * step;
  return speechBodyBlend;
}

function applySpeechBodyOverlay(nowSec, targetEnergy, energyScale = 1) {
  if (!avatarLoaded) return;
  cacheIdleBones();
  const blend = updateSpeechBodyBlend(nowSec, targetEnergy) * Math.max(0, energyScale);
  if (blend <= 0.001) return;
  const phase = (nowSec / SPEECH_BODY_SWAY_PERIOD_SEC) * Math.PI * 2;
  const pulse = 0.88 + 0.12 * Math.sin(phase);
  const sway = Math.sin(phase * 0.6) * 0.5;

  slerpBoneTowardIdle(
    idleBones.spine1,
    idleRestQuats.spine1,
    SPEECH_SPINE_MAX_PITCH,
    0,
    0,
    blend * pulse
  );
  slerpBoneTowardIdle(
    idleBones.spine2,
    idleRestQuats.spine2,
    SPEECH_SPINE_MAX_PITCH * 0.8,
    0,
    0,
    blend * pulse
  );
  slerpBoneTowardIdle(
    idleBones.spine3,
    idleRestQuats.spine3,
    SPEECH_SPINE_MAX_PITCH * 0.6,
    0,
    0,
    blend * pulse
  );
  slerpBoneTowardIdle(
    idleBones.neck,
    idleRestQuats.neck,
    SPEECH_NECK_MAX_PITCH * pulse,
    SPEECH_NECK_MAX_YAW * sway,
    0,
    blend
  );
  // NOTE: shoulders are intentionally excluded here — their animation
  // comes from the server (SMPL-X body_pose joints 13/14 = left/right collar,
  // 16/17 = left/right shoulder). Overriding them during live playback
  // would jam the server-driven shoulder gestures.
}

function applyUtteranceTail(nowMs) {
  if (!idleTailArmed) return 1;
  if (idleTailStartMs === null) {
    idleTailStartMs = nowMs;
  }
  const progress = Math.max(0, Math.min(1, (nowMs - idleTailStartMs) / SPEECH_BODY_TAIL_MS));
  applyStallEase(progress);
  cacheIdleBones();
  slerpBoneTowardIdle(idleBones.jaw, idleRestQuats.jaw, 0, 0, 0, progress);
  if (progress >= 1) {
    idleTailArmed = false;
    idleTailStartMs = null;
  }
  return 1 - progress;
}

function applyMorphTargets(targets, value) {
  for (const t of targets) {
    const base = t.base ?? 0;
    t.mesh.morphTargetInfluences[t.index] = base + value;
  }
}

function applyManualOverride(nowSec, overlayOnly = false) {
  if (!manualOverride.active) return false;
  const elapsed = nowSec - manualOverride.startSec;
  const t = manualOverride.durationSec > 0 ? elapsed / manualOverride.durationSec : 1;
  if (t >= 1) {
    manualOverride.active = false;
    if (!overlayOnly) {
      resetAllMorphsToBase();
      resetIdlePose();
    }
    return false;
  }
  if (!avatarLoaded) return true;
  if (!overlayOnly) resetAllMorphsToBase();
  const tri = t < 0.5 ? t * 2 : (1 - t) * 2;
  const ease = easeInOut(t);
  if (manualOverride.type === "blink") {
    let blinkApplied = false;
    if (debugMorphTargets.blink && debugMorphTargets.blink.length) {
      applyMorphTargets(debugMorphTargets.blink, tri * 0.8 * idleBlinkStrengthScale);
      blinkApplied = true;
    }
    if (!blinkApplied) {
      // NOTE: Standard SMPL-X Exp000-Exp099 do NOT contain pure eyelid blendshapes.
      // We simulate a blink-squint using Exp002 (brow lower / squint) as a fallback
      // so the button does *something* visibly.
      const fakeBlink = morphTargetMap.get("Exp002");
      if (fakeBlink && fakeBlink.length) {
        applyMorphTargets(fakeBlink, tri * -1.5);
      }
    }
    return true;
  }
  if (manualOverride.type === "mouth") {
    // Only use Exp000 (jaw open PCA) to avoid breaking local bone rotations 
    const exp0 = morphTargetMap.get("Exp000");
    if (exp0 && exp0.length) applyMorphTargets(exp0, tri * 2.0); // Amplify since it's subtle natively
    return true;
  }
  if (manualOverride.type === "lip") {
    // Subtle lip action via Exp003 and Exp004
    const exp3 = morphTargetMap.get("Exp003");
    const exp4 = morphTargetMap.get("Exp004");
    if (exp3 && exp3.length) applyMorphTargets(exp3, tri * 1.5);
    if (exp4 && exp4.length) applyMorphTargets(exp4, tri * 0.8);
    return true;
  }

  if (manualOverride.type === "emotion") {
    const payload = manualOverride.payload || {};

    // Legacy fallback parameters (if payload uses generic groups)
    const mouthUp = debugMorphTargets.mouthUp || [];
    const mouthDown = debugMorphTargets.mouthDown || [];
    const browUp = debugMorphTargets.browUp || [];
    const browDown = debugMorphTargets.browDown || [];
    const mUp = (payload.mouthUp || 0) * ease;
    const mDown = (payload.mouthDown || 0) * ease;
    const bUp = (payload.browUp || 0) * ease;
    const bDown = (payload.browDown || 0) * ease;
    let appliedAny = false;
    if (mUp && mouthUp.length) { applyMorphTargets(mouthUp, mUp); appliedAny = true; }
    if (mDown && mouthDown.length) { applyMorphTargets(mouthDown, mDown); appliedAny = true; }
    if (bUp && browUp.length) { applyMorphTargets(browUp, bUp); appliedAny = true; }
    if (bDown && browDown.length) { applyMorphTargets(browDown, bDown); appliedAny = true; }

    // Explicit FLAME / Blendshape explicit target names
    for (const [key, val] of Object.entries(payload)) {
      if (key.startsWith("Exp") || key.startsWith("Shape") || key.startsWith("Mouth")) {
        const targets = morphTargetMap.get(key);
        if (targets && targets.length) {
          applyMorphTargets(targets, val * ease);
          appliedAny = true;
        }
      }
    }

    // Fallback if appliedAny is false: just do a slight lip/jaw blendshape fallback
    if (!appliedAny) {
      const exp0 = morphTargetMap.get("Exp000");
      if (exp0 && exp0.length) applyMorphTargets(exp0, Math.max(mUp, mDown) * tri);
    }
    return true;
  }
  return true;
}

let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let ambient = null;
let dir = null;
const AUTO_CENTER = true;
const AUTO_SCALE = true;
const TARGET_HEIGHT = 1.7;

function initThree(canvasEl) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  new THREE.TextureLoader().load("./assets/wallpaper.webp", (texture) => {
    scene.background = texture;
  });

  camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.01, 50);
  camera.position.set(1.2, 1.4, 2.5);

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 1.0, 0);
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.6;
  controls.update();

  ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);
  dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(2, 3, 2);
  scene.add(dir);
}

let avatarRoot = null;
let skinnedMeshes = [];
let morphMeshes = [];
let bonesByName = new Map();
let boneOrder = [];
let boneRestQuats = [];
let boneRestPos = [];
let rootBone = null;
let rootRestPos = new THREE.Vector3();
let stabilizedRootOffset = new THREE.Vector3();
let rootStabilizerReady = false;
let morphTargetMap = new Map();
let morphAliasCount = 0;
let loggedAliasSummary = false;
let morphNames = [];
let mouthMorphIndices = [];
let mouthMorphIndexSet = new Set();
let expMeshCount = 0;
let avatarLoaded = false;
let frameCount = 0;
let cameraFitted = false;
let loggedMorphInfluence = false;
let lastAppliedMorphSamples = null;
let lastAppliedAt = null;
let lastWorkerFrameIndexSeen = null;
let lastWorkerFrameAt = null;
let lastWorkerFrameBytes = null;

function exposeViewerDebug() {
  if (typeof window === "undefined") return;
  const viewer = window.__viewer || {};
  viewer.scene = scene;
  viewer.camera = camera;
  viewer.renderer = renderer;
  viewer.avatarRoot = avatarRoot;
  viewer.morphMeshes = morphMeshes;
  viewer.morphTargetMap = morphTargetMap;
  viewer.morphNames = morphNames;
  viewer.dumpMorphStatus = () => {
    const missingMorphs = morphNames.filter((name) => !morphTargetMap.has(name));
    return {
      morphTargetMapSize: morphTargetMap.size,
      morphNamesCount: morphNames.length,
      missingMorphsCount: missingMorphs.length,
      missingMorphsSample: missingMorphs.slice(0, 8),
      expMeshes: expMeshCount,
      lastAppliedMorphSamples,
      lastAppliedAt,
      lastWorkerFrameIndexSeen,
      lastWorkerFrameAt,
      lastWorkerFrameBytes,
      workerReady,
      workerAlive,
      workerQueueLen,
      workerInFps,
      workerFrameIndex,
      frameDirty,
      streamFps,
      playState: playStateEl?.textContent || "",
      status: statusEl?.textContent || "",
      pipeline: pipelineModeEl?.textContent || "",
    };
  };
  viewer.sampleMorph = (name) => {
    const targets = morphTargetMap.get(name);
    if (!targets || !targets.length) return null;
    const t = targets[0];
    return {
      name,
      mesh: t.mesh?.name || "",
      index: t.index,
      value: t.mesh?.morphTargetInfluences?.[t.index] ?? null,
    };
  };
  window.__viewer = viewer;
}

const NORMAL_EVERY = 15;
let streamFps = null;
let streamFpsReady = false;
let maxBufferSeconds = 10;

let currentFrameIndex = 0;
let manualPaused = false;
let playCount = 0;

let gridHelper = null;
let axesHelper = null;
let lastBounds = null;
let faceOffset = 0.0;
let userOffset = new THREE.Vector3();
let basePos = new THREE.Vector3();
let transformControls = null;
let gizmoAnchor = null;
let isDragging = false;
let lastRateTime = performance.now();
let currentPlayFps = 0;

let audioCtx = null;
let audioEnabled = false;
let audioStarted = false;
let audioStartTime = null;
let audioScheduledTime = 0;
let audioQueue = [];
let audioQueuedSec = 0;
let conversationClient = null;
let conversationConnected = false;
let conversationId = null;
let conversationStreamSessionId = null;
let assistantLifecycleState = "idle";
let pttActive = false;
let pttPressed = false;
let pttStarting = false;
let pttSeq = 0;
let latestUserTranscript = "";
let latestAssistantResponseText = "";
let recognition = null;
let currentSpeechTranscript = "";
let silentStartTime = null;
let animOffsetSec = 0;
let animOffsetSet = false;
const AUDIO_JITTER_SEC = 0.08;
const AUDIO_START_LEAD_SEC = Number(import.meta.env.VITE_AUDIO_START_LEAD_SEC || 0.03);
const AUDIO_START_BUFFER_SEC = Number(import.meta.env.VITE_AUDIO_START_BUFFER_SEC || 0.4);
const MESH_START_BUFFER_SEC = Number(import.meta.env.VITE_MESH_START_BUFFER_SEC || 0.25);
const AUDIO_LOW_BUFFER_SEC = Number(import.meta.env.VITE_AUDIO_LOW_BUFFER_SEC || 0.25);
const SESSION_MISMATCH_GRACE_MS = Number(import.meta.env.VITE_SESSION_MISMATCH_GRACE_MS || 3000);
const STARTUP_SUPPRESS_MS = 4000;
const STALL_HOLD_MS = 300;
const STALL_EASE_MS = 700;
const STALL_IDLE_BLEND_MS = Number(import.meta.env.VITE_STALL_IDLE_BLEND_MS || 2200);
const workerEnabled = typeof Worker !== "undefined";
const WS_HOST =
  (import.meta && import.meta.env && import.meta.env.VITE_WS_HOST) ||
  ((import.meta && import.meta.env && import.meta.env.DEV)
    ? `${window.location.hostname}:8000`
    : window.location.host);
let workerAlive = false;
let worker = null;
let workerFrame = null;
let workerBounds = null;
let workerFrameIndex = -1;
let workerQueueLen = 0;
let workerSnapshotDropped = 0;
let workerLiveDropped = 0;
let workerResyncSkipped = 0;
let workerInFps = 0;
let workerPlaybackFps = 0;
let workerTiming = null;
let frameDirty = false;
let heldFrameData = null;
let lagIdleBlend = 0;
let lagIdleStarted = false;
// Tracks when workerInFps last dropped to 0 (server stopped sending new frames)
let serverSilentSinceMs = null;
let workerReady = false;
let workerFallbackTimer = null;
let workerRestartTimer = null;
let workerRestartPending = false;
let fallbackTickElapsedSec = 0;
let fallbackTickLastMs = 0;
let lastFrameUpdate = 0;
let workerMaxIndex = -1;
let workerMinIndex = -1;
let driftStart = null;
const DRIFT_THRESHOLD_FRAMES = 20;
const RESYNC_GRACE_MS = 3000;
let pendingInit = null;
let loggedFirstFrame = false;
let smoothedMorphs = null;
let startupSuppressUntilMs = 0;
let stallSinceMs = null;
let workerSessionId = null;
let audioSessionId = null;
let serverBootId = null;
let serverClockId = null;
let protocolVersion = 1;
let resyncing = false;
let buildLogged = false;
let sessionMismatchWarned = false;
let lastSessionResetAt = 0;
let speechEnergySmoothed = 0;
let speechBodyBlend = 0;
let speechOverlayLastSec = 0;
let idleTailArmed = false;
let idleTailStartMs = null;

const MORPH_EMA_ALPHA = 0.35;
const MORPH_CLAMP = Number(import.meta.env.VITE_EXPRESSION_MAX_ABS || 0.85);
const ROOT_XZ_DEADBAND_M = Number(import.meta.env.VITE_ROOT_XZ_DEADBAND_M || 0.005);
const ROOT_XZ_EMA_ALPHA = Number(import.meta.env.VITE_ROOT_XZ_EMA_ALPHA || 0.15);
const ROOT_XZ_MAX_STEP_M = Number(import.meta.env.VITE_ROOT_XZ_MAX_STEP_M || 0.015);
const SPEECH_BODY_FADE_IN_SEC = 0.12;
const SPEECH_BODY_FADE_OUT_SEC = 0.18;
const SPEECH_BODY_TAIL_MS = 180;
const SPEECH_SPINE_MAX_PITCH = (1.5 * Math.PI) / 180;
const SPEECH_NECK_MAX_PITCH = (0.8 * Math.PI) / 180;
const SPEECH_NECK_MAX_YAW = (0.8 * Math.PI) / 180;
const SPEECH_SHOULDER_MAX_PITCH = (-1.0 * Math.PI) / 180;
const SPEECH_BODY_SWAY_PERIOD_SEC = 2.4;

const gltfLoader = new GLTFLoader();
gltfLoader.setMeshoptDecoder(MeshoptDecoder);
const tempQuat = new THREE.Quaternion();
const tempQuat2 = new THREE.Quaternion();
const tempEuler = new THREE.Euler();
const tempVec = new THREE.Vector3();

function buildMorphTargetMap(meshes) {
  morphTargetMap = new Map();
  morphAliasCount = 0;
  for (const mesh of meshes) {
    if (!mesh.morphTargetInfluences) continue;
    let dict = mesh.morphTargetDictionary;
    const hasNamedTargets =
      dict && Object.keys(dict).some((k) => k.startsWith("Exp") || k.startsWith("Shape"));
    if (!hasNamedTargets) {
      const targetNames =
        (mesh.userData && Array.isArray(mesh.userData.targetNames) && mesh.userData.targetNames) ||
        (mesh.geometry &&
          mesh.geometry.userData &&
          Array.isArray(mesh.geometry.userData.targetNames) &&
          mesh.geometry.userData.targetNames) ||
        null;
      if (targetNames && targetNames.length === mesh.morphTargetInfluences.length) {
        dict = {};
        for (let i = 0; i < targetNames.length; i++) {
          dict[targetNames[i]] = i;
        }
        mesh.morphTargetDictionary = dict;
      }
    }
    if (!dict || Object.keys(dict).length === 0) continue;
    // Ensure materials support morph targets.
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of materials) {
      if (!mat) continue;
      if (mat.morphTargets !== true) {
        mat.morphTargets = true;
        mat.needsUpdate = true;
      }
    }
    // Some loaders require this to initialize morphTargetInfluences.
    if (typeof mesh.updateMorphTargets === "function") {
      mesh.updateMorphTargets();
    }
    for (const [name, index] of Object.entries(dict)) {
      if (!morphTargetMap.has(name)) morphTargetMap.set(name, []);
      const base = Array.isArray(mesh.morphTargetInfluences)
        ? (mesh.morphTargetInfluences[index] ?? 0)
        : 0;
      morphTargetMap.get(name).push({ mesh, index, base });
    }
  }
  // Build Exp↔Shape aliases so server Exp### drives Shape### morphs (and vice versa).
  const entries = Array.from(morphTargetMap.entries());
  for (const [name, targets] of entries) {
    const shapeMatch = /^Shape(\d+)$/i.exec(name);
    const expMatch = /^Exp(\d+)$/i.exec(name);
    if (shapeMatch) {
      const idx = shapeMatch[1].padStart(3, "0");
      const alias = `Exp${idx}`;
      if (!morphTargetMap.has(alias)) {
        morphTargetMap.set(alias, targets);
        morphAliasCount += 1;
      }
    } else if (expMatch) {
      const idx = expMatch[1].padStart(3, "0");
      const alias = `Shape${idx}`;
      if (!morphTargetMap.has(alias)) {
        morphTargetMap.set(alias, targets);
        morphAliasCount += 1;
      }
    }
  }
}

async function loadAvatar() {
  idleRestQuatsCached = false;
  try {
    const gltf = await gltfLoader.loadAsync("./assets/head.glb", (xhr) => {
      if (xhr.loaded === xhr.total) {
        console.log(`head.glb file size: ${xhr.total} bytes`);
      }
    });
    avatarRoot = gltf.scene;
    scene.add(avatarRoot);
    skinnedMeshes = [];
    morphMeshes = [];
    bonesByName = new Map();
    avatarRoot.traverse((obj) => {
      if (obj.isSkinnedMesh) {
        skinnedMeshes.push(obj);
      }
      if (obj.isMesh) {
        morphMeshes.push(obj);
      }
    });
    for (const mesh of skinnedMeshes) {
      if (mesh.skeleton && mesh.skeleton.bones) {
        for (const bone of mesh.skeleton.bones) {
          if (bone.name && !bonesByName.has(bone.name)) {
            bonesByName.set(bone.name, bone);
          }
        }
      }
    }
    buildMorphTargetMap(morphMeshes);
    exposeViewerDebug();
    let meshMorphCount = 0;
    let meshMorphTargets = 0;
    for (const mesh of morphMeshes) {
      const mt = mesh.morphTargetInfluences ? mesh.morphTargetInfluences.length : 0;
      const md = mesh.morphTargetDictionary ? Object.keys(mesh.morphTargetDictionary).length : 0;
      if (mt > 0 || md > 0) {
        meshMorphCount += 1;
        meshMorphTargets = Math.max(meshMorphTargets, md);
      }
    }
    const box = new THREE.Box3().setFromObject(avatarRoot);
    const center = box.getCenter(new THREE.Vector3());
    basePos.copy(center).multiplyScalar(-1);
    avatarRoot.position.copy(basePos).add(userOffset);
    controls.target.copy(userOffset);
    lastBounds = { min: [box.min.x, box.min.y, box.min.z], max: [box.max.x, box.max.y, box.max.z] };
    lodLevelEl.textContent = "Rig";
    avatarLoaded = true;
    cacheIdleBones();
    if (toggleTranslateEl && toggleTranslateEl.checked) {
      setTranslateEnabled(true);
    }
    if (pendingInit) {
      applyAnimInit(pendingInit);
      pendingInit = null;
    }
    autoDetectFaceRegions();
    autoDetectBlinkMorphs();
    refreshIdleBlinkTargets();
    refreshDebugMorphTargets();
    fitCameraToAvatar();
    log("Avatar loaded:", {
      meshes: skinnedMeshes.length,
      bones: bonesByName.size,
      morphs: morphTargetMap.size,
      morphMeshes: meshMorphCount,
      morphTargetsMax: meshMorphTargets,
    });
    exposeViewerDebug();
    if (ENABLE_MORPH_DEBUGGER) initMorphDebugPanel();
  } catch (err) {
    error("Failed to load head.glb", err);
  }
}

function fitCameraToAvatar() {
  if (!avatarRoot) return;
  const box = new THREE.Box3().setFromObject(avatarRoot);
  const dx = box.max.x - box.min.x;
  const dy = box.max.y - box.min.y;
  const dz = box.max.z - box.min.z;
  const size = Math.max(dx, dy, dz);
  const dist = Math.max(1.0, size * 2.2);
  camera.position.set(0, 0, dist);
  camera.near = Math.max(0.01, dist / 100);
  camera.far = dist * 10;
  camera.updateProjectionMatrix();
  controls.update();
  cameraFitted = true;
  log("Camera fitted:", { size, dist });
}

function initMorphDebugPanel() {
  if (document.getElementById('morphDebugPanel')) return;

  const style = document.createElement('style');
  style.textContent = `
    #morphDebugPanel {
      position: fixed;
      right: 20px;
      top: 20px;
      width: 280px;
      max-height: 85vh;
      background: rgba(25, 25, 25, 0.95);
      color: #eee;
      border: 1px solid #444;
      border-radius: 12px;
      padding: 16px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      z-index: 10000;
      overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    #morphDebugPanel h3 {
      margin: 0 0 16px 0;
      font-size: 15px;
      font-weight: 600;
      border-bottom: 1px solid #333;
      padding-bottom: 10px;
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .debug-row {
      margin-bottom: 14px;
    }
    .debug-label {
      display: block;
      margin-bottom: 6px;
      color: #888;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .debug-select {
      width: 100%;
      background: #111;
      color: #fff;
      border: 1px solid #444;
      border-radius: 6px;
      padding: 8px;
      outline: none;
      cursor: pointer;
    }
    .debug-slider-container {
      display: flex;
      align-items: center;
      gap: 12px;
      background: #111;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid #444;
    }
    .debug-slider {
      flex: 1;
      cursor: pointer;
    }
    .debug-slider-value {
      min-width: 45px;
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #4caf50;
      font-weight: 600;
    }
    .debug-button-group {
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }
    .debug-btn {
      flex: 1;
      padding: 10px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12px;
      transition: all 0.2s ease;
    }
    .debug-btn-apply { background: #2e7d32; color: white; }
    .debug-btn-apply:hover { background: #388e3c; transform: translateY(-1px); }
    .debug-btn-reset { background: #c62828; color: white; }
    .debug-btn-reset:hover { background: #d32f2f; transform: translateY(-1px); }
    
    /* Custom Scrollbar */
    #morphDebugPanel::-webkit-scrollbar { width: 6px; }
    #morphDebugPanel::-webkit-scrollbar-track { background: transparent; }
    #morphDebugPanel::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
  `;
  document.head.appendChild(style);

  const panel = document.createElement('div');
  panel.id = 'morphDebugPanel';
  panel.innerHTML = `
    <h3>
      <span>Morph Debugger</span>
      <span style="font-size: 10px; opacity: 0.5;">v1.1</span>
    </h3>
    <div class="debug-row">
      <label class="debug-label" style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: #4caf50;">
        <input type="checkbox" id="debugPauseAnim" style="margin: 0;">
        Pause System Animation
      </label>
    </div>
    <div class="debug-row">
      <label class="debug-label">Target Group</label>
      <select id="debugMorphType" class="debug-select">
        <option value="Exp">Expressions (Exp###)</option>
        <option value="Pose">Corrective Poses (Pose###)</option>
        <option value="Shape">Shape Targets (Shape###)</option>
      </select>
    </div>
    <div class="debug-row">
      <label class="debug-label">Select Morph</label>
      <select id="debugMorphTarget" class="debug-select"></select>
    </div>
    <div class="debug-row">
      <label class="debug-label">Intensity Control</label>
      <div class="debug-slider-container">
        <input type="range" id="debugMorphSlider" class="debug-slider" min="-10" max="10" step="0.01" value="0">
        <span id="debugMorphVal" class="debug-slider-value">0.00</span>
      </div>
    </div>
    <div class="debug-button-group">
      <button id="debugMorphReset" class="debug-btn debug-btn-reset">Reset</button>
      <button id="debugMorphApply" class="debug-btn debug-btn-apply">Set Value</button>
    </div>
  `;
  document.body.appendChild(panel);

  const pauseCheckbox = document.getElementById('debugPauseAnim');
  const typeSelect = document.getElementById('debugMorphType');
  const targetSelect = document.getElementById('debugMorphTarget');
  const slider = document.getElementById('debugMorphSlider');
  const valLabel = document.getElementById('debugMorphVal');
  const applyBtn = document.getElementById('debugMorphApply');
  const resetBtn = document.getElementById('debugMorphReset');

  pauseCheckbox.addEventListener('change', () => {
    debugManualMode = pauseCheckbox.checked;
    log(`[Debug] Manual mode ${debugManualMode ? 'ON' : 'OFF'}`);
  });

  function populateDropdown() {
    const type = typeSelect.value;
    const currentSelection = targetSelect.value;
    targetSelect.innerHTML = '';

    const names = Array.from(morphTargetMap.keys())
      .filter(name => name.startsWith(type))
      .sort((a, b) => {
        const numA = parseInt(a.replace(/\D/g, '')) || 0;
        const numB = parseInt(b.replace(/\D/g, '')) || 0;
        return numA - numB;
      });

    if (names.length === 0) {
      const opt = document.createElement('option');
      opt.textContent = 'None found';
      opt.disabled = true;
      targetSelect.appendChild(opt);
    } else {
      names.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        targetSelect.appendChild(opt);
      });
      if (names.includes(currentSelection)) {
        targetSelect.value = currentSelection;
      }
    }
  }

  typeSelect.addEventListener('change', populateDropdown);

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valLabel.textContent = val.toFixed(2);
    valLabel.style.color = val === 0 ? '#888' : (val > 0 ? '#4caf50' : '#ff9800');
  });

  applyBtn.addEventListener('click', () => {
    const name = targetSelect.value;
    if (!name || targetSelect.disabled) return;

    const value = parseFloat(slider.value);
    const targets = morphTargetMap.get(name);

    if (targets) {
      targets.forEach(t => {
        t.mesh.morphTargetInfluences[t.index] = value;
      });
      log(`[Debug] ${name} set to ${value}`);

      // Feedback animation
      applyBtn.style.transform = 'scale(0.95)';
      setTimeout(() => applyBtn.style.transform = '', 100);
    }
  });

  resetBtn.addEventListener('click', () => {
    const name = targetSelect.value;
    if (!name || targetSelect.disabled) return;

    slider.value = 0;
    valLabel.textContent = "0.00";
    valLabel.style.color = '#888';

    const targets = morphTargetMap.get(name);
    if (targets) {
      targets.forEach(t => {
        t.mesh.morphTargetInfluences[t.index] = 0;
      });
      log(`[Debug] ${name} reset`);

      // Feedback animation
      resetBtn.style.transform = 'scale(0.95)';
      setTimeout(() => resetBtn.style.transform = '', 100);
    }
  });

  // Initial population
  populateDropdown();
}

function applyAnimInit(msg) {
  updateStreamFpsState(msg.streamFps);
  if (Number.isFinite(msg.bufferSeconds)) {
    maxBufferSeconds = msg.bufferSeconds;
  }
  streamFpsEl.textContent = hasReadyStreamFps() ? `${streamFps}` : "-";
  if (msg.blink) {
    if (Array.isArray(msg.blink.intervalSec)) {
      idleConfig.blink.intervalSec = msg.blink.intervalSec.map((v) => Number(v));
    }
    if (Number.isFinite(msg.blink.durationSec)) {
      idleConfig.blink.durationSec = Number(msg.blink.durationSec);
    }
    if (Number.isFinite(msg.blink.strength)) {
      idleConfig.blink.strength = Number(msg.blink.strength);
    }
    const left = Array.isArray(msg.blink.left) ? msg.blink.left : [];
    const right = Array.isArray(msg.blink.right) ? msg.blink.right : [];
    idleBlinkPreferredNames = Array.from(new Set([...left, ...right]));
  }
  if (msg.saccade) {
    if (Array.isArray(msg.saccade.intervalSec)) {
      idleConfig.saccade.intervalSec = msg.saccade.intervalSec.map((v) => Number(v));
    }
    if (Number.isFinite(msg.saccade.yawDeg)) {
      idleConfig.saccade.yawDeg = Number(msg.saccade.yawDeg);
    }
    if (Number.isFinite(msg.saccade.pitchDeg)) {
      idleConfig.saccade.pitchDeg = Number(msg.saccade.pitchDeg);
    }
  }
  if (msg.mouth && Array.isArray(msg.mouth.morphs)) {
    debugMouthPreferredNames = msg.mouth.morphs.slice();
  } else {
    debugMouthPreferredNames = [];
  }
  morphNames = msg.morphs || [];
  smoothedMorphs = new Float32Array(morphNames.length);
  const boneNames = msg.bones || [];
  boneOrder = boneNames.map((name) => bonesByName.get(name) || null);
  boneRestQuats = boneOrder.map((b) => (b ? b.quaternion.clone() : new THREE.Quaternion()));
  boneRestPos = boneOrder.map((b) => (b ? b.position.clone() : new THREE.Vector3()));
  const rootName = msg.rootBone || "Hips";
  rootBone = bonesByName.get(rootName) || boneOrder.find((b) => b);
  rootRestPos = rootBone ? rootBone.position.clone() : new THREE.Vector3();
  const missingBones = boneNames.filter((name) => !bonesByName.get(name));
  const missingMorphs = morphNames.filter((name) => !morphTargetMap.has(name));
  const expMeshes = new Set();
  for (const [name, targets] of morphTargetMap.entries()) {
    if (!name.startsWith("Exp")) continue;
    for (const t of targets) {
      expMeshes.add(t.mesh);
    }
  }
  expMeshCount = expMeshes.size;
  if (!loggedAliasSummary) {
    loggedAliasSummary = true;
    log("Morph alias summary:", {
      morphsReceived: morphNames.length,
      aliasAdded: morphAliasCount,
      missingMorphs: missingMorphs.length,
    });
  }
  log("Anim init:", {
    bones: boneOrder.length,
    morphs: morphNames.length,
    missingBones: missingBones.slice(0, 8),
    missingMorphs: missingMorphs.slice(0, 8),
    expMeshes: expMeshCount,
  });
  exposeViewerDebug();
  if (expMeshCount === 0) {
    warn("No meshes with Exp morph targets found; face expressions will not render.");
  }
  if (missingMorphs.some((name) => name.startsWith("Exp"))) {
    warn("Missing Exp morph targets:", missingMorphs.slice(0, 8));
  }
  autoDetectFaceRegions();
  autoDetectBlinkMorphs();
  refreshIdleBlinkTargets();
  refreshDebugMorphTargets();
  refreshSpeechMorphIndices();
  resetLivePlaybackFilters();
}

function applyAnimFrame(frame, { advanceRootSmoothing = true } = {}) {
  if (!avatarLoaded || !boneOrder.length) return;
  const nb = boneOrder.length;
  const nm = morphNames.length;
  if (frame.length < 3 + nb * 4 + nm) return;
  const rootX = frame[0];
  const rootY = frame[1];
  const rootZ = frame[2];
  let offset = 3;
  // Apply bone/root pose so jaw bone animation is not suppressed.
  for (let i = 0; i < nb; i++) {
    const bone = boneOrder[i];
    if (bone) {
      tempQuat.set(frame[offset], frame[offset + 1], frame[offset + 2], frame[offset + 3]);
      bone.quaternion.copy(boneRestQuats[i]).multiply(tempQuat);
    }
    offset += 4;
  }
  if (rootBone) {
    const rootOffset = stabilizeRootTranslation(rootX, rootY, rootZ, advanceRootSmoothing);
    tempVec.copy(rootOffset);
    rootBone.position.copy(rootRestPos).add(tempVec);
  }
  // Offset already advanced inside the loop above.
  if (DEBUG) {
    log("applyAnimFrame execute", {
      playCount,
      firstMorph: frame[3 + nb * 4],
      nmorphs: nm
    });
  }
  let maxAbs = 0;
  let mouthEnergy = 0;
  let mouthCount = 0;
  if (!smoothedMorphs || smoothedMorphs.length !== nm) {
    smoothedMorphs = new Float32Array(nm);
  }
  for (let i = 0; i < nm; i++) {
    const targets = morphTargetMap.get(morphNames[i]);
    let v = frame[offset + i];
    if (v > MORPH_CLAMP) v = MORPH_CLAMP;
    else if (v < -MORPH_CLAMP) v = -MORPH_CLAMP;
    const smoothed =
      smoothedMorphs[i] * (1 - MORPH_EMA_ALPHA) + v * MORPH_EMA_ALPHA;
    smoothedMorphs[i] = smoothed;
    if (targets) {
      for (const t of targets) {
        const base = t.base ?? 0;
        t.mesh.morphTargetInfluences[t.index] = base + smoothed;
      }
    }
    const av = Math.abs(v);
    if (av > maxAbs) maxAbs = av;
    if (mouthMorphIndexSet.has(i)) {
      mouthEnergy += Math.abs(smoothed);
      mouthCount += 1;
    }
  }
  const nextSpeechEnergy = mouthCount > 0 ? mouthEnergy / mouthCount : maxAbs;
  speechEnergySmoothed = speechEnergySmoothed * 0.7 + nextSpeechEnergy * 0.3;
  lastAppliedAt = performance.now();
  const sample = {};
  for (const name of ["Exp000", "Exp010", "Exp020"]) {
    const idx = morphNames.indexOf(name);
    sample[name] = idx >= 0 ? frame[offset + idx] : null;
  }
  lastAppliedMorphSamples = sample;
  frameCount += 1;
  if (!loggedFirstFrame) {
    loggedFirstFrame = true;
    const exp0Idx = morphNames.indexOf("Exp000");
    const exp0Val = exp0Idx >= 0 ? frame[3 + nb * 4 + exp0Idx] : null;
    const exp10Idx = morphNames.indexOf("Exp010");
    const exp10Val = exp10Idx >= 0 ? frame[3 + nb * 4 + exp10Idx] : null;
    const exp20Idx = morphNames.indexOf("Exp020");
    const exp20Val = exp20Idx >= 0 ? frame[3 + nb * 4 + exp20Idx] : null;
    log("Anim frame sample:", {
      root: [rootX.toFixed(3), rootY.toFixed(3), rootZ.toFixed(3)],
      quat0: nb > 0 ? frame.slice(3, 7).map((v) => v.toFixed(3)) : [],
      morph0: nm > 0 ? frame[3 + nb * 4].toFixed(3) : null,
      exp0: exp0Val !== null ? exp0Val.toFixed(3) : null,
      exp10: exp10Val !== null ? exp10Val.toFixed(3) : null,
      exp20: exp20Val !== null ? exp20Val.toFixed(3) : null,
      expMeshes: expMeshCount,
    });
  }
  if (!loggedMorphInfluence) {
    loggedMorphInfluence = true;
    const sample = {};
    for (const name of ["Exp000", "Exp010", "Exp020"]) {
      const targets = morphTargetMap.get(name);
      if (targets && targets.length) {
        sample[name] = targets[0].mesh.morphTargetInfluences[targets[0].index]?.toFixed(3);
      } else {
        sample[name] = null;
      }
    }
    log("Morph influence sample:", sample);
  }
}

let animationFrameId = null;
let hudFrameId = null;
let running = false;

function animate() {
  if (!running) return;
  animationFrameId = requestAnimationFrame(animate);

  if (!debugManualMode) {
    // Track idle state based on whether server is actively sending new frames.
    // applyIdlePose is called INSIDE updatePlaybackFrameWorker so it can never
    // override server stream data — it only applies after server data is written.
    updateIdleState();

    // Server animation is authoritative — idle is applied only inside this
    // function, after server frames have been written (or skipped).
    updatePlaybackFrameWorker();
  }

  controls.update();
  renderer.render(scene, camera);
}

function startManualOverride(type, durationSec, payload) {
  manualOverride.active = true;
  manualOverride.type = type;
  manualOverride.startSec = performance.now() / 1000;
  manualOverride.durationSec = durationSec;
  manualOverride.payload = payload || null;
  autoDetectFaceRegions();
  autoDetectBlinkMorphs();
  refreshIdleBlinkTargets();
  refreshDebugMorphTargets();
}

window.testMorph = function (name, value) {
  const targets = morphTargetMap.get(name);
  if (!targets) {
    error(`Morph target ${name} not found in map.`);
    return;
  }
  targets.forEach(t => {
    t.mesh.morphTargetInfluences[t.index] = value;
    log(`Applied manual morph: ${name} = ${value} on ${t.mesh.name}`);
  });
};

function initWorker() {
  if (!workerEnabled) return;
  if (worker) {
    try {
      worker.terminate();
    } catch {
      // Ignore terminate errors.
    }
    worker = null;
  }
  pipelineModeEl.textContent = "Worker";
  streamFps = null;
  streamFpsReady = false;
  workerTiming = null;
  currentPlayFps = 0;
  resetLivePlaybackFilters();
  worker = new Worker(new URL("./anim_worker.js", import.meta.url), { type: "module" });
  startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
  const known = loadKnownSessionState();
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "init") {
      updateStreamFpsState(msg.streamFps);
      streamFpsReady = Boolean(msg.streamFpsReady || hasReadyStreamFps());
      workerReady = true;
      if (avatarLoaded) {
        applyAnimInit(msg);
      } else {
        pendingInit = msg;
      }
    } else if (msg.type === "frame") {
      const incomingIndex = Number.isFinite(msg.frameIndex) ? msg.frameIndex : -1;
      if (incomingIndex >= 0 && lastWorkerFrameIndexSeen >= 0 && incomingIndex < lastWorkerFrameIndexSeen) {
        if (DEBUG) warn("Dropping out-of-order frame", { incomingIndex, lastWorkerFrameIndexSeen });
        return;
      }
      updateStreamFpsState(msg.streamFps);
      streamFpsReady = Boolean(msg.streamFpsReady || hasReadyStreamFps());
      streamFpsEl.textContent = hasReadyStreamFps() ? `${streamFps}` : "-";
      updateWorkerTiming(msg.timing);
      workerQueueLen = msg.queueLen ?? workerQueueLen;
      workerSnapshotDropped = msg.snapshotDropCount ?? workerSnapshotDropped;
      workerLiveDropped = msg.liveDropCount ?? workerLiveDropped;
      workerResyncSkipped = msg.resyncSkipped ?? workerResyncSkipped;
      workerFrameIndex = msg.frameIndex ?? workerFrameIndex;
      workerFrame = new Float32Array(msg.buffer);
      heldFrameData = workerFrame;
      frameDirty = true;
      lagIdleBlend = 0;
      lagIdleStarted = false;
      lastWorkerFrameIndexSeen = workerFrameIndex;
      lastWorkerFrameAt = performance.now();
      lastWorkerFrameBytes = msg.buffer ? msg.buffer.byteLength : null;
      if (msg.sessionId) workerSessionId = msg.sessionId;
      clearMismatchIfAligned();
      if (msg.serverBootId) serverBootId = msg.serverBootId;
      if (msg.serverClockId) serverClockId = msg.serverClockId;
      if (Number.isFinite(msg.protocolVersion)) protocolVersion = msg.protocolVersion;
      persistKnownSessionState();
      const wasResyncing = resyncing;
      resyncing = false;
      sessionMismatchWarned = false;
      // If we just cleared resyncing, attempt to flush any audio queued during the resync window.
      if (wasResyncing && audioEnabled && audioCtx && !audioStarted) {
        tryStartPlayback();
      }
    } else if (msg.type === "handshake") {
      updateStreamFpsState(msg.streamFps);
      streamFpsReady = Boolean(msg.streamFpsReady || hasReadyStreamFps());
      if (msg.sessionId) workerSessionId = msg.sessionId;
      clearMismatchIfAligned();
      if (msg.serverBootId) serverBootId = msg.serverBootId;
      if (msg.serverClockId) serverClockId = msg.serverClockId;
      if (Number.isFinite(msg.protocolVersion)) protocolVersion = msg.protocolVersion;
      if (!buildLogged) {
        buildLogged = true;
        log("Worker handshake", {
          buildId: msg.buildId || BUILD_ID,
          protocolVersion,
          serverBootId,
          serverClockId,
          sessionId: workerSessionId,
          mode: msg.mode,
        });
      }
      persistKnownSessionState();
    } else if (msg.type === "snapshot_anchor") {
      startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
      resyncing = true;
      resetLivePlaybackFilters();
    } else if (msg.type === "resync") {
      resyncing = true;
      startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
      resetLivePlaybackFilters();
      if (DEBUG) warn("Worker requested resync", msg);
    } else if (msg.type === "session_switch") {
      if (msg.sessionId) workerSessionId = msg.sessionId;
      workerTiming = null;
      heldFrameData = null;
      lagIdleBlend = 0;
      lagIdleStarted = false;
      resetLivePlaybackFilters();
      clearMismatchIfAligned();
      resyncing = false;
      sessionMismatchWarned = false;
      startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
      persistKnownSessionState();
    } else if (msg.type === "status") {
      updateStreamFpsState(msg.streamFps);
      streamFpsReady = Boolean(msg.streamFpsReady || hasReadyStreamFps());
      updateWorkerTiming(msg.timing);
      if (msg.status === "connected") {
        statusEl.textContent = "Connected";
        pipelineModeEl.textContent = "Worker";
        workerAlive = true;
        resyncing = false;
        if (workerFallbackTimer) {
          clearTimeout(workerFallbackTimer);
          workerFallbackTimer = null;
        }
      } else if (msg.status === "reset_required") {
        warn("Worker requested reset - clearing stale session state", msg);
        sessionStorage.removeItem("viewer_known_boot_id");
        sessionStorage.removeItem("viewer_known_server_clock_id");
        sessionStorage.removeItem("viewer_known_session_id");
        sessionStorage.removeItem("viewer_last_applied_frame");
        workerSessionId = null;
        workerTiming = null;
        pipelineModeEl.textContent = "Resync";
        statusEl.textContent = "Resyncing";
        workerAlive = false;
        workerReady = false;
        resyncing = true;
        startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
        resetLivePlaybackFilters();
        scheduleWorkerRestart("reset_required");
      } else if (msg.status === "error" || msg.status === "closed") {
        if (workerRestartPending) {
          pipelineModeEl.textContent = "Resync";
          statusEl.textContent = "Reconnecting";
          workerAlive = false;
          workerReady = false;
          return;
        }
        warn("Worker status:", msg);
        pipelineModeEl.textContent = "Error";
        statusEl.textContent = "Disconnected";
        workerAlive = false;
        workerTiming = null;
        if (audioCtx) {
          audioCtx.suspend();
        }
      }
    } else if (msg.type === "stats") {
      workerInFps = msg.inFps ?? workerInFps;
      workerQueueLen = msg.queueLen ?? workerQueueLen;
      workerSnapshotDropped = msg.snapshotDropCount ?? workerSnapshotDropped;
      workerLiveDropped = msg.liveDropCount ?? workerLiveDropped;
      workerResyncSkipped = msg.resyncSkipped ?? workerResyncSkipped;
      updateStreamFpsState(msg.streamFps);
      streamFpsReady = Boolean(msg.streamFpsReady || hasReadyStreamFps());
      updateWorkerTiming(msg.timing);
      workerMaxIndex = Number.isFinite(msg.maxIndex) ? msg.maxIndex : workerMaxIndex;
      workerMinIndex = Number.isFinite(msg.minIndexEstimate) ? msg.minIndexEstimate : workerMinIndex;
      if (msg.sessionId) workerSessionId = msg.sessionId;
      if (msg.playbackState) {
        resyncing = msg.playbackState === "resyncing" || msg.playbackState === "snapshot_loading";
      }
      if ((audioStarted || silentStartTime !== null) && !animOffsetSet) {
        maybeSetAnimOffset();
      }
      if (DEBUG) {
        lastStatsLog = logDelta(
          "Worker",
          {
            inFps: workerInFps,
            queueLen: workerQueueLen,
            snapshotDropped: workerSnapshotDropped,
            liveDropped: workerLiveDropped,
            resyncSkipped: workerResyncSkipped,
            maxIndex: msg.maxIndex,
            minIndex: msg.minIndexEstimate,
            streamFps: hasReadyStreamFps() ? streamFps : "-",
            transportagems: workerTiming?.transportagems ?? "-",
            infer_ms: workerTiming?.infer_ms ?? "-",
            state: msg.playbackState,
            sessionId: workerSessionId,
          },
          lastStatsLog
        );
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
  worker.postMessage({
    type: "init",
    host: WS_HOST,
    knownBootId: known.knownBootId,
    knownServerClockId: known.knownServerClockId,
    knownSessionId: known.knownSessionId,
    lastAppliedFrame: known.knownLastAppliedFrame,
    buildId: BUILD_ID,
  });
  workerFallbackTimer = setTimeout(() => {
    if (!workerReady) {
      warn("Worker not ready");
      statusEl.textContent = "Connecting...";
    }
  }, 5000);
}

function scheduleWorkerRestart(reason, delayMs = 120) {
  if (!workerEnabled || !initialized) return;
  if (workerRestartPending) return;
  workerRestartPending = true;
  if (workerRestartTimer) {
    clearTimeout(workerRestartTimer);
    workerRestartTimer = null;
  }
  workerRestartTimer = setTimeout(() => {
    workerRestartTimer = null;
    workerRestartPending = false;
    if (!initialized) return;
    if (DEBUG) {
      warn("Restarting worker", { reason });
    }
    initWorker();
  }, delayMs);
}

function updateConversationHud() {
  if (conversationStatusEl) {
    conversationStatusEl.textContent = conversationConnected ? "connected" : "disconnected";
  }
  if (conversationStateEl) {
    conversationStateEl.textContent = assistantLifecycleState;
  }
  if (conversationSessionEl) {
    conversationSessionEl.textContent = conversationStreamSessionId || "-";
  }
}

function sendConversationMessage(payload) {
  if (conversationClient) conversationClient.send(payload);
}

function handleAssistantAudioChunk(chunkData, audioHeader) {
  if (!audioCtx || !audioHeader) return;
  const sr = audioHeader.sr || 16000;
  const pcm = new Int16Array(chunkData);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    floats[i] = pcm[i] / 32768;
  }
  const buffer = audioCtx.createBuffer(1, floats.length, sr);
  buffer.copyToChannel(floats, 0);
  const duration = buffer.duration;
  if (hasSessionMismatch()) {
    if (!sessionMismatchWarned) {
      sessionMismatchWarned = true;
      warn("Session mismatch between anim/audio", { workerSessionId, audioSessionId });
      beginSessionRecovery("anim_audio_mismatch");
    }
    queueAudioChunk(buffer, duration);
    return;
  }
  if (resyncing) {
    queueAudioChunk(buffer, duration);
    return;
  }
  if (!audioStarted) {
    queueAudioChunk(buffer, duration);
  } else {
    scheduleAudioBuffer(buffer, duration);
  }
}

function connectConversationSocket() {
  if (conversationClient) return;
  conversationClient = new ConversationClient({
    wsHost: WS_HOST,
    buildId: BUILD_ID,
    callbacks: {
      onConnectionChange: (connected) => {
        conversationConnected = connected;
        updateConversationHud();
      },
      onLifecycleChange: (state) => {
        if (state === "idle_from_thinking") {
          assistantLifecycleState = "idle";
        } else {
          assistantLifecycleState = state;
        }
        updateConversationHud();
      },
      onStreamSessionChange: (sessionId) => {
        if (sessionId && conversationStreamSessionId && sessionId !== conversationStreamSessionId) {
          beginSessionRecovery("conversation_stream_switch");
        }
        conversationStreamSessionId = sessionId || conversationStreamSessionId;
        updateConversationHud();
      },
      onError: (msg) => {
        warn("Conversation error", msg);
      },
      onHelloAck: (meta) => {
        conversationId = meta.conversationId || conversationId;
        serverBootId = meta.serverBootId || serverBootId;
        serverClockId = meta.serverClockId || serverClockId;
        protocolVersion = meta.protocolVersion || protocolVersion;
        updateConversationHud();
        if (DEBUG) {
          log("Conversation hello_ack", meta);
        }
      },
      onAssistantText: (text) => {
        latestAssistantResponseText += text;
        console.log("[RAG Debug] Assistant text chunk:", text);
      },
      onAssistantTextComplete: async (fullText) => {
        latestAssistantResponseText = fullText;
        if (latestUserTranscript && fullText) {
          console.log(`[RAG] Save Interaction: "${latestUserTranscript}" -> "${fullText}"`);
          try {
            const { addMemory } = await import("../rag/retrieval/ragService.js");
            const interactionText = `User asked: "${latestUserTranscript}"\nAssistant replied: "${fullText}"`;
            await addMemory(interactionText);
          } catch (err) {
            console.error("[RAG] Save Interaction error:", err);
          }
          latestUserTranscript = "";
          latestAssistantResponseText = "";
        }
      },
      onAudioConnectionChange: (connected) => {
        if (!audioStatusEl) return;
        audioStatusEl.textContent = connected ? "connected" : "disconnected";
      },
      onAudioHeader: (audioHeader) => {
        const nextAudioSessionId = audioHeader.stream_session_id || audioHeader.session_id || null;
        if (nextAudioSessionId && conversationSessionEl) {
          conversationSessionEl.textContent = nextAudioSessionId;
        }
        if (audioSessionId && nextAudioSessionId !== audioSessionId) {
          beginSessionRecovery("audio_session_switch");
        }
        if (nextAudioSessionId) audioSessionId = nextAudioSessionId;
        clearMismatchIfAligned();
      },
      onAudioControl: (msg) => {
        if (msg.action === "stop") {
          beginSessionRecovery("audio_control_stop");
        }
      },
      onAudioChunk: (chunkData, audioHeader) => {
        handleAssistantAudioChunk(chunkData, audioHeader);
      },
    }
  });
  conversationClient.connectConversation();
}

async function ensureConversationConnected(timeoutMs = 4000) {
  if (!conversationClient) connectConversationSocket();
  await conversationClient.ensureConnected(timeoutMs);
}

function initSpeechRecognition() {
  if (recognition) return;
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("[SpeechRecognition] SpeechRecognition is not supported in this browser.");
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        final += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }
    currentSpeechTranscript = (final || interim || "").trim();
    console.log("[SpeechRecognition] Real-time transcript:", currentSpeechTranscript);
  };

  recognition.onerror = (event) => {
    console.error("[SpeechRecognition] Error:", event.error);
  };

  recognition.onend = () => {
    console.log("[SpeechRecognition] Ended");
  };
}

async function getFinalTranscriptAndSearch() {
  if (recognition) {
    try {
      recognition.stop();
    } catch (e) {
      console.warn("[SpeechRecognition] Stop failed", e);
    }
  }

  // Poll for transcript up to 500ms to allow final result to settle
  let attempts = 0;
  while (attempts < 5 && !currentSpeechTranscript) {
    await new Promise((resolve) => setTimeout(resolve, 100));
    attempts++;
  }

  const transcript = currentSpeechTranscript.trim();
  if (!transcript) {
    return { transcript: "", context: "" };
  }

  console.log(`[RAG] Transcribed Query: "${transcript}"`);
  try {
    const { semanticSearch } = await import("../rag/retrieval/ragService.js");
    const matches = await semanticSearch(transcript);
    
    // Filter matches with similarity score > 0.3
    const relevant = matches.filter((m) => m.score > 0.3);
    if (relevant.length > 0) {
      const contextStr = relevant.map((m) => `- ${m.text}`).join("\n");
      console.log(`[RAG] Found matching memories:`, contextStr);
      return { transcript, context: contextStr };
    } else {
      console.log(`[RAG] No matching memories found above threshold`);
    }
  } catch (err) {
    console.error("[RAG] Failed to search memories:", err);
  }

  return { transcript, context: "" };
}

async function startPushToTalk() {
  if (pttActive || pttStarting) return;
  pttStarting = true;
  currentSpeechTranscript = ""; // Clear old transcript
  latestAssistantResponseText = ""; // Clear old response

  // Start speech recognition
  initSpeechRecognition();
  if (recognition) {
    try {
      recognition.start();
      console.log("[SpeechRecognition] Started capturing...");
    } catch (e) {
      console.warn("[SpeechRecognition] Start error (already running?):", e);
    }
  }

  try {
    if (!audioEnabled) await onEnableAudio();
    await ensureConversationConnected();
    await ensureMicCapture({
      isPttActive: () => pttActive,
      isConvConnected: () => conversationConnected,
      sendMsg: sendConversationMessage,
      getPttSeq: () => pttSeq++
    });
    if (!pttPressed) {
      // Pointer was released while async setup was in-flight.
      if (recognition) {
        try { recognition.stop(); } catch (e) {}
      }
      sendConversationMessage({ type: "ptt_end" });
      return;
    }
    pttActive = true;
    pttSeq = 0;
    assistantLifecycleState = "listening";
    updateConversationHud();
    sendConversationMessage({ type: "ptt_start" });
    if (!pttPressed) {
      stopPushToTalk({ forceSend: true });
      return;
    }
    if (pttButton) pttButton.classList.add("active");
  } finally {
    pttStarting = false;
  }
}

async function stopPushToTalk({ forceSend = false } = {}) {
  if (!pttActive) {
    if (forceSend && conversationConnected) {
      if (assistantLifecycleState === "listening") {
        assistantLifecycleState = "thinking";
        updateConversationHud();
      }
      const { transcript, context } = await getFinalTranscriptAndSearch();
      latestUserTranscript = transcript;
      sendConversationMessage({ type: "ptt_end", context, transcript });
      
      // Save transcript to local memory if it is a statement or fact (>= 3 words)
      if (transcript && transcript.split(" ").length >= 3) {
        try {
          const { addMemory } = await import("../rag/retrieval/ragService.js");
          await addMemory(transcript);
        } catch (err) {
          console.error("[RAG] Failed to save memory:", err);
        }
      }
    }
    if (pttButton) pttButton.classList.remove("active");
    return;
  }
  pttActive = false;
  if (pttButton) pttButton.classList.remove("active");

  if (assistantLifecycleState === "listening") {
    assistantLifecycleState = "thinking";
    updateConversationHud();
  }

  const { transcript, context } = await getFinalTranscriptAndSearch();
  latestUserTranscript = transcript;
  sendConversationMessage({ type: "ptt_end", context, transcript });

  // Save transcript to local memory if it is a statement or fact (>= 3 words)
  if (transcript && transcript.split(" ").length >= 3) {
    try {
      const { addMemory } = await import("../rag/retrieval/ragService.js");
      await addMemory(transcript);
    } catch (err) {
      console.error("[RAG] Failed to save memory:", err);
    }
  }
}

function onInterruptReply() {
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
  }
  sendConversationMessage({ type: "interrupt" });
  beginSessionRecovery("manual_interrupt");
}

function onDisconnectMic() {
  pttPressed = false;
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
  }
  stopPushToTalk({ forceSend: true });
  sendConversationMessage({ type: "interrupt" });
  teardownMicCapture();
  assistantLifecycleState = "idle";
  updateConversationHud();
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

function hasReadyStreamFps() {
  return Number.isFinite(streamFps) && streamFps > 0;
}

function updateStreamFpsState(nextFps) {
  if (Number.isFinite(nextFps) && nextFps > 0) {
    streamFps = nextFps;
    streamFpsReady = true;
  }
}

function getDisplayFrameBudgetMs() {
  return hasReadyStreamFps() ? 1000 / streamFps : 50;
}

function formatMetric(value, suffix = "", digits = 1) {
  if (!Number.isFinite(value)) return "-";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function updateWorkerTiming(timing) {
  if (!timing || typeof timing !== "object") return;
  const prevFlushReason = workerTiming?.flush_reason;
  workerTiming = { ...(workerTiming || {}), ...timing };
  if (timing.flush_reason === "idle_timeout" && prevFlushReason !== "idle_timeout") {
    idleTailArmed = true;
    idleTailStartMs = null;
  } else if (timing.flush_reason === "batch_complete" && prevFlushReason !== "batch_complete") {
    idleTailArmed = false;
    idleTailStartMs = null;
  }
}

function updateInferBudgetColor() {
  if (!inferMsEl) return;
  const inferMs = Number(workerTiming?.infer_ms);
  if (!Number.isFinite(inferMs)) {
    inferMsEl.style.color = "";
    return;
  }
  const frameBudgetMs = getDisplayFrameBudgetMs();
  if (inferMs <= 0.75 * frameBudgetMs) {
    inferMsEl.style.color = "#4ade80";
  } else if (inferMs <= frameBudgetMs) {
    inferMsEl.style.color = "#facc15";
  } else {
    inferMsEl.style.color = "#f87171";
  }
}

function meshBufferedSeconds() {
  if (!hasReadyStreamFps()) return 0;
  return workerQueueLen / streamFps;
}

function loadKnownSessionState() {
  try {
    const knownBootId = sessionStorage.getItem("viewer_known_boot_id");
    const knownServerClockId = sessionStorage.getItem("viewer_known_server_clock_id");
    const knownSessionId = sessionStorage.getItem("viewer_known_session_id");
    const rawFrame = sessionStorage.getItem("viewer_last_applied_frame");
    const knownLastAppliedFrame = rawFrame !== null ? Number(rawFrame) : -1;
    return {
      knownBootId: knownBootId || null,
      knownServerClockId: knownServerClockId || null,
      knownSessionId: knownSessionId || null,
      knownLastAppliedFrame: Number.isFinite(knownLastAppliedFrame) ? knownLastAppliedFrame : -1,
    };
  } catch {
    return { knownBootId: null, knownServerClockId: null, knownSessionId: null, knownLastAppliedFrame: -1 };
  }
}

function persistKnownSessionState() {
  try {
    if (serverBootId) sessionStorage.setItem("viewer_known_boot_id", serverBootId);
    if (serverClockId) sessionStorage.setItem("viewer_known_server_clock_id", serverClockId);
    if (workerSessionId) sessionStorage.setItem("viewer_known_session_id", workerSessionId);
    if (Number.isFinite(lastWorkerFrameIndexSeen)) {
      sessionStorage.setItem("viewer_last_applied_frame", String(lastWorkerFrameIndexSeen));
    }
  } catch {
    // Ignore storage failures.
  }
}

function applyStallEase(progress) {
  const p = Math.max(0, Math.min(1, progress));
  const scale = 1 - p;
  for (const m of skinnedMeshes) {
    if (!m.morphTargetInfluences) continue;
    for (let i = 0; i < m.morphTargetInfluences.length; i++) {
      m.morphTargetInfluences[i] *= scale;
    }
  }
}


function hasSessionMismatch() {
  if (!workerSessionId || !audioSessionId) return false;
  if (workerSessionId === audioSessionId) {
    mismatchStartMs = null;
    return false;
  }
  const now = performance.now();
  if (mismatchStartMs === null) mismatchStartMs = now;
  // Only treat as mismatch if it persists beyond grace.
  return (now - mismatchStartMs) > SESSION_MISMATCH_GRACE_MS;
}

function clearMismatchIfAligned() {
  if (!workerSessionId || !audioSessionId) return false;
  if (workerSessionId !== audioSessionId) return false;
  mismatchStartMs = null;
  sessionMismatchWarned = false;
  return true;
}

function queueAudioChunk(buffer, duration) {
  audioQueue.push({ buffer, duration });
  audioQueuedSec += duration;
}

function beginSessionRecovery(reason) {
  const now = performance.now();
  if (now - lastSessionResetAt < 250) return;
  lastSessionResetAt = now;
  const hardReset =
    reason === "clock_mismatch" ||
    reason === "server_clock_mismatch" ||
    reason === "reset_required";
  mismatchStartMs = now;
  resyncing = true;
  startupSuppressUntilMs = now + STARTUP_SUPPRESS_MS;
  sessionMismatchWarned = false;
  animOffsetSec = 0;
  animOffsetSet = false;
  if (hardReset) {
    audioStarted = false;
    audioStartTime = null;
    audioScheduledTime = audioCtx ? audioCtx.currentTime + AUDIO_JITTER_SEC : 0;
    audioQueue = [];
    audioQueuedSec = 0;
    fallbackTickElapsedSec = 0;
    fallbackTickLastMs = now;
  }
  workerFrame = null;
  heldFrameData = null;
  workerFrameIndex = -1;
  workerMaxIndex = -1;
  workerMinIndex = -1;
  workerQueueLen = 0;
  workerTiming = null;
  frameDirty = false;
  lagIdleBlend = 0;
  lagIdleStarted = false;
  lastWorkerFrameIndexSeen = -1;
  resetLivePlaybackFilters();
  if (worker) worker.postMessage({ type: "reset" });
  if (DEBUG) {
    warn("Session recovery started", {
      reason,
      hardReset,
      workerSessionId,
      audioSessionId,
    });
  }
}

function computeAnimOffsetSec() {
  if (!hasReadyStreamFps()) return 0;
  if (workerMinIndex >= 0) {
    return workerMinIndex / streamFps;
  }
  if (workerMaxIndex >= 0 && workerQueueLen > 0) {
    return Math.max(0, (workerMaxIndex - workerQueueLen + 1) / streamFps);
  }
  return 0;
}

function maybeSetAnimOffset() {
  if (animOffsetSet) return;
  const hasIndex = workerMinIndex >= 0 || (workerMaxIndex >= 0 && workerQueueLen > 0);
  if (!hasIndex) return;
  animOffsetSec = computeAnimOffsetSec();
  animOffsetSet = true;
  if (DEBUG) {
    log("Anim offset set", {
      animOffsetSec: animOffsetSec.toFixed(3),
      workerMinIndex,
      workerMaxIndex,
      workerQueueLen,
    });
  }
}

function computePlaybackFps(bufferSec) {
  if (!hasReadyStreamFps()) return 0;
  const maxFps = streamFps;
  const minFps = Math.max(8, streamFps * 0.85);
  const t = Math.min(1, Math.max(0, (bufferSec - 0.5) / 3.5));
  return minFps + t * (maxFps - minFps);
}

function computeWorkerTickElapsed(nowMs, useMonotonicFallback = false) {
  if (!useMonotonicFallback && audioStarted && audioCtx && audioStartTime !== null) {
    maybeSetAnimOffset();
    const synced = Math.max(0, audioCtx.currentTime - audioStartTime + animOffsetSec);
    fallbackTickElapsedSec = Math.max(fallbackTickElapsedSec, synced);
    fallbackTickLastMs = nowMs;
    return synced;
  }
  if (!fallbackTickLastMs) {
    fallbackTickLastMs = nowMs;
  }
  const dt = Math.max(0, Math.min(0.25, (nowMs - fallbackTickLastMs) / 1000));
  fallbackTickElapsedSec += dt;
  fallbackTickLastMs = nowMs;
  return fallbackTickElapsedSec;
}

function tryStartPlayback() {
  if (audioStarted || !audioEnabled || !audioCtx) return;
  if (!streamFpsReady || !workerReady) return;
  if (resyncing || hasSessionMismatch()) return;
  const meshBuf = Math.max(workerQueueLen / streamFps, (workerMaxIndex + 1) / streamFps);
  const audioBuf = audioQueuedSec;
  if (meshBuf >= MESH_START_BUFFER_SEC && audioBuf >= AUDIO_START_BUFFER_SEC) {
    audioStartTime = audioCtx.currentTime + AUDIO_START_LEAD_SEC;
    audioScheduledTime = audioStartTime;
    for (const item of audioQueue) {
      scheduleAudioBuffer(item.buffer, item.duration);
    }
    audioQueue = [];
    audioQueuedSec = 0;
    audioStarted = true;
    silentStartTime = null;
    maybeSetAnimOffset();
    log("Audio started:", { audioStartTime });
  } else if (DEBUG) {
    log("Audio wait", { meshBuf, audioBuf, workerQueueLen, workerMaxIndex });
  }
}


function updatePlaybackFrameWorker() {
  const nowSec = performance.now() / 1000;
  const nowMs = performance.now();
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
  if (!streamFpsReady) {
    currentPlayFps = 0;
    playStateEl.textContent = "awaiting_init";
    return;
  }
  if (manualPaused) {
    playStateEl.textContent = "paused";
    return;
  }
  if (resyncing || hasSessionMismatch()) {
    playStateEl.textContent = "resyncing";
    // return;
  }
  tryStartPlayback();
  const audioBuf = audioBufferedSeconds();
  const useMonotonicFallback = audioBuf < AUDIO_LOW_BUFFER_SEC;
  const elapsed = computeWorkerTickElapsed(nowMs, useMonotonicFallback);
  if (worker) {
    worker.postMessage({ type: "tick", elapsed });
  }
  if (!audioStarted || !audioCtx || audioStartTime === null) {
    playStateEl.textContent = "buffering";
    return;
  }
  if (resyncing || hasSessionMismatch()) {
    playStateEl.textContent = "resyncing";
  } else if (audioBuf < AUDIO_LOW_BUFFER_SEC) {
    playStateEl.textContent = "holding";
  } else {
    playStateEl.textContent = "playing";
  }
  if (elapsed < 0) {
    playStateEl.textContent = "buffering";
    return;
  }

  // Keep rendering alive during lag by replaying last frame and blending idle additively.
  if (workerFrame && frameDirty) {
    // ── LIVE STREAM PATH ── server is actively sending new frames.
    // NO idle is ever applied here — server data is the only authority.
    resetAllMorphsToBase();
    applyAnimFrame(workerFrame, { advanceRootSmoothing: true });
    applySpeechBodyOverlay(nowSec, Math.min(1, speechEnergySmoothed / 0.35));
    heldFrameData = workerFrame;
    lagIdleBlend = 0;
    lagIdleStarted = false;
    if (DEBUG && playCount % 30 === 0) {
      log("Applying audio-synced frame", { playCount, elapsed: elapsed.toFixed(3) });
    }
    frameDirty = false;
    playCount += 1;
    lastFrameUpdate = nowMs;
    stallSinceMs = null;
  } else if (heldFrameData) {
    // ── HELD FRAME PATH ── server stopped; showing last received frame.
    // Apply held frame first, then blend idle on top as time passes.
    resetAllMorphsToBase();
    applyAnimFrame(heldFrameData, { advanceRootSmoothing: false });
    const stalledForMs = Math.max(0, nowMs - lastFrameUpdate);
    let speechOverlayScale = 1;
    if (stalledForMs <= STALL_HOLD_MS) {
      playStateEl.textContent = "stalled_hold";
      lagIdleBlend = 0;
      lagIdleStarted = false;
    } else if (stalledForMs <= IDLE_START_MS) {
      playStateEl.textContent = "stalled_hold";
      lagIdleBlend = 0;
      lagIdleStarted = false;
    } else {
      if (!lagIdleStarted) {
        beginLagIdleOverlay(nowSec);
        lagIdleStarted = true;
      }
      lagIdleBlend = Math.min(1, (stalledForMs - IDLE_START_MS) / STALL_IDLE_BLEND_MS);
      // Additive idle on top of held server frame — does NOT override collar/shoulder
      // that server last set; those will only be overridden when we reach the no-frame path.
      applyIdlePoseAdditive(nowSec, lagIdleBlend);
      playStateEl.textContent = "stalled_idle";
    }
    if (idleTailArmed) {
      speechOverlayScale = applyUtteranceTail(nowMs);
      playStateEl.textContent = "tail_decay";
    }
    applySpeechBodyOverlay(nowSec, Math.min(1, speechEnergySmoothed / 0.35), speechOverlayScale);
    if (DEBUG && !frameDirty && playCount % 30 === 0) {
      log("Audio holding - replaying last frame", {
        elapsed: elapsed.toFixed(3),
        stalledForMs: Math.round(stalledForMs),
        lagIdleBlend: lagIdleBlend.toFixed(2),
      });
    }
  } else {
    // ── NO FRAME PATH ── worker has no data yet.
    // Full idle is safe here since there is no server frame to protect.
    if (stallSinceMs === null) stallSinceMs = nowMs;
    const stalledForMs = nowMs - stallSinceMs;
    if (idleActive) {
      applyIdlePose(nowSec);
      playStateEl.textContent = "stalled_idle";
    } else if (stalledForMs <= STALL_HOLD_MS) {
      playStateEl.textContent = "stalled_hold";
    } else {
      const easeT = Math.min(1, (stalledForMs - STALL_HOLD_MS) / STALL_EASE_MS);
      applyStallEase(easeT);
      playStateEl.textContent = "stalled_ease";
    }
  }

  if (manualOverride.active) {
    applyManualOverride(nowSec, true);
  }

  if (workerFrameIndex >= 0) {
    currentFrameIndex = workerFrameIndex;
  }
  const bufferSec = meshBufferedSeconds();
  currentPlayFps = computePlaybackFps(bufferSec);
  if (streamFpsReady && workerMaxIndex >= 0) {
    const audioFrame = Math.floor(elapsed * streamFps);
    const drift = audioFrame - workerMaxIndex;
    const allowResync = bufferSec < 2.0;
    // Don't count drift if audio hasn't fully started — offset may not be calibrated yet.
    const audioLive = audioStarted && audioStartTime !== null;
    if (allowResync && audioLive && drift > DRIFT_THRESHOLD_FRAMES) {
      if (driftStart === null) {
        driftStart = performance.now();
      } else if (performance.now() - driftStart > RESYNC_GRACE_MS) {
        const oldStart = audioStartTime;
        audioStartTime = audioCtx.currentTime + animOffsetSec - (workerMaxIndex / streamFps);
        driftStart = null;
        warn("Resync drift", {
          drift,
          oldStart,
          newStart: audioStartTime,
          maxIndex: workerMaxIndex,
          animOffsetSec: animOffsetSec.toFixed(3),
        });
      }
    } else {
      driftStart = null;
    }
  }
  if (DEBUG && audioStarted) {
    const now = performance.now();
    if (now > startupSuppressUntilMs && now - lastFrameUpdate > 2000) {
      warn("No frames applied for >2s", { workerQueueLen, workerReady, streamFps });
      lastFrameUpdate = now;
    }
  }
}

function updateHud() {
  if (!running) return;
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
  const displayPlayFps = currentPlayFps;
  playFpsEl.textContent = `${Math.round(displayPlayFps)}`;
  streamFpsEl.textContent = hasReadyStreamFps() ? `${streamFps}` : "-";
  audioBufferEl.textContent = `${audioBufferedSeconds().toFixed(1)}s`;
  bufferFillEl.style.width = `${Math.min(100, (bufferSec / maxBufferSeconds) * 100)}%`;
  if (transportAgeEl) transportAgeEl.textContent = formatMetric(Number(workerTiming?.transportagems), "ms", 1);
  if (inputWaitEl) inputWaitEl.textContent = formatMetric(Number(workerTiming?.inputqueuewait_ms), "ms", 1);
  if (inferMsEl) inferMsEl.textContent = formatMetric(Number(workerTiming?.infer_ms), "ms", 1);
  if (resampleMsEl) resampleMsEl.textContent = formatMetric(Number(workerTiming?.resample_ms), "ms", 1);
  if (retargetMsEl) retargetMsEl.textContent = formatMetric(Number(workerTiming?.retarget_ms), "ms", 1);
  if (outputWaitEl) outputWaitEl.textContent = formatMetric(Number(workerTiming?.outputqueuewait_ms), "ms", 1);
  if (flushReasonEl) flushReasonEl.textContent = workerTiming?.flush_reason || "-";
  updateInferBudgetColor();
  if (DEBUG) {
    const now = performance.now();
    if (now - lastDebugSummary > 1000) {
      lastDebugSummary = now;
      lastHudLog = logDelta(
        "HUD",
        {
          status: statusEl.textContent,
          pipeline: pipelineModeEl.textContent,
          bufferSec: bufferSec.toFixed(2),
          queueLen: workerQueueLen,
          transportagems: workerTiming?.transportagems ?? "-",
          inFps: inFpsEl.textContent,
          outFps: outFpsEl.textContent,
          streamFps: hasReadyStreamFps() ? streamFps : "-",
          inputqueuewait_ms: workerTiming?.inputqueuewait_ms ?? "-",
          infer_ms: workerTiming?.infer_ms ?? "-",
          resample_ms: workerTiming?.resample_ms ?? "-",
          retarget_ms: workerTiming?.retarget_ms ?? "-",
          outputqueuewait_ms: workerTiming?.outputqueuewait_ms ?? "-",
          flush_reason: workerTiming?.flush_reason ?? "-",
          audioBuf: audioBufferedSeconds().toFixed(2),
          audioStarted,
          workerReady,
          workerSessionId,
          audioSessionId,
          serverBootId,
          protocolVersion,
          snapshotDropped: workerSnapshotDropped,
          liveDropped: workerLiveDropped,
          resyncSkipped: workerResyncSkipped,
        },
        lastHudLog
      );
    }
  }
  if (expVal0El && lastAppliedMorphSamples) {
    expVal0El.textContent = (lastAppliedMorphSamples.Exp000 || 0).toFixed(3);
    expVal1El.textContent = (lastAppliedMorphSamples.Exp010 || 0).toFixed(3);
    expVal2El.textContent = (lastAppliedMorphSamples.Exp020 || 0).toFixed(3);
  }
  hudFrameId = requestAnimationFrame(updateHud);
}

function ensureHelpers() {
  if (!gridHelper) {
    gridHelper = new THREE.GridHelper(5, 10, 0xcccccc, 0xeeeeee);
  }
  if (!axesHelper) {
    axesHelper = new THREE.AxesHelper(1.2);
  }
}

function setHelperVisible(helper, visible) {
  if (!helper) return;
  if (visible) {
    if (!scene.children.includes(helper)) scene.add(helper);
  } else {
    scene.remove(helper);
  }
}

function ensureTransformControls() {
  if (transformControls) return;
  if (!gizmoAnchor) {
    gizmoAnchor = new THREE.Object3D();
    gizmoAnchor.position.copy(userOffset);
    scene.add(gizmoAnchor);
  }
  transformControls = new TransformControls(camera, renderer.domElement);
  transformControls.setMode("translate");
  transformControls.setSpace("world");
  transformControls.addEventListener("dragging-changed", (event) => {
    controls.enabled = !event.value;
    isDragging = event.value;
    if (!isDragging && gizmoAnchor) {
      userOffset.copy(gizmoAnchor.position);
      if (avatarRoot) {
        avatarRoot.position.copy(basePos).add(userOffset);
      }
      controls.target.copy(userOffset);
    }
  });
  transformControls.addEventListener("objectChange", () => {
    if (!avatarRoot || !gizmoAnchor) return;
    userOffset.copy(gizmoAnchor.position);
    avatarRoot.position.copy(basePos).add(userOffset);
    controls.target.copy(userOffset);
  });
}

function setTranslateEnabled(enabled) {
  if (!avatarRoot) return;
  if (enabled) {
    ensureTransformControls();
    if (gizmoAnchor) {
      gizmoAnchor.position.copy(userOffset);
      transformControls.attach(gizmoAnchor);
    }
    if (!scene.children.includes(transformControls)) scene.add(transformControls);
  } else if (transformControls) {
    transformControls.detach();
    scene.remove(transformControls);
  }
}

function setCameraPreset(dir) {
  if (!avatarRoot) return;
  const target = controls.target.clone();
  const dist = camera.position.distanceTo(target);
  const v = dir.clone().normalize().multiplyScalar(dist);
  camera.position.set(target.x + v.x, target.y + v.y, target.z + v.z);
  camera.updateProjectionMatrix();
  controls.update();
}

function focusFaceView() {
  if (!avatarRoot) return;
  let target = null;
  const headBone = bonesByName.get("Head");
  if (headBone) {
    target = headBone.getWorldPosition(new THREE.Vector3());
    target.y += faceOffset;
  } else if (lastBounds) {
    const min = lastBounds.min;
    const max = lastBounds.max;
    const height = Math.max(1e-6, max[1] - min[1]);
    const base = 0.12;
    const factor = Math.min(0.45, Math.max(-0.2, base + faceOffset));
    const headY = max[1] - height * factor;
    const headX = (min[0] + max[0]) * 0.5;
    const headZ = (min[2] + max[2]) * 0.5;
    target = new THREE.Vector3(headX, headY, headZ).add(avatarRoot.position);
  } else {
    target = controls.target.clone();
  }
  controls.target.copy(target);
  const dist = Math.max(0.3, camera.position.distanceTo(target) * 0.6);
  const dir = new THREE.Vector3(0.6, 0.2, 1).normalize().multiplyScalar(dist);
  camera.position.copy(target.clone().add(dir));
  camera.near = Math.max(0.01, dist / 100);
  camera.far = dist * 10;
  camera.updateProjectionMatrix();
  controls.update();
}

function onTogglePlay() {
  manualPaused = !manualPaused;
  if (togglePlayBtn) {
    togglePlayBtn.textContent = manualPaused ? "Resume" : "Pause";
  }
  if (audioCtx) {
    if (manualPaused) {
      audioCtx.suspend();
    } else {
      audioCtx.resume();
    }
  }
}

function onClearBuffer() {
  currentFrameIndex = 0;
  workerQueueLen = 0;
  workerMaxIndex = -1;
  workerMinIndex = -1;
  workerFrameIndex = -1;
  workerFrame = null;
  heldFrameData = null;
  frameDirty = false;
  lagIdleBlend = 0;
  lagIdleStarted = false;
  fallbackTickElapsedSec = 0;
  fallbackTickLastMs = performance.now();
  stallSinceMs = null;
  resyncing = false;
  sessionMismatchWarned = false;
  startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
  animOffsetSec = 0;
  animOffsetSet = false;
  userOffset.set(0, 0, 0);
  if (avatarRoot) {
    avatarRoot.position.copy(basePos);
  }
  if (gizmoAnchor) {
    gizmoAnchor.position.copy(userOffset);
  }
  for (const m of skinnedMeshes) {
    if (m.morphTargetInfluences) {
      m.morphTargetInfluences.fill(0);
    }
  }
  if (worker) {
    worker.postMessage({ type: "reset" });
  }
}

function onResetCam() {
  camera.position.set(1.2, 1.4, 2.5);
  userOffset.set(0, 0, 0);
  if (avatarRoot) {
    avatarRoot.position.copy(basePos);
  }
  if (gizmoAnchor) {
    gizmoAnchor.position.copy(userOffset);
  }
  controls.target.set(0, 1.0, 0);
  controls.update();
}

function onFitView() {
  fitCameraToAvatar();
}

function onViewFace() {
  focusFaceView();
}

function onViewFront() {
  setCameraPreset(new THREE.Vector3(0, 0, 1));
}

function onViewBack() {
  setCameraPreset(new THREE.Vector3(0, 0, -1));
}

function onViewLeft() {
  setCameraPreset(new THREE.Vector3(-1, 0, 0));
}

function onViewRight() {
  setCameraPreset(new THREE.Vector3(1, 0, 0));
}

function onViewTop() {
  setCameraPreset(new THREE.Vector3(0, 1, 0));
}

function onViewIso() {
  setCameraPreset(new THREE.Vector3(1, 0.8, 1));
}

function onFaceOffsetInput() {
  if (!faceOffsetEl) return;
  faceOffset = parseFloat(faceOffsetEl.value) || 0;
  if (faceOffsetValEl) faceOffsetValEl.textContent = faceOffset.toFixed(2);
  focusFaceView();
}

function onToggleGrid() {
  ensureHelpers();
  setHelperVisible(gridHelper, toggleGridEl.checked);
}

function onToggleAxes() {
  ensureHelpers();
  setHelperVisible(axesHelper, toggleAxesEl.checked);
}

function onToggleWireframe() {
  for (const m of skinnedMeshes) {
    if (m.material) {
      m.material.wireframe = toggleWireframeEl.checked;
    }
  }
}

function onToggleAutoRotate() {
  controls.autoRotate = toggleAutoRotateEl.checked;
}

function onToggleTranslate() {
  setTranslateEnabled(toggleTranslateEl.checked);
}

async function onEnableAudio() {
  if (audioEnabled && audioCtx?.state === "running") return;
  
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  
  try {
    await audioCtx.resume();
  } catch (e) {
    warn("Failed to resume audio context", e);
  }

  if (audioCtx.state === "running") {
    audioEnabled = true;
    if (audioStatusEl) audioStatusEl.textContent = "enabled";
    silentStartTime = null;
    startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
    connectConversationSocket();
    conversationClient.connectAudioOut();
  }
}

export function isAudioReady() {
  return audioEnabled && audioCtx?.state === "running";
}

function onConnectConversation() {
  connectConversationSocket();
  updateConversationHud();
}

function onPttPointerDown(event) {
  event.preventDefault();
  pttPressed = true;
  startPushToTalk().catch((err) => {
    warn("PTT start failed", err);
    pttPressed = false;
    pttActive = false;
    if (pttButton) pttButton.classList.remove("active");
  });
}

function onPttPointerUp(event) {
  event.preventDefault();
  pttPressed = false;
  stopPushToTalk({ forceSend: true });
}

function onResize() {
  if (!camera || !renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function bindUi() {
  if (boundUi) return;
  boundUi = true;
  if (togglePlayBtn) togglePlayBtn.addEventListener("click", onTogglePlay);
  if (clearBufferBtn) clearBufferBtn.addEventListener("click", onClearBuffer);
  if (resetCamBtn) resetCamBtn.addEventListener("click", onResetCam);
  if (fitViewBtn) fitViewBtn.addEventListener("click", onFitView);
  if (viewFaceBtn) viewFaceBtn.addEventListener("click", onViewFace);
  if (viewFrontBtn) viewFrontBtn.addEventListener("click", onViewFront);
  if (viewBackBtn) viewBackBtn.addEventListener("click", onViewBack);
  if (viewLeftBtn) viewLeftBtn.addEventListener("click", onViewLeft);
  if (viewRightBtn) viewRightBtn.addEventListener("click", onViewRight);
  if (viewTopBtn) viewTopBtn.addEventListener("click", onViewTop);
  if (viewIsoBtn) viewIsoBtn.addEventListener("click", onViewIso);
  if (faceOffsetEl) faceOffsetEl.addEventListener("input", onFaceOffsetInput);
  if (toggleGridEl) toggleGridEl.addEventListener("change", onToggleGrid);
  if (toggleAxesEl) toggleAxesEl.addEventListener("change", onToggleAxes);
  if (toggleWireframeEl) toggleWireframeEl.addEventListener("change", onToggleWireframe);
  if (toggleAutoRotateEl) toggleAutoRotateEl.addEventListener("change", onToggleAutoRotate);
  if (toggleTranslateEl) toggleTranslateEl.addEventListener("change", onToggleTranslate);
  if (enableAudioBtn) enableAudioBtn.addEventListener("click", onEnableAudio);
  if (connectConversationBtn) connectConversationBtn.addEventListener("click", onConnectConversation);
  if (interruptReplyBtn) interruptReplyBtn.addEventListener("click", onInterruptReply);
  if (disconnectMicButton) disconnectMicButton.addEventListener("click", onDisconnectMic);
  if (pttButton) {
    pttButton.addEventListener("mousedown", onPttPointerDown);
    pttButton.addEventListener("mouseup", onPttPointerUp);
    pttButton.addEventListener("mouseleave", onPttPointerUp);
    pttButton.addEventListener("touchstart", onPttPointerDown, { passive: false });
    pttButton.addEventListener("touchend", onPttPointerUp, { passive: false });
    pttButton.addEventListener("touchcancel", onPttPointerUp, { passive: false });
  }
  resizeHandler = onResize;
  window.addEventListener("resize", resizeHandler);

  // Automatically enable audio on the first user interaction (browser requirement)
  window.addEventListener("click", () => {
    onEnableAudio().catch(err => warn("Auto-enable audio failed", err));
  }, { once: true });
  window.addEventListener("keydown", () => {
    onEnableAudio().catch(err => warn("Auto-enable audio failed", err));
  }, { once: true });
}

function unbindUi() {
  if (!boundUi) return;
  boundUi = false;
  if (togglePlayBtn) togglePlayBtn.removeEventListener("click", onTogglePlay);
  if (clearBufferBtn) clearBufferBtn.removeEventListener("click", onClearBuffer);
  if (resetCamBtn) resetCamBtn.removeEventListener("click", onResetCam);
  if (fitViewBtn) fitViewBtn.removeEventListener("click", onFitView);
  if (viewFaceBtn) viewFaceBtn.removeEventListener("click", onViewFace);
  if (viewFrontBtn) viewFrontBtn.removeEventListener("click", onViewFront);
  if (viewBackBtn) viewBackBtn.removeEventListener("click", onViewBack);
  if (viewLeftBtn) viewLeftBtn.removeEventListener("click", onViewLeft);
  if (viewRightBtn) viewRightBtn.removeEventListener("click", onViewRight);
  if (viewTopBtn) viewTopBtn.removeEventListener("click", onViewTop);
  if (viewIsoBtn) viewIsoBtn.removeEventListener("click", onViewIso);
  if (faceOffsetEl) faceOffsetEl.removeEventListener("input", onFaceOffsetInput);
  if (toggleGridEl) toggleGridEl.removeEventListener("change", onToggleGrid);
  if (toggleAxesEl) toggleAxesEl.removeEventListener("change", onToggleAxes);
  if (toggleWireframeEl) toggleWireframeEl.removeEventListener("change", onToggleWireframe);
  if (toggleAutoRotateEl) toggleAutoRotateEl.removeEventListener("change", onToggleAutoRotate);
  if (toggleTranslateEl) toggleTranslateEl.removeEventListener("change", onToggleTranslate);
  if (enableAudioBtn) enableAudioBtn.removeEventListener("click", onEnableAudio);
  if (connectConversationBtn) connectConversationBtn.removeEventListener("click", onConnectConversation);
  if (interruptReplyBtn) interruptReplyBtn.removeEventListener("click", onInterruptReply);
  if (disconnectMicButton) disconnectMicButton.removeEventListener("click", onDisconnectMic);
  if (pttButton) {
    pttButton.removeEventListener("mousedown", onPttPointerDown);
    pttButton.removeEventListener("mouseup", onPttPointerUp);
    pttButton.removeEventListener("mouseleave", onPttPointerUp);
    pttButton.removeEventListener("touchstart", onPttPointerDown);
    pttButton.removeEventListener("touchend", onPttPointerUp);
    pttButton.removeEventListener("touchcancel", onPttPointerUp);
  }
  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
    resizeHandler = null;
  }
}

function collectDom() {
  statusEl = document.getElementById("status");
  bufferSecEl = document.getElementById("bufferSec");
  queueLenEl = document.getElementById("queueLen");
  inFpsEl = document.getElementById("inFps");
  outFpsEl = document.getElementById("outFps");
  playFpsEl = document.getElementById("playFps");
  streamFpsEl = document.getElementById("streamFps");
  pipelineModeEl = document.getElementById("pipelineMode");
  audioStatusEl = document.getElementById("audioStatus");
  audioBufferEl = document.getElementById("audioBuffer");
  conversationStatusEl = document.getElementById("conversationStatus");
  conversationStateEl = document.getElementById("conversationState");
  conversationSessionEl = document.getElementById("conversationSession");
  playStateEl = document.getElementById("playState");
  lodLevelEl = document.getElementById("lodLevel");
  bufferFillEl = document.getElementById("bufferFill");
  transportAgeEl = document.getElementById("transportAge");
  inputWaitEl = document.getElementById("inputWait");
  inferMsEl = document.getElementById("inferMs");
  resampleMsEl = document.getElementById("resampleMs");
  retargetMsEl = document.getElementById("retargetMs");
  outputWaitEl = document.getElementById("outputWait");
  flushReasonEl = document.getElementById("flushReason");
  fitViewBtn = document.getElementById("fitView");
  viewFaceBtn = document.getElementById("viewFace");
  viewFrontBtn = document.getElementById("viewFront");
  viewBackBtn = document.getElementById("viewBack");
  viewLeftBtn = document.getElementById("viewLeft");
  viewRightBtn = document.getElementById("viewRight");
  viewTopBtn = document.getElementById("viewTop");
  viewIsoBtn = document.getElementById("viewIso");
  faceOffsetEl = document.getElementById("faceOffset");
  faceOffsetValEl = document.getElementById("faceOffsetVal");
  toggleGridEl = document.getElementById("toggleGrid");
  toggleAxesEl = document.getElementById("toggleAxes");
  toggleWireframeEl = document.getElementById("toggleWireframe");
  toggleAutoRotateEl = document.getElementById("toggleAutoRotate");
  toggleTranslateEl = document.getElementById("toggleTranslate");
  enableAudioBtn = document.getElementById("enableAudio");
  connectConversationBtn = document.getElementById("connectConversation");
  pttButton = document.getElementById("pttButton");
  disconnectMicButton = document.getElementById("disconnectMic");
  interruptReplyBtn = document.getElementById("interruptReply");
  togglePlayBtn = document.getElementById("togglePlay");
  clearBufferBtn = document.getElementById("clearBuffer");
  resetCamBtn = document.getElementById("resetCam");
  expVal0El = document.getElementById("expVal0");
  expVal1El = document.getElementById("expVal1");
  expVal2El = document.getElementById("expVal2");
  canvas = document.getElementById("canvas");
}

export async function initViewer() {
  if (initialized) return;
  initialized = true;
  collectDom();
  updateConversationHud();
  if (!canvas) {
    error("Canvas element not found (#canvas).");
    return;
  }
  initThree(canvas);
  bindUi();
  await loadAvatar();
  if (workerEnabled) {
    initWorker();
  } else if (pipelineModeEl && statusEl) {
    pipelineModeEl.textContent = "Unsupported";
    statusEl.textContent = "Disconnected";
  }
  running = true;
  animate();
  updateHud();

  // Attempt auto-enable (may be blocked by browser)
  onEnableAudio().catch(() => {});
}

export function destroyViewer() {
  if (!initialized) return;
  running = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (hudFrameId) cancelAnimationFrame(hudFrameId);
  animationFrameId = null;
  hudFrameId = null;
  unbindUi();
  if (worker) {
    worker.terminate();
    worker = null;
  }
  if (workerRestartTimer) {
    clearTimeout(workerRestartTimer);
    workerRestartTimer = null;
  }
  workerRestartPending = false;
  if (conversationClient) {
    conversationClient.disconnect();
    conversationClient = null;
  }
  teardownMicCapture();
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  initialized = false;
}
