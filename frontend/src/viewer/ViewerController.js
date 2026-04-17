import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
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
  blink: { intervalSec: [3.0, 6.0], durationSec: 0.12, strength: 0.6 },
  saccade: { intervalSec: [1.0, 3.0], yawDeg: 2.0, pitchDeg: 1.0 },
  headSway: { yawDeg: 2.0, pitchDeg: 1.0, rollDeg: 0.5, periodSec: [5.0, 7.0] },
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

function resetIdlePose() {
  if (idleRestQuats.head && idleBones.head) idleBones.head.quaternion.copy(idleRestQuats.head);
  if (idleRestQuats.neck && idleBones.neck) idleBones.neck.quaternion.copy(idleRestQuats.neck);
  if (idleRestQuats.leftEye && idleBones.leftEye)
    idleBones.leftEye.quaternion.copy(idleRestQuats.leftEye);
  if (idleRestQuats.rightEye && idleBones.rightEye)
    idleBones.rightEye.quaternion.copy(idleRestQuats.rightEye);
  if (idleRestQuats.jaw && idleBones.jaw) idleBones.jaw.quaternion.copy(idleRestQuats.jaw);
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


function applyIdlePose(nowSec) {
  if (!idleActive || !avatarLoaded) return;
  if (!idleRestQuats.head && idleBones.head) {
    cacheIdleBones();
  }
  const elapsed = nowSec - idleStartSec;
  // Blend blinks atop existing morphs instead of resetting everything.
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
  if (idleBones.head && idleRestQuats.head) {
    tempEuler.set(pitch, yaw, roll, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.head.quaternion.copy(idleRestQuats.head).multiply(tempQuat2);
  }
  if (idleBones.neck && idleRestQuats.neck) {
    const scale = idleConfig.neckScale ?? 0.5;
    tempEuler.set(pitch * scale, yaw * scale, roll * scale, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.neck.quaternion.copy(idleRestQuats.neck).multiply(tempQuat2);
  }
  // Subtle spine sway.
  const spineYaw = yaw * 0.5;
  const spinePitch = pitch * 0.4;
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
  // Relax arms from T-pose.
  const shoulderPitch = (-8 * Math.PI) / 180;
  const elbowBend = (-5 * Math.PI) / 180;
  if (idleBones.leftShoulder && idleRestQuats.leftShoulder) {
    tempEuler.set(shoulderPitch, 0, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftShoulder.quaternion.copy(idleRestQuats.leftShoulder).multiply(tempQuat2);
  }
  if (idleBones.rightShoulder && idleRestQuats.rightShoulder) {
    tempEuler.set(shoulderPitch, 0, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.rightShoulder.quaternion.copy(idleRestQuats.rightShoulder).multiply(tempQuat2);
  }
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
  if (idleBones.leftEye && idleBones.rightEye && idleRestQuats.leftEye && idleRestQuats.rightEye) {
    const { yaw: eyeYaw, pitch: eyePitch } = updateIdleSaccade(elapsed);
    tempEuler.set(eyePitch, eyeYaw, 0, "YXZ");
    tempQuat2.setFromEuler(tempEuler);
    idleBones.leftEye.quaternion.copy(idleRestQuats.leftEye).multiply(tempQuat2);
    idleBones.rightEye.quaternion.copy(idleRestQuats.rightEye).multiply(tempQuat2);
  }
  if (idleBlinkTargets.length) {
    const blinkVal =
      updateIdleBlink(elapsed) * (idleConfig.blink.strength || 0.6) * idleBlinkStrengthScale;
    for (const t of idleBlinkTargets) {
      const base = t.base ?? 0;
      const v = Math.max(-1, Math.min(1, base + blinkVal));
      t.mesh.morphTargetInfluences[t.index] = Math.max(-1, Math.min(1, (t.base ?? 0) + blinkVal));
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
  const noFrames = now - lastFrameUpdate > IDLE_START_MS;
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
let morphTargetMap = new Map();
let morphAliasCount = 0;
let loggedAliasSummary = false;
let morphNames = [];
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
let streamFps = 20;
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
let currentPlayFps = streamFps;

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
let silentStartTime = null;
let animOffsetSec = 0;
let animOffsetSet = false;
const AUDIO_JITTER_SEC = 0.08;
const AUDIO_START_LEAD_SEC = Number(import.meta.env.VITE_AUDIO_START_LEAD_SEC || 0.03);
const AUDIO_START_BUFFER_SEC = Number(import.meta.env.VITE_AUDIO_START_BUFFER_SEC || 0.18);
const MESH_START_BUFFER_SEC = Number(import.meta.env.VITE_MESH_START_BUFFER_SEC || 0.18);
const AUDIO_LOW_BUFFER_SEC = Number(import.meta.env.VITE_AUDIO_LOW_BUFFER_SEC || 0.12);
const STARTUP_SUPPRESS_MS = 4000;
const STALL_HOLD_MS = 300;
const STALL_EASE_MS = 700;
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
let frameDirty = false;
let workerReady = false;
let workerFallbackTimer = null;
let lastFrameUpdate = 0;
let workerMaxIndex = -1;
let workerMinIndex = -1;
let driftStart = null;
const DRIFT_THRESHOLD_FRAMES = 20;
const RESYNC_GRACE_MS = 1500;
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

const MORPH_EMA_ALPHA = 0.35;
const MORPH_CLAMP = 2.5;

const gltfLoader = new GLTFLoader();
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
    const gltf = await gltfLoader.loadAsync("./assets/head.glb");
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
  streamFps = msg.streamFps || streamFps;
  if (Number.isFinite(msg.bufferSeconds)) {
    maxBufferSeconds = msg.bufferSeconds;
  }
  streamFpsEl.textContent = `${streamFps}`;
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
}

function applyAnimFrame(frame) {
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
    tempVec.set(rootX, rootY, rootZ);
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
  if (!smoothedMorphs || smoothedMorphs.length !== nm) {
    smoothedMorphs = new Float32Array(nm);
  }
  for (let i = 0; i < nm; i++) {
    const name = morphNames[i];
    const targets = morphTargetMap.get(name);
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
  }
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
    /* TEMPORARY DEBUG BLOCK: Start - Disabling idle breathing/sway */
    /* 
    // 1. Update idle state (breathing, etc.)
    updateIdleState();
    if (idleActive) {
      applyIdlePose(performance.now() / 1000);
    }
    */
    /* TEMPORARY DEBUG BLOCK: End */

    // 2. Apply streaming animation on top (authoritative)
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
  pipelineModeEl.textContent = "Worker";
  worker = new Worker(new URL("./anim_worker.js", import.meta.url), { type: "module" });
  startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
  const known = loadKnownSessionState();
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "init") {
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
      streamFps = msg.streamFps || streamFps;
      streamFpsEl.textContent = `${streamFps}`;
      workerQueueLen = msg.queueLen ?? workerQueueLen;
      workerSnapshotDropped = msg.snapshotDropCount ?? workerSnapshotDropped;
      workerLiveDropped = msg.liveDropCount ?? workerLiveDropped;
      workerResyncSkipped = msg.resyncSkipped ?? workerResyncSkipped;
      workerFrameIndex = msg.frameIndex ?? workerFrameIndex;
      workerFrame = new Float32Array(msg.buffer);
      frameDirty = true;
      lastWorkerFrameIndexSeen = workerFrameIndex;
      lastWorkerFrameAt = performance.now();
      lastWorkerFrameBytes = msg.buffer ? msg.buffer.byteLength : null;
      if (msg.sessionId) workerSessionId = msg.sessionId;
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
      if (msg.sessionId) workerSessionId = msg.sessionId;
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
    } else if (msg.type === "resync") {
      resyncing = true;
      startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
      if (DEBUG) warn("Worker requested resync", msg);
    } else if (msg.type === "session_switch") {
      if (msg.sessionId) workerSessionId = msg.sessionId;
      resyncing = false;
      sessionMismatchWarned = false;
      startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
      persistKnownSessionState();
    } else if (msg.type === "status") {
      if (msg.status === "connected") {
        statusEl.textContent = "Connected";
        pipelineModeEl.textContent = "Worker";
        workerReady = true;
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
        pipelineModeEl.textContent = "Resync";
        statusEl.textContent = "Resyncing";
        workerAlive = false;
        workerReady = false;
        resyncing = true;
        startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
        // Optionally reload or re-init here if needed, but the next connect should be clean.
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
      workerSnapshotDropped = msg.snapshotDropCount ?? workerSnapshotDropped;
      workerLiveDropped = msg.liveDropCount ?? workerLiveDropped;
      workerResyncSkipped = msg.resyncSkipped ?? workerResyncSkipped;
      streamFps = msg.streamFps || streamFps;
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
            streamFps,
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

async function startPushToTalk() {
  if (pttActive || pttStarting) return;
  pttStarting = true;
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

function stopPushToTalk({ forceSend = false } = {}) {
  if (!pttActive) {
    if (forceSend && conversationConnected) {
      sendConversationMessage({ type: "ptt_end" });
      if (assistantLifecycleState === "listening") {
        assistantLifecycleState = "thinking";
        updateConversationHud();
      }
    }
    if (pttButton) pttButton.classList.remove("active");
    return;
  }
  pttActive = false;
  sendConversationMessage({ type: "ptt_end" });
  if (assistantLifecycleState === "listening") {
    assistantLifecycleState = "thinking";
    updateConversationHud();
  }
  if (pttButton) pttButton.classList.remove("active");
}

function onInterruptReply() {
  sendConversationMessage({ type: "interrupt" });
  beginSessionRecovery("manual_interrupt");
}

function onDisconnectMic() {
  pttPressed = false;
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

function meshBufferedSeconds() {
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
  // Only treat as mismatch if it persists > 500ms
  return (now - mismatchStartMs) > 500;
}

function queueAudioChunk(buffer, duration) {
  audioQueue.push({ buffer, duration });
  audioQueuedSec += duration;
}

function beginSessionRecovery(reason) {
  const now = performance.now();
  if (now - lastSessionResetAt < 250) return;
  lastSessionResetAt = now;
  resyncing = true;
  startupSuppressUntilMs = now + STARTUP_SUPPRESS_MS;
  sessionMismatchWarned = false;
  animOffsetSec = 0;
  animOffsetSet = false;
  audioStarted = false;
  audioStartTime = null;
  audioScheduledTime = audioCtx ? audioCtx.currentTime + AUDIO_JITTER_SEC : 0;
  audioQueue = [];
  audioQueuedSec = 0;
  workerFrame = null;
  workerFrameIndex = -1;
  workerMaxIndex = -1;
  workerMinIndex = -1;
  workerQueueLen = 0;
  frameDirty = false;
  lastWorkerFrameIndexSeen = -1;
  if (worker) worker.postMessage({ type: "reset" });
  if (DEBUG) {
    warn("Session recovery started", { reason, workerSessionId, audioSessionId });
  }
}

function computeAnimOffsetSec() {
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
  const maxFps = streamFps;
  const minFps = Math.max(8, streamFps * 0.85);
  const t = Math.min(1, Math.max(0, (bufferSec - 0.5) / 3.5));
  return minFps + t * (maxFps - minFps);
}

function tryStartPlayback() {
  if (audioStarted || !audioEnabled || !audioCtx) return;
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
  if (manualPaused) {
    playStateEl.textContent = "paused";
    return;
  }
  if (resyncing || hasSessionMismatch()) {
    playStateEl.textContent = "resyncing";
    // return;
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
  maybeSetAnimOffset();
  const elapsed = audioCtx.currentTime - audioStartTime + animOffsetSec;
  if (elapsed < 0) {
    playStateEl.textContent = "buffering";
    return;
  }
  if (worker) {
    worker.postMessage({ type: "tick", elapsed });
  }

  // Apply streamed frame only when new data arrives.
  if (workerFrame && frameDirty) {
    resetAllMorphsToBase();
    applyAnimFrame(workerFrame);
    if (DEBUG && playCount % 30 === 0) {
      log("Applying audio-synced frame", { playCount, elapsed: elapsed.toFixed(3) });
    }
    frameDirty = false;
    playCount += 1;
    playStateEl.textContent = "playing";
    lastFrameUpdate = performance.now();
    stallSinceMs = null;
  } else {
    if (stallSinceMs === null) stallSinceMs = nowMs;
    const stalledForMs = nowMs - stallSinceMs;
    if (stalledForMs <= STALL_HOLD_MS) {
      playStateEl.textContent = "stalled_hold";
    } else {
      const easeT = Math.min(1, (stalledForMs - STALL_HOLD_MS) / STALL_EASE_MS);
      applyStallEase(easeT);
      playStateEl.textContent = "stalled_ease";
    }
    if (DEBUG && !frameDirty && playCount % 30 === 0) {
      log("Audio holding - no new worker frame", {
        elapsed: elapsed.toFixed(3),
        stalledForMs: Math.round(stalledForMs),
      });
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
  if (workerMaxIndex >= 0) {
    const audioFrame = Math.floor(elapsed * streamFps);
    const drift = audioFrame - workerMaxIndex;
    const allowResync = bufferSec < 2.0;
    if (allowResync && drift > DRIFT_THRESHOLD_FRAMES) {
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
  streamFpsEl.textContent = `${streamFps}`;
  audioBufferEl.textContent = `${audioBufferedSeconds().toFixed(1)}s`;
  bufferFillEl.style.width = `${Math.min(100, (bufferSec / maxBufferSeconds) * 100)}%`;
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
          inFps: inFpsEl.textContent,
          outFps: outFpsEl.textContent,
          streamFps,
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
  frameDirty = false;
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
  if (audioEnabled) return;
  audioEnabled = true;
  if (audioStatusEl) audioStatusEl.textContent = "enabled";
  silentStartTime = null;
  startupSuppressUntilMs = performance.now() + STARTUP_SUPPRESS_MS;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  connectConversationSocket();
  conversationClient.connectAudioOut();
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