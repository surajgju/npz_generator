/**
 * HUDManager.js — DOM element references and HUD update helpers.
 *
 * Exports a single `HUD` object that:
 *  - caches all DOM element references in one place (replaces 50+ module-level lets)
 *  - provides a typed `update(state)` helper called once per animation frame
 *
 * ViewerController.js imports `HUD` and calls HUD.bind() on initialisation.
 */

export const HUD = {
  // ── Status display elements ────────────────────────────────────────────
  status: null,
  bufferSec: null,
  queueLen: null,
  inFps: null,
  outFps: null,
  playFps: null,
  streamFps: null,
  pipelineMode: null,
  audioStatus: null,
  audioBuffer: null,
  conversationStatus: null,
  conversationState: null,
  conversationSession: null,
  playState: null,
  lodLevel: null,
  bufferFill: null,
  expVal0: null,
  expVal1: null,
  expVal2: null,

  // ── Control buttons and inputs ─────────────────────────────────────────
  fitViewBtn: null,
  viewFaceBtn: null,
  viewFrontBtn: null,
  viewBackBtn: null,
  viewLeftBtn: null,
  viewRightBtn: null,
  viewTopBtn: null,
  viewIsoBtn: null,
  faceOffset: null,
  faceOffsetVal: null,
  toggleGrid: null,
  toggleAxes: null,
  toggleWireframe: null,
  toggleAutoRotate: null,
  toggleTranslate: null,
  enableAudioBtn: null,
  connectConversationBtn: null,
  pttButton: null,
  disconnectMicButton: null,
  interruptReplyBtn: null,
  togglePlayBtn: null,
  clearBufferBtn: null,
  resetCamBtn: null,

  /**
   * Bind all DOM elements by their IDs.
   * Call once after the DOM is ready.
   * @param {Record<string, string>} ids — map of property name → element ID
   */
  bind(ids) {
    for (const [prop, id] of Object.entries(ids)) {
      this[prop] = document.getElementById(id);
    }
  },

  /**
   * Update all display elements from a state snapshot.
   * @param {object} state
   */
  updateStats({
    bufferSec = 0,
    queueLen = 0,
    inFps = 0,
    outFps = 0,
    playFps = 0,
    streamFpsVal = 0,
    audioBufferSec = 0,
    maxBufferSeconds = 10,
    conversationConnected = false,
    assistantLifecycleState = "idle",
    conversationStreamSessionId = null,
    lastAppliedMorphSamples = null,
    playStateText = "",
  } = {}) {
    if (this.bufferSec) this.bufferSec.textContent = `${bufferSec.toFixed(1)}s`;
    if (this.queueLen) this.queueLen.textContent = `${queueLen}`;
    if (this.inFps) this.inFps.textContent = `${inFps}`;
    if (this.outFps) this.outFps.textContent = `${outFps}`;
    if (this.playFps) this.playFps.textContent = `${Math.round(playFps)}`;
    if (this.streamFps) this.streamFps.textContent = `${streamFpsVal}`;
    if (this.audioBuffer) this.audioBuffer.textContent = `${audioBufferSec.toFixed(1)}s`;
    if (this.bufferFill) {
      this.bufferFill.style.width = `${Math.min(100, (bufferSec / maxBufferSeconds) * 100)}%`;
    }
    if (this.conversationStatus) {
      this.conversationStatus.textContent = conversationConnected ? "connected" : "disconnected";
    }
    if (this.conversationState) {
      this.conversationState.textContent = assistantLifecycleState;
    }
    if (this.conversationSession) {
      this.conversationSession.textContent = conversationStreamSessionId || "-";
    }
    if (this.playState && playStateText) {
      this.playState.textContent = playStateText;
    }
    if (this.expVal0 && lastAppliedMorphSamples) {
      this.expVal0.textContent = (lastAppliedMorphSamples.Exp000 || 0).toFixed(3);
      this.expVal1.textContent = (lastAppliedMorphSamples.Exp010 || 0).toFixed(3);
      this.expVal2.textContent = (lastAppliedMorphSamples.Exp020 || 0).toFixed(3);
    }
  },
};
