/**
 * AudioCapture.js — Microphone capture via AudioWorkletNode.
 *
 * Provides:
 *  - ensureMicCapture()  — initialise mic stream + AudioWorklet processor
 *  - teardownMicCapture() — cleanly release all mic hardware tracks and nodes
 *
 * Depends on `sendConversationMessage` and state from ConversationClient, and
 * the global `pttActive` / `conversationConnected` flags kept in ViewerController.
 * These are passed in as callbacks to avoid circular imports.
 */

const CONVERSATION_AUDIO_SAMPLE_RATE = 16000;
const CONVERSATION_PTT_BUFFER = 2048;

/** Internal mic state — not exported; managed exclusively by the functions below. */
let micStream = null;
let micCaptureCtx = null;
let micSource = null;
let micProcessor = null;
let micMuteGain = null;

/** Read-only public accessor for the current sample rate (set after ensureMicCapture). */
export let micSampleRate = CONVERSATION_AUDIO_SAMPLE_RATE;

/**
 * Initialise the microphone capture pipeline.
 * Creates an AudioWorkletNode connected to the mic stream.
 * No-ops if already initialised.
 *
 * @param {object} callbacks
 * @param {() => boolean}  callbacks.isPttActive       — returns true if PTT is currently active
 * @param {() => boolean}  callbacks.isConvConnected   — returns true if conversation WS is open
 * @param {(msg: object) => void} callbacks.sendMsg    — send a message on the conversation WebSocket
 * @param {() => number}  callbacks.getPttSeq          — returns and increments PTT sequence number
 */
export async function ensureMicCapture({ isPttActive, isConvConnected, sendMsg, getPttSeq }) {
  if (micCaptureCtx && micProcessor && micSource) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia is not supported in this browser");
  }

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: CONVERSATION_AUDIO_SAMPLE_RATE,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  micCaptureCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: CONVERSATION_AUDIO_SAMPLE_RATE,
  });
  await micCaptureCtx.resume();
  micSampleRate = micCaptureCtx.sampleRate || CONVERSATION_AUDIO_SAMPLE_RATE;
  micSource = micCaptureCtx.createMediaStreamSource(micStream);

  if (!micCaptureCtx.audioWorklet) {
    throw new Error(
      "AudioWorklet is not available in this browser. Ensure the server sends the required COOP/COEP headers."
    );
  }

  const workletUrl = new URL("./mic-capture.worklet.js", import.meta.url);
  try {
    await micCaptureCtx.audioWorklet.addModule(workletUrl);
  } catch (e) {
    throw new Error(
      `AudioWorklet failed to load mic-capture processor: ${e.message}. ` +
        "Ensure the server sets the required COOP/COEP headers."
    );
  }

  micProcessor = new AudioWorkletNode(micCaptureCtx, "mic-capture-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { bufferSize: CONVERSATION_PTT_BUFFER },
  });

  micProcessor.port.onmessage = (event) => {
    if (!isPttActive() || !isConvConnected()) return;
    const samples = event.data;
    const int16 = _float32ToInt16(samples);
    sendMsg({
      type: "user_audio",
      seq: getPttSeq(),
      sr: micSampleRate,
      dtype: "int16",
      pcm_b64: _int16ToBase64(int16),
    });
  };

  micMuteGain = micCaptureCtx.createGain();
  micMuteGain.gain.value = 0;
  micSource.connect(micProcessor);
  micProcessor.connect(micMuteGain);
  micMuteGain.connect(micCaptureCtx.destination);
}

/**
 * Tear down all mic hardware and audio graph nodes.
 * Safe to call even if capture was never started.
 */
export function teardownMicCapture() {
  if (micProcessor) {
    micProcessor.disconnect();
    if (micProcessor.port) micProcessor.port.onmessage = null;
    micProcessor = null;
  }
  if (micSource) {
    micSource.disconnect();
    micSource = null;
  }
  if (micMuteGain) {
    micMuteGain.disconnect();
    micMuteGain = null;
  }
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  if (micCaptureCtx) {
    micCaptureCtx.close();
    micCaptureCtx = null;
  }
}

// ── Private helpers ────────────────────────────────────────────────────────

// Matches the reference `float32ToPcm16` from useLiveAPI audio-utils:
// clamp to [-1, 1] then scale with separate negative/positive ranges.
function _float32ToInt16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// Matches the reference `arrayBufferToBase64` from useLiveAPI audio-utils:
// iterate byte-by-byte to avoid call-stack limits on large typed arrays.
function _int16ToBase64(int16Array) {
  const bytes = new Uint8Array(int16Array.buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
