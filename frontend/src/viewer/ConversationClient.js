/**
 * ConversationClient.js — unified conversation + audio_out WebSocket client.
 *
 * Exports a single `ConversationClient` class that manages the lifecycle of
 * `/ws/conversation` and `/ws/audio_out`: connect, disconnect, send, and
 * message dispatch.
 *
 * All assistant lifecycle state changes are communicated via callbacks so that
 * ViewerController.js can update its own rendering state without circular deps.
 */

export class ConversationClient {
  /**
   * @param {object} options
   * @param {string} options.wsHost — hostname:port for the WebSocket URL
   * @param {string} options.buildId — build identifier sent in the hello message
   * @param {object} options.callbacks — event callbacks
   * @param {(connected: boolean) => void} options.callbacks.onConnectionChange
   * @param {(state: string) => void} options.callbacks.onLifecycleChange
   * @param {(sessionId: string|null) => void} options.callbacks.onStreamSessionChange
   * @param {(msg: object) => void} options.callbacks.onError
   * @param {(meta: object) => void} options.callbacks.onHelloAck
   * @param {(connected: boolean) => void} options.callbacks.onAudioConnectionChange
   * @param {(header: object) => void} options.callbacks.onAudioHeader
   * @param {(msg: object) => void} options.callbacks.onAudioControl
   * @param {(chunk: ArrayBuffer, header: object) => void} options.callbacks.onAudioChunk
   */
  constructor({ wsHost, buildId, callbacks = {} }) {
    this._wsHost = wsHost;
    this._buildId = buildId;
    this._cb = callbacks;

    this._ws = null;
    this._audioWs = null;
    this._pendingAudioHeader = null;
    this.connected = false;
    this.audioConnected = false;
    this.conversationId = null;
    this.lifecycleState = "idle";
  }

  /** Open both sockets if not already open. */
  connect() {
    this.connectConversation();
    this.connectAudioOut();
  }

  /** Open /ws/conversation if not already open. */
  connectConversation() {
    if (this._ws) return;
    this._ws = new WebSocket(`ws://${this._wsHost}/ws/conversation`);

    this._ws.onopen = () => {
      this.connected = true;
      this._cb.onConnectionChange?.(true);
      this.send({ type: "hello", protocol_version: 1, build_id: this._buildId });
    };

    this._ws.onclose = () => {
      this.connected = false;
      this._ws = null;
      this._cb.onConnectionChange?.(false);
      this._setLifecycle("idle");
    };

    this._ws.onerror = () => {
      this.connected = false;
      this._cb.onConnectionChange?.(false);
      this._setLifecycle("idle");
    };

    this._ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (!msg || typeof msg !== "object") return;
      this._dispatch(msg);
    };
  }

  /** Open /ws/audio_out if not already open. */
  connectAudioOut() {
    if (this._audioWs) return;
    this._audioWs = new WebSocket(`ws://${this._wsHost}/ws/audio_out`);
    this._audioWs.binaryType = "arraybuffer";

    this._audioWs.onopen = () => {
      this.audioConnected = true;
      this._cb.onAudioConnectionChange?.(true);
    };

    this._audioWs.onclose = () => {
      this.audioConnected = false;
      this._pendingAudioHeader = null;
      this._audioWs = null;
      this._cb.onAudioConnectionChange?.(false);
    };

    this._audioWs.onerror = () => {
      this.audioConnected = false;
      this._cb.onAudioConnectionChange?.(false);
    };

    this._audioWs.onmessage = (event) => {
      if (typeof event.data === "string") {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          this._pendingAudioHeader = null;
          return;
        }
        if (!msg || typeof msg !== "object") {
          this._pendingAudioHeader = null;
          return;
        }
        if (msg.type === "audio") {
          this._pendingAudioHeader = msg;
          this._cb.onAudioHeader?.(msg);
          this._cb.onStreamSessionChange?.(msg.stream_session_id || msg.session_id || null);
        } else if (msg.type === "audio_control") {
          this._pendingAudioHeader = null;
          this._cb.onAudioControl?.(msg);
        }
        return;
      }

      if (!(event.data instanceof ArrayBuffer)) return;
      const header = this._pendingAudioHeader;
      this._pendingAudioHeader = null;
      if (!header || header.type !== "audio") return;
      this._cb.onAudioChunk?.(event.data, header);
    };
  }

  /** Wait until the WebSocket is open (polls with 40 ms intervals). */
  async ensureConnected(timeoutMs = 4000) {
    if (this.connected && this._ws && this._ws.readyState === WebSocket.OPEN) return;
    this.connectConversation();
    const start = performance.now();
    while (!(this.connected && this._ws && this._ws.readyState === WebSocket.OPEN)) {
      if (performance.now() - start > timeoutMs) {
        throw new Error("Conversation socket connection timeout");
      }
      await new Promise((resolve) => setTimeout(resolve, 40));
    }
  }

  /** Send a JSON message. Silently drops if not connected. */
  send(payload) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    this._ws.send(JSON.stringify(payload));
  }

  /** Disconnect and clean up. */
  disconnect() {
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    if (this._audioWs) {
      this._audioWs.close();
      this._audioWs = null;
    }
    this.connected = false;
    this.audioConnected = false;
    this._pendingAudioHeader = null;
    this._setLifecycle("idle");
  }

  // ── Private message dispatch ─────────────────────────────────────────────

  _setLifecycle(state) {
    this.lifecycleState = state;
    this._cb.onLifecycleChange?.(state);
  }

  _dispatch(msg) {
    switch (msg.type) {
      case "hello_ack":
        this.conversationId = msg.conversation_id || this.conversationId;
        this._cb.onHelloAck?.({
          conversationId: msg.conversation_id,
          serverBootId: msg.server_boot_id,
          serverClockId: msg.server_clock_id,
          protocolVersion: msg.protocol_version,
        });
        break;

      case "listening":
        this._setLifecycle("listening");
        break;

      case "assistant_thinking_start":
        this._setLifecycle("thinking");
        break;

      case "assistant_thinking_end":
        if (this.lifecycleState !== "speaking") {
          this._setLifecycle("idle");
        }
        break;

      case "assistant_speaking_start":
        this._setLifecycle("speaking");
        this._cb.onStreamSessionChange?.(msg.stream_session_id || msg.session_id || null);
        break;

      case "assistant_speaking_end":
        this._setLifecycle("idle");
        break;

      case "interrupted":
      case "interrupt_ack":
        this._setLifecycle("idle");
        this._cb.onStreamSessionChange?.(msg.stream_session_id || msg.session_id || null);
        break;

      case "error":
        this._cb.onError?.(msg);
        this._setLifecycle("idle");
        break;

      default:
        break;
    }
  }
}
