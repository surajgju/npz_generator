/**
 * ConversationClient.js — /ws/conversation WebSocket protocol client.
 *
 * Exports a single `ConversationClient` class that manages the lifecycle of
 * the conversation WebSocket: connect, disconnect, send, and message dispatch.
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
   * @param {(state: string) => void}      options.callbacks.onLifecycleChange
   * @param {(sessionId: string) => void}  options.callbacks.onStreamSessionChange
   * @param {(msg: object) => void}        options.callbacks.onError
   * @param {(meta: object) => void}       options.callbacks.onHelloAck
   */
  constructor({ wsHost, buildId, callbacks = {} }) {
    this._wsHost = wsHost;
    this._buildId = buildId;
    this._cb = callbacks;

    this._ws = null;
    this.connected = false;
    this.conversationId = null;
  }

  /** Open the WebSocket if not already open. No-ops if already connected. */
  connect() {
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
      this._cb.onLifecycleChange?.("idle");
    };

    this._ws.onerror = () => {
      this.connected = false;
      this._cb.onConnectionChange?.(false);
      this._cb.onLifecycleChange?.("idle");
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

  /** Wait until the WebSocket is open (polls with 40 ms intervals). */
  async ensureConnected(timeoutMs = 4000) {
    if (this.connected && this._ws && this._ws.readyState === WebSocket.OPEN) return;
    this.connect();
    const start = performance.now();
    while (!(this.connected && this._ws && this._ws.readyState === WebSocket.OPEN)) {
      if (performance.now() - start > timeoutMs) {
        throw new Error("Conversation socket connection timeout");
      }
      // eslint-disable-next-line no-await-in-loop
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
    this.connected = false;
  }

  // ── Private message dispatch ─────────────────────────────────────────────

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
        this._cb.onLifecycleChange?.("listening");
        break;

      case "assistant_thinking_start":
        this._cb.onLifecycleChange?.("thinking");
        break;

      case "assistant_thinking_end":
        // Only transition back to idle if we haven't moved to speaking yet.
        this._cb.onLifecycleChange?.("idle_from_thinking");
        break;

      case "assistant_speaking_start":
        this._cb.onLifecycleChange?.("speaking");
        this._cb.onStreamSessionChange?.(msg.stream_session_id || msg.session_id || null);
        break;

      case "assistant_speaking_end":
        this._cb.onLifecycleChange?.("idle");
        break;

      case "interrupted":
        this._cb.onLifecycleChange?.("idle");
        break;

      case "error":
        this._cb.onError?.(msg);
        this._cb.onLifecycleChange?.("idle");
        break;

      default:
        break;
    }
  }
}
