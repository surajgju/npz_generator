/**
 * ConversationPanel.jsx
 *
 * Premium floating conversation UI for Gemini Live voice interaction.
 * All HUD element IDs expected by ViewerController.js are rendered as
 * invisible ghost elements so the controller can still bind to them
 * via document.getElementById(). Visual state is driven by CSS classes
 * on a single data attribute updated by a MutationObserver.
 */
import { useEffect, useRef, useState } from "react";

// ─── Mini hook: observe a DOM element's text content ─────────────────────────
// Uses requestAnimationFrame for the initial read so we never call setState
// synchronously inside the effect body (avoids cascading-render lint warning).
function useElementText(id) {
  const [text, setText] = useState("");
  useEffect(() => {
    const el = document.getElementById(id);
    if (!el) return;

    // Deferred initial sync — runs after the browser has painted, by which
    // time ViewerController will have written its initial values.
    const raf = requestAnimationFrame(() => setText(el.textContent));

    const obs = new MutationObserver(() => setText(el.textContent));
    obs.observe(el, { childList: true, characterData: true, subtree: true });

    return () => {
      cancelAnimationFrame(raf);
      obs.disconnect();
    };
  }, [id]);
  return text;
}

// ─── Mic SVG icons ────────────────────────────────────────────────────────────
function MicIcon({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}
function MicOffIcon({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}
function PowerIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/>
    </svg>
  );
}
function ZapIcon({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  );
}

// ─── State label map ──────────────────────────────────────────────────────────
const STATE_LABEL = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};
const STATE_COLOR = {
  idle: "var(--cv-idle)",
  listening: "var(--cv-listening)",
  thinking: "var(--cv-thinking)",
  speaking: "var(--cv-speaking)",
};

// ─── Main component ───────────────────────────────────────────────────────────
export function ConversationPanel() {
  const convStatus = useElementText("conversationStatus");   // "connected" | "disconnected"
  const convState  = useElementText("conversationState");    // "idle" | "listening" | "thinking" | "speaking"

  const isConnected = convStatus === "connected";
  const state = convState || "idle";

  // ── PTT toggle state ──────────────────────────────────────────────────────
  // ViewerController uses mousedown/mouseup (not click) to manage pttPressed.
  // We track our own toggle so we can dispatch the correct event each press.
  const [localPttActive, setLocalPttActive] = useState(false);

  // Button refs — we dispatch real DOM events on the ghost HUD buttons
  const connectBtnRef    = useRef(null);
  const pttBtnRef        = useRef(null);
  const disconnectBtnRef = useRef(null);
  const interruptBtnRef  = useRef(null);

  // Bind ghost-button refs after mount
  useEffect(() => {
    connectBtnRef.current    = document.getElementById("connectConversation");
    pttBtnRef.current        = document.getElementById("pttButton");
    disconnectBtnRef.current = document.getElementById("disconnectMic");
    interruptBtnRef.current  = document.getElementById("interruptReply");
  }, []);

  // localPttActive drives button/orb visuals instantly (optimistic UI).
  // Controller-reported state drives the status label/dot (authoritative).

  const dispatchPointer = (el, eventType) => {
    if (!el) return;
    el.dispatchEvent(new MouseEvent(eventType, {
      bubbles: true, cancelable: true, view: window,
    }));
  };

  const handleConnectToggle = () => {
    if (isConnected) {
      // Disconnect: first stop PTT if active, then disconnect mic
      if (localPttActive) {
        dispatchPointer(pttBtnRef.current, "mouseup");
        setLocalPttActive(false);
      }
      disconnectBtnRef.current?.click();
    } else {
      connectBtnRef.current?.click();
    }
  };

  const handlePtt = () => {
    if (!isConnected) {
      // If not connected, clicking initiates connection first
      connectBtnRef.current?.click();
      return;
    }
    const btn = pttBtnRef.current;
    if (!localPttActive) {
      // ── Start PTT ── dispatch mousedown so ViewerController sets pttPressed=true
      dispatchPointer(btn, "mousedown");
      setLocalPttActive(true);
    } else {
      // ── Stop PTT ── dispatch mouseup so ViewerController calls stopPushToTalk()
      dispatchPointer(btn, "mouseup");
      setLocalPttActive(false);
    }
  };

  const handleInterrupt = () => {
    if (localPttActive) {
      dispatchPointer(pttBtnRef.current, "mouseup");
      setLocalPttActive(false);
    }
    interruptBtnRef.current?.click();
  };


  const label = STATE_LABEL[state] || state;
  const color = STATE_COLOR[state] || STATE_COLOR.idle;
  // Use localPttActive for instant visual feedback; controller state for labels.
  const isPttActive = localPttActive;
  const isSpeaking  = !localPttActive && state === "speaking";
  const isThinking  = !localPttActive && state === "thinking";

  return (
    <div className="cv-panel" data-connected={isConnected} data-state={state}>

      {/* ── Connection badge (top-right corner) ── */}
      <button
        id="cv-connect-btn"
        className={`cv-connect-badge ${isConnected ? "connected" : ""}`}
        onClick={handleConnectToggle}
        title={isConnected ? "Disconnect voice" : "Connect voice"}
      >
        <PowerIcon size={14} />
        <span>{isConnected ? "Connected" : "Connect Voice"}</span>
      </button>

      {/* ── Central orb ── */}
      <div
        className={`cv-orb-wrap ${isConnected ? "active" : ""} ${isPttActive ? "listening" : ""} ${isSpeaking ? "speaking" : ""}`}
        onClick={handlePtt}
        title={isConnected ? (isPttActive ? "Stop recording" : "Push to talk") : "Connect to start"}
      >
        {/* Pulse rings */}
        {isPttActive && (
          <>
            <span className="cv-ring cv-ring1" />
            <span className="cv-ring cv-ring2" />
          </>
        )}
        {isSpeaking && (
          <span className="cv-ring cv-ring-speak" />
        )}

        {/* Orb surface */}
        <div className="cv-orb">
          <div className="cv-orb-glow" />
          <span className="cv-orb-icon">
            {isConnected
              ? isPttActive
                ? <MicIcon size={36} />
                : <MicOffIcon size={36} />
              : <MicOffIcon size={36} />}
          </span>
        </div>
      </div>

      {/* ── Status label ── */}
      <div className="cv-status-row">
        <span className="cv-dot" style={{ background: color }} />
        <span className="cv-status-label">{isConnected ? label : "Voice off"}</span>
        {(isThinking || isSpeaking) && (
          <button
            className="cv-interrupt-btn"
            onClick={handleInterrupt}
            title="Interrupt assistant"
          >
            <ZapIcon size={11} />
            Interrupt
          </button>
        )}
      </div>

      {/* ── PTT instruction ── */}
      <p className="cv-hint">
        {isConnected
          ? isPttActive
            ? "Recording — click orb or push to stop"
            : "Click orb or push button to talk"
          : "Connect to start a voice conversation"}
      </p>

      {/* ── Large PTT button ── */}
      <button
        id="cv-ptt-big"
        className={`cv-ptt-btn ${isPttActive ? "recording" : ""} ${!isConnected ? "disabled" : ""}`}
        onClick={handlePtt}
        title={isConnected ? "Push To Talk" : "Connect first"}
      >
        {isPttActive ? <MicIcon size={22} /> : <MicOffIcon size={22} />}
        <span>{isPttActive ? "Stop" : "Push to Talk"}</span>
      </button>

    </div>
  );
}
