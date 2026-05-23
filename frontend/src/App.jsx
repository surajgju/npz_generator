import { useEffect, useState } from "react";
import { destroyViewer, initViewer, isAudioReady } from "./viewer/ViewerController.js";
import { ConversationPanel } from "./ConversationPanel.jsx";

function MicIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}

function App() {
  const [showPermissionModal, setShowPermissionModal] = useState(false);

  useEffect(() => {
    initViewer().then(() => {
      // Short delay to allow onEnableAudio to complete if it can
      setTimeout(() => {
        if (!isAudioReady()) {
          setShowPermissionModal(true);
        }
      }, 500);
    });
    return () => {
      destroyViewer();
      setShowPermissionModal(false);
    };
  }, []);

  const startExperience = () => {
    setShowPermissionModal(false);
    // Trigger the global click handler in ViewerController to enable audio
    document.getElementById("enableAudio")?.click();
  };

  return (
    <div id="app">
      {showPermissionModal && (
        <div className="cv-modal-overlay">
          <div className="cv-modal">
            <div className="cv-modal-header">
              <MicIcon size={24} />
              <h3>Audio Permission</h3>
            </div>
            <p>
              This experience uses high-fidelity spatial audio and voice 
              interaction. Please allow audio and microphone access to proceed.
            </p>
            <button className="cv-modal-btn" onClick={startExperience}>
              Enable Audio & Mic
            </button>
          </div>
        </div>
      )}

      {/* ── Existing HUD (stats, view controls) ── */}
      <div id="hud">
        {/* Hidden debug statistics panel (kept in DOM to prevent Three.js controller crashes) */}
        <div style={{ display: "none" }}>
          <div className="row">
            <span>Status</span>
            <span id="status">Connecting...</span>
          </div>
          <div className="row section-title">
            <span>Pipeline</span>
            <span></span>
          </div>
          <div className="row">
            <span>Buffer</span>
            <span id="bufferSec">0.0s</span>
          </div>
          <div id="bufferBar">
            <div id="bufferFill"></div>
          </div>
          <div className="row">
            <span>Queue</span>
            <span id="queueLen">0</span>
          </div>
          <div className="row">
            <span>In FPS</span>
            <span id="inFps">0</span>
          </div>
          <div className="row">
            <span>Out FPS</span>
            <span id="outFps">0</span>
          </div>
          <div className="row">
            <span>Play FPS</span>
            <span id="playFps">0</span>
          </div>
          <div className="row">
            <span>Stream FPS</span>
            <span id="streamFps">-</span>
          </div>
          <div className="row">
            <span>Pipeline</span>
            <span id="pipelineMode">Worker</span>
          </div>
          <div className="row">
            <span>Audio</span>
            <span id="audioStatus">disabled</span>
          </div>
          <div className="row">
            <span>Audio Buf</span>
            <span id="audioBuffer">0.0s</span>
          </div>

          {/*
            ── Ghost conversation status elements ──
            ViewerController reads these by ID. We keep them in the DOM but
            invisible — the ConversationPanel reads them via MutationObserver.
          */}
          <span id="conversationStatus">disconnected</span>
          <span id="conversationState">idle</span>
          <span id="conversationSession">-</span>

          <div className="row">
            <span>Playback</span>
            <span id="playState">buffering</span>
          </div>
          <div className="row">
            <span>Transport Age</span>
            <span id="transportAge">-</span>
          </div>
          <div className="row">
            <span>LOD</span>
            <span id="lodLevel">LOD0</span>
          </div>
          <div className="row section-title">
            <span>Backend</span>
            <span></span>
          </div>
          <div className="row">
            <span>Input Wait</span>
            <span id="inputWait">-</span>
          </div>
          <div className="row">
            <span>Infer</span>
            <span id="inferMs">-</span>
          </div>
          <div className="row">
            <span>Resample</span>
            <span id="resampleMs">-</span>
          </div>
          <div className="row">
            <span>Retarget</span>
            <span id="retargetMs">-</span>
          </div>
          <div className="row">
            <span>Output Wait</span>
            <span id="outputWait">-</span>
          </div>
          <div className="row">
            <span>Flush Reason</span>
            <span id="flushReason">-</span>
          </div>
          <div className="row section-title">
            <span>Expressions</span>
            <span></span>
          </div>
          <div className="row">
            <span>Exp000 (Jaw)</span>
            <span id="expVal0">0.00</span>
          </div>
          <div className="row">
            <span>Exp010</span>
            <span id="expVal1">0.00</span>
          </div>
          <div className="row">
            <span>Exp020</span>
            <span id="expVal2">0.00</span>
          </div>
        </div>

        <div className="row">
          <strong>View</strong>
          <span></span>
        </div>
        <div className="btns">
          <button id="fitView">Fit</button>
          <button id="viewFace">Face</button>
          <button id="viewFront">Front</button>
          <button id="viewBack">Back</button>
          <button id="viewLeft">Left</button>
          <button id="viewRight">Right</button>
          <button id="viewTop">Top</button>
          <button id="viewIso">Iso</button>
        </div>
        <div className="row">
          <span>Face Y</span>
          <span>
            <input
              type="range"
              id="faceOffset"
              min="-0.25"
              max="0.25"
              step="0.01"
              defaultValue="0"
            />
            <span id="faceOffsetVal">0.00</span>
          </span>
        </div>
        <div className="row">
          <label>
            <input type="checkbox" id="toggleGrid" /> Grid
          </label>
          <span></span>
        </div>
        <div className="row">
          <label>
            <input type="checkbox" id="toggleAxes" /> Axes
          </label>
          <span></span>
        </div>
        <div className="row">
          <label>
            <input type="checkbox" id="toggleWireframe" /> Wireframe
          </label>
          <span></span>
        </div>
        <div className="row">
          <label>
            <input type="checkbox" id="toggleAutoRotate" /> Auto-Rotate
          </label>
          <span></span>
        </div>
        <div className="row">
          <label>
            <input type="checkbox" id="toggleTranslate" /> Translate
          </label>
          <span></span>
        </div>
        <div className="btns">
          <button id="enableAudio" style={{ display: "none" }}>Enable Audio</button>
          {/*
            ── Ghost conversation control buttons ──
            These are visually hidden but remain in the DOM so ViewerController
            can attach its click handlers. ConversationPanel.jsx calls .click()
            on them programmatically.
          */}
          <button id="connectConversation" style={{ display: "none" }}>Connect Voice</button>
          <button id="pttButton"           style={{ display: "none" }}>Push To Talk</button>
          <button id="disconnectMic"       style={{ display: "none" }}>Disconnect Mic</button>
          <button id="interruptReply"      style={{ display: "none" }}>Interrupt</button>
          <button id="togglePlay">Pause</button>
          <button id="clearBuffer">Clear Buffer</button>
          <button id="resetCam">Reset Cam</button>
        </div>
      </div>

      <canvas id="canvas"></canvas>

      {/* ── Premium conversation panel overlay ── */}
      <ConversationPanel />
    </div>
  );
}

export default App;
