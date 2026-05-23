import { pipeline, env } from "@xenova/transformers";

// Force remote model loading in the browser
env.allowLocalModels = false;
env.allowRemoteModels = true;

// Set the correct base URL and disable automatic path resolution
env.remoteHost = 'https://huggingface.co/';

// Add cache busting to avoid stale responses
env.useBrowserCache = false;

let extractor = null;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

async function loadExtractor() {
  if (extractor) return extractor;
  
  self.postMessage({ type: "load_progress", message: `Attempting to load model: ${MODEL_NAME}` });
  
  const instance = await pipeline(
    "feature-extraction",
    MODEL_NAME,
    {
      quantized: true,
      progress_callback: (progress) => {
        self.postMessage({ type: "load_progress", progress });
      }
    }
  );
  
  extractor = instance;
  self.postMessage({ type: "load_success" });
  return instance;
}

self.onmessage = async (event) => {
  const { id, type, text } = event.data;
  
  if (type === "get_embedding") {
    try {
      if (!extractor) {
        extractor = await loadExtractor();
      }
      
      const output = await extractor(text, {
        pooling: "mean",
        normalize: true
      });
      
      const embedding = Array.from(output.data);
      self.postMessage({ id, type: "embedding_result", embedding });
    } catch (err) {
      self.postMessage({ id, type: "embedding_error", error: err.message || String(err) });
    }
  } else if (type === "clear_cache") {
    extractor = null;
    self.postMessage({ id, type: "clear_cache_result" });
  }
};
