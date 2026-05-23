import { embeddingMemory } from "./embeddingMemory.js";

// Log all fetch requests for debugging
const originalFetch = window.fetch;
window.fetch = async (...args) => {
  const url = typeof args[0] === 'string' 
    ? args[0] 
    : args[0].url || args[0].href || '';
  if (url.includes('huggingface') || url.includes('onnx')) {
    console.log('[Fetch Debug] Requesting:', url);
  }
  const response = await originalFetch(...args);
  if (!response.ok && (url.includes('huggingface') || url.includes('onnx'))) {
    console.error('[Fetch Debug] Failed:', url, response.status);
    const text = await response.clone().text();
    console.error('[Fetch Debug] Response preview:', text.substring(0, 200));
  }
  return response;
};

// Worker setup
let worker = null;
let messageIdCounter = 0;
const pendingRequests = new Map();

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./embeddingWorker.js', import.meta.url), {
      type: 'module'
    });
    
    worker.onmessage = (event) => {
      const { id, type, embedding, error, progress, message } = event.data;
      
      if (type === 'load_progress') {
        if (message) console.log(message);
        if (progress) console.log(`Model loading progress:`, progress);
      } else if (type === 'load_success') {
        embeddingMemory.recordLoadSuccess();
        console.log(`✓ Embedding model loaded successfully`);
        embeddingMemory.setModelLoading(false);
      } else if (type === 'embedding_result' || type === 'embedding_error' || type === 'clear_cache_result') {
        const promiseCallbacks = pendingRequests.get(id);
        if (promiseCallbacks) {
          if (type === 'embedding_error') {
            promiseCallbacks.reject(new Error(error));
          } else {
            promiseCallbacks.resolve(type === 'embedding_result' ? embedding : null);
          }
          pendingRequests.delete(id);
        }
      }
    };
  }
  return worker;
}

// ensure worker is initialized
getWorker();

export async function getEmbedding(text) {
  // Check cache first
  const cached = embeddingMemory.getCachedEmbedding(text);
  if (cached) {
    console.log(`[Cache hit] Using cached embedding for text: "${text.substring(0, 30)}..."`);
    return cached;
  }

  console.log(`Generating embedding for text: "${text.substring(0, 30)}..."`);
  
  const id = ++messageIdCounter;
  
  const promise = new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject });
  });
  
  getWorker().postMessage({ id, type: "get_embedding", text });
  
  try {
    const embedding = await promise;
    // Cache the result
    embeddingMemory.cacheEmbedding(text, embedding);
    console.log(`✓ Embedding generated (dimensions: ${embedding.length})`);
    return embedding;
  } catch (err) {
    console.error(`Failed to generate embedding:`, err.message);
    throw err;
  }
}

export function getEmbeddingSystemState() {
  return embeddingMemory.getState();
}

export function clearEmbeddingCache() {
  embeddingMemory.clear();
  if (worker) {
    const id = ++messageIdCounter;
    worker.postMessage({ id, type: "clear_cache" });
  }
  console.log("Embedding cache and model cleared");
}
