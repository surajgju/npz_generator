import { pipeline, env } from "@xenova/transformers";
import { embeddingMemory } from "./embeddingMemory.js";

// Force remote model loading in the browser
env.allowLocalModels = false;
env.allowRemoteModels = true;

// Set the correct base URL and disable automatic path resolution
env.remoteHost = 'https://huggingface.co/';

// Add cache busting to avoid stale responses
env.useBrowserCache = false;

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

let extractor = null;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

async function loadExtractor(retryCount = 0) {
  if (embeddingMemory.isModelLoading()) {
    await new Promise(resolve => setTimeout(resolve, 500));
    return extractor;
  }

  if (extractor) {
    return extractor;
  }

  embeddingMemory.setModelLoading(true);

  try {
    console.log(`Attempting to load model: ${MODEL_NAME} (attempt ${retryCount + 1})`);
    
    // Explicitly specify the model configuration
    const instance = await pipeline(
      "feature-extraction",
      MODEL_NAME,
      {
        quantized: true,
        progress_callback: (progress) => {
          console.log(`Model loading progress:`, progress);
        }
      }
    );
    
    extractor = instance;
    embeddingMemory.recordLoadSuccess();
    console.log(`✓ Embedding model loaded successfully`);
    embeddingMemory.setModelLoading(false);
    return instance;
  } catch (err) {
    embeddingMemory.setModelLoading(false);
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`Failed to load embedding model:`, errorMsg);
    embeddingMemory.recordLoadFailure(errorMsg);

    // Retry logic with exponential backoff
    if (embeddingMemory.canRetry() && retryCount < embeddingMemory.MAX_RETRIES) {
      const delay = embeddingMemory.getRetryDelay();
      console.log(`Retrying model load in ${delay}ms... (attempt ${retryCount + 2})`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return loadExtractor(retryCount + 1);
    } else {
      console.error(`Max retries exceeded for embedding model`);
      throw new Error(
        `Failed to load embedding model after ${retryCount + 1} attempts. Last error: ${errorMsg}`
      );
    }
  }
}

export async function getEmbedding(text) {
  // Check cache first
  const cached = embeddingMemory.getCachedEmbedding(text);
  if (cached) {
    console.log(`[Cache hit] Using cached embedding for text: "${text.substring(0, 30)}..."`);
    return cached;
  }

  // Load model if not already loaded
  if (!extractor) {
    extractor = await loadExtractor();
  }

  console.log(`Generating embedding for text: "${text.substring(0, 30)}..."`);
  
  const output = await extractor(text, {
    pooling: "mean",
    normalize: true
  });

  const embedding = Array.from(output.data);
  
  // Cache the result
  embeddingMemory.cacheEmbedding(text, embedding);
  
  console.log(`✓ Embedding generated (dimensions: ${embedding.length})`);
  return embedding;
}

export function getEmbeddingSystemState() {
  return embeddingMemory.getState();
}

export function clearEmbeddingCache() {
  embeddingMemory.clear();
  extractor = null;
  console.log("Embedding cache and model cleared");
}
