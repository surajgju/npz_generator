class EmbeddingMemory {
  constructor() {
    this.embeddingCache = {};
    this.modelState = {
      loading: false,
      retries: 0
    };
    this.MAX_RETRIES = 5;
    this.RETRY_DELAY = 2000; // ms
  }

  /**
   * Get cached embedding if available
   */
  getCachedEmbedding(text) {
    const hash = this.hashText(text);
    return this.embeddingCache[hash] || null;
  }

  /**
   * Cache embedding result
   */
  cacheEmbedding(text, embedding) {
    const hash = this.hashText(text);
    this.embeddingCache[hash] = embedding;
  }

  /**
   * Check if model is currently loading
   */
  isModelLoading() {
    return this.modelState.loading;
  }

  /**
   * Set model loading state
   */
  setModelLoading(loading) {
    this.modelState.loading = loading;
  }

  /**
   * Can we retry loading the model?
   */
  canRetry() {
    return this.modelState.retries < this.MAX_RETRIES;
  }

  /**
   * Record a failed model load attempt
   */
  recordLoadFailure(error) {
    this.modelState.retries++;
    this.modelState.lastError = error;
    this.modelState.lastAttempt = Date.now();
  }

  /**
   * Reset retry count on success
   */
  recordLoadSuccess() {
    this.modelState.retries = 0;
    this.modelState.lastError = undefined;
  }

  /**
   * Should we wait before retrying?
   */
  getRetryDelay() {
    // Exponential backoff: 2s, 4s, 8s, 16s, 32s
    return this.RETRY_DELAY * Math.pow(2, Math.max(0, this.modelState.retries - 1));
  }

  /**
   * Get current state for debugging
   */
  getState() {
    return {
      cacheSizeBytes: JSON.stringify(this.embeddingCache).length,
      cachedItems: Object.keys(this.embeddingCache).length,
      modelState: this.modelState,
      maxRetries: this.MAX_RETRIES
    };
  }

  /**
   * Clear memory (useful for debugging or resetting)
   */
  clear() {
    this.embeddingCache = {};
    this.modelState = {
      loading: false,
      retries: 0
    };
  }

  /**
   * Simple hash function for text caching
   */
  hashText(text) {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return `hash_${Math.abs(hash).toString(36)}`;
  }
}

export const embeddingMemory = new EmbeddingMemory();
