import { v4 as uuidv4 } from "uuid";
import { db } from "../db/indexeddb.js";
import { getEmbedding } from "../embeddings/embeddingService.js";
import { addVector, searchVector } from "../vector/vectorStore.js";

const pendingMemories = new Map();

/**
 * Inserts a new text memory into the local browser database.
 */
export async function addMemory(text) {
  if (!text || !text.trim()) {
    return { success: false, error: "Empty memory text" };
  }
  const memoryId = uuidv4();

  try {
    const embedding = await getEmbedding(text);

    const record = {
      id: memoryId,
      text: text.trim(),
      embedding,
      timestamp: Date.now(),
      importance: 1,
      accessCount: 0
    };

    await db.vectors.add(record);
    const count = await db.vectors.count();
    await addVector(embedding, count);

    // Remove from pending if it succeeded
    pendingMemories.delete(memoryId);

    console.log(`✓ Memory stored successfully: "${text.substring(0, 40)}..." [ID: ${memoryId}]`);
    return { success: true, id: memoryId };
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`Failed to add memory: ${errorMsg}`);

    // Store in pending queue for retry
    const pending = pendingMemories.get(memoryId) || {
      id: memoryId,
      text,
      attempts: 0,
      createdAt: Date.now()
    };

    pending.attempts++;
    pending.lastError = errorMsg;
    pendingMemories.set(memoryId, pending);

    return { 
      success: false, 
      id: memoryId,
      error: `Failed to process memory: ${errorMsg}`
    };
  }
}

/**
 * Searches the local database for relevant memories matching the query text.
 */
export async function semanticSearch(query) {
  if (!query || !query.trim()) return [];
  
  try {
    const embedding = await getEmbedding(query);
    const results = await searchVector(embedding, 5);
    
    // Increment access count on matches asynchronously
    for (const match of results) {
      if (match.score > 0.3) { // Only count as access if reasonably similar
        db.vectors.where("id").equals(match.id).modify(record => {
          record.accessCount = (record.accessCount || 0) + 1;
        }).catch(err => console.warn("Failed to update access count:", err));
      }
    }
    
    return results;
  } catch (err) {
    console.error("Semantic search failed:", err);
    return [];
  }
}

/**
 * Retrieve current operational status of the RAG system
 */
export function getRAGSystemStatus() {
  return {
    pendingMemoriesCount: pendingMemories.size,
    pendingItems: Array.from(pendingMemories.values())
  };
}
