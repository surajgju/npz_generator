import { db } from "../db/indexeddb.js";

/**
 * Calculates cosine similarity between two numeric vectors.
 */
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Add a vector representation. Since the embeddings are saved directly 
 * within the IndexedDB record, we don't need a separate in-memory index file.
 */
export async function addVector(embedding, id) {
  console.log(`[VectorStore] Indexed vector for ID: ${id}`);
}

/**
 * Perform a semantic vector search against all stored memories.
 */
export async function searchVector(embedding, k = 5) {
  const records = await db.vectors.toArray();
  if (records.length === 0) return [];

  const scored = records.map(record => {
    const score = cosineSimilarity(embedding, record.embedding);
    return {
      id: record.id,
      score: score,
      text: record.text,
      timestamp: record.timestamp,
      importance: record.importance,
      accessCount: record.accessCount
    };
  });

  // Sort descending by similarity score
  scored.sort((a, b) => b.score - a.score);

  const topMatches = scored.slice(0, k);
  console.log("%c[VectorStore] top search candidates:", "color: #9c27b0; font-weight: bold;");
  console.table(topMatches.map(s => ({
    Score: s.score.toFixed(4),
    Text: s.text.substring(0, 60) + (s.text.length > 60 ? "..." : ""),
    ID: s.id.substring(0, 8) + "..."
  })));

  // Return the top K items
  return topMatches;
}

/**
 * Inspect in-browser store capacity
 */
export async function inspectVectorStore() {
  const count = await db.vectors.count();
  console.log(`[VectorStore] Inspection: ${count} elements stored locally.`);
  return { count };
}
