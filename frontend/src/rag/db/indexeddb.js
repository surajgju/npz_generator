import Dexie from "dexie";

class RagDatabase extends Dexie {
  constructor() {
    super("rag_database");
    this.version(1).stores({
      vectors: "id,timestamp,importance,accessCount"
    });
  }
}

export const db = new RagDatabase();
