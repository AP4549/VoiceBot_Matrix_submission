import numpy as np  # type: ignore
import faiss  # type: ignore
import json
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class VectorMemoryManager:
    """
    Manages vector embeddings for conversation memory with semantic search capabilities.
    Supports both memory storage and retrieval based on semantic similarity.
    """
    def __init__(self, 
                 embedding_model_id: str = "amazon.titan-embed-text-v2:0",
                 bedrock_client=None,
                 vector_dim: int = 1536,
                 index_path: str = "Data/memory_index.faiss",
                 metadata_path: str = "Data/memory_metadata.json",
                 relevance_threshold: float = 0.65):
        import traceback
        try:
            self.embedding_model_id = embedding_model_id
            self.bedrock_client = bedrock_client
            self.vector_dim = vector_dim
            self.index_path = index_path
            self.metadata_path = metadata_path
            self.relevance_threshold = relevance_threshold

            if self.bedrock_client is None:
                logger.warning("No Bedrock client provided - embeddings will not be available")

            self.index = None
            self.metadata = []

            self._initialize_memory_storage()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error initializing VectorMemoryManager: {e}\n{error_details}")
            self.index = None
            self.metadata = []
            raise

    def _initialize_memory_storage(self):
        """Initialize or load existing vector storage"""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing vector index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                # Update vector_dim to match loaded index
                try:
                    self.vector_dim = self.index.d
                except Exception:
                    pass
                # Load metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded {len(self.metadata)} memory entries metadata")
            else:
                # Infer embedding dimension from the model if possible
                if self.bedrock_client:
                    test_emb = self._get_embedding("initialize_memory")
                    if test_emb is not None:
                        inferred_dim = test_emb.shape[0]
                        logger.info(f"Inferred embedding dimension: {inferred_dim}")
                        self.vector_dim = inferred_dim
                logger.info(f"Creating new HNSWFlat vector index with dimension {self.vector_dim}")
                # Initialize HNSW index
                self.index = faiss.IndexHNSWFlat(self.vector_dim, 32)
                self.index.hnsw.efConstruction = 40
                self.index.hnsw.efSearch = 16
                self.metadata = []
        except Exception as e:
            logger.error(f"Error initializing vector memory storage: {e}")
            # Fallback to a simple L2 flat index
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata = []

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text using AWS Bedrock"""
        if not self.bedrock_client:
            logger.error("Bedrock client not available for embedding generation")
            return None
        try:
            body = json.dumps({"inputText": text})
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.embedding_model_id,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            return np.array(embedding).astype('float32') if embedding else None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def store_memory(self, user_id: str, message: str, response: str, importance: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        try:
            combined_text = f"Human: {message}\nAssistant: {response}"
            embedding = self._get_embedding(combined_text)
            if embedding is None:
                return False
            embedding = embedding.reshape(1, -1)
            # Add to index
            self.index.add(embedding)

            entry_metadata = {
                "id": len(self.metadata),
                "user_id": user_id,
                "message": message,
                "response": response,
                "combined": combined_text,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "custom": metadata or {}
            }
            self.metadata.append(entry_metadata)
            # Persist memory storage
            self._save_memory()
            return True
        except Exception as e:
            import traceback
            logger.error(f"Error storing memory: {e}\n{traceback.format_exc()}")
            return False

    def retrieve_memories(self, query: str, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []

            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k * 2)
            results = []

            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]

                if idx < 0 or idx >= len(self.metadata):
                    continue

                memory = self.metadata[idx]
                if memory["user_id"] != user_id:
                    continue

                similarity_score = max(0.0, min(100.0, (1 - (dist / 2)) * 100))
                if similarity_score < self.relevance_threshold * 100:
                    continue

                memory["access_count"] += 1
                memory["last_accessed"] = datetime.now().isoformat()
                result = {**memory, "similarity": similarity_score}
                results.append(result)

                if len(results) >= top_k:
                    break

            self._save_memory()
            return results

        except Exception as e:
            import traceback
            logger.error(f"Error retrieving memories: {e}\n{traceback.format_exc()}")
            return []

    def hybrid_retrieve(self, query: str, user_id: str, recent_limit: int = 3, semantic_limit: int = 3) -> List[Dict[str, Any]]:
        try:
            semantic_matches = self.retrieve_memories(query, user_id, semantic_limit)

            recent_matches = []
            user_memories = [m for m in self.metadata if m["user_id"] == user_id]
            sorted_by_time = sorted(user_memories, key=lambda x: x["timestamp"], reverse=True)

            recent_ids = set(m["id"] for m in semantic_matches)
            for memory in sorted_by_time:
                if memory["id"] not in recent_ids and len(recent_matches) < recent_limit:
                    memory_with_recency = {**memory, "retrieval_type": "recency"}
                    recent_matches.append(memory_with_recency)

            for match in semantic_matches:
                match["retrieval_type"] = "semantic"

            combined_results = semantic_matches + recent_matches

            for result in combined_results:
                if "similarity" not in result:
                    result["similarity"] = 0

                time_diff = (datetime.now() - datetime.fromisoformat(result["timestamp"])).total_seconds()
                time_factor = max(0, min(1, 1 - (time_diff / (7 * 24 * 60 * 60))))
                importance_factor = result["importance"] * (1 + (result["access_count"] / 10))

                result["ranking_score"] = (
                    (0.4 * result["similarity"] / 100) +
                    (0.4 * time_factor) +
                    (0.2 * importance_factor)
                ) * 100

            return sorted(combined_results, key=lambda x: x["ranking_score"], reverse=True)

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.debug(f"Saved memory index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def format_memories_as_context(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""

        context = "LONG-TERM MEMORY (MOST RELEVANT PAST CONVERSATIONS):\n\n"
        for i, memory in enumerate(memories):
            relevance = memory.get("retrieval_type", "unknown")
            similarity = memory.get("similarity", 0)

            context += f"Memory {i+1} [Relevance: {relevance}, "
            if relevance == "semantic":
                context += f"Similarity: {similarity:.1f}%"
            elif relevance == "recency":
                context += f"From: {memory.get('timestamp', 'unknown')}"
            context += "]:\n"
            context += f"Human: {memory['message']}\n"
            context += f"Assistant: {memory['response']}\n\n"

        return context.strip()
