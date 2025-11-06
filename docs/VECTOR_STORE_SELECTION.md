# Vector Store Selection

**Overview:** Choose the right vector store based on your scale, persistence needs, and deployment environment.

## Decision Guide

| Use Case | Recommended Store | Reason |
|----------|------------------|--------|
| Prototyping, <10K docs | `SimpleVectorStore` | Fast, no setup required |
| Development, testing | `SimpleVectorStore` | Easy to reset and iterate |
| Production, <100K docs | External (Qdrant/Pinecone) | Persistence, better performance |
| Production, >100K docs | External (Qdrant/Pinecone) | Scalability, distributed search |
| Multi-user, concurrent | External (Qdrant/Pinecone) | Thread-safe, ACID guarantees |

## Example: Using External Vector Store (Qdrant)

```python
from agents import ClouderaEmbeddingClient, EmbeddingConfig
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

# Initialize embedding client
config = EmbeddingConfig(
    base_url=os.getenv("CLOUDERA_EMBEDDING_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    query_model=os.getenv("CLOUDERA_QUERY_MODEL"),
    passage_model=os.getenv("CLOUDERA_PASSAGE_MODEL"),
)
embedding_client = ClouderaEmbeddingClient(config)

# Initialize Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "documents"

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# Generate embeddings and store
documents = ["Document 1", "Document 2", ...]
embeddings = embedding_client.embed_batch(documents, use_passage=True)

# Store in Qdrant
points = [
    {
        "id": idx,
        "vector": embedding,
        "payload": {"text": doc, "metadata": {}}
    }
    for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
]
qdrant_client.upsert(collection_name=collection_name, points=points)

# Search
query_embedding = embedding_client.embed_query("search query")
results = qdrant_client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)
```

