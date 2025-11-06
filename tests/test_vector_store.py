"""
Unit tests for SimpleVectorStore
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from agents import SimpleVectorStore, ClouderaEmbeddingClient, EmbeddingConfig


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client"""
    config = EmbeddingConfig(
        base_url="https://test-endpoint.com/v1",
        api_key="test-key",
        query_model="test-query-model",
        passage_model="test-passage-model"
    )
    client = Mock(spec=ClouderaEmbeddingClient)
    client.config = config

    # Mock embedding responses
    def mock_embed_query(text):
        # Return deterministic embedding based on text
        return [0.1] * 1024

    def mock_embed_passage(text):
        # Return deterministic embedding based on text
        return [0.2] * 1024

    client.embed_query = Mock(side_effect=mock_embed_query)
    client.embed_passage = Mock(side_effect=mock_embed_passage)
    client.embed_batch = Mock(side_effect=lambda texts, use_passage=False:
                              [[0.2] * 1024 for _ in texts] if use_passage
                              else [[0.1] * 1024 for _ in texts])

    return client


@pytest.fixture
def vector_store(mock_embedding_client):
    """Create a vector store instance"""
    return SimpleVectorStore(mock_embedding_client)


class TestSimpleVectorStore:
    """Test SimpleVectorStore functionality"""

    def test_add_document(self, vector_store):
        """Test adding a single document"""
        vector_store.add_document("Test document")

        assert len(vector_store.documents) == 1
        assert len(vector_store.vectors) == 1
        assert vector_store.documents[0]["text"] == "Test document"
        assert len(vector_store.vectors[0]) == 1024

    def test_add_document_with_metadata(self, vector_store):
        """Test adding document with metadata"""
        metadata = {"source": "test", "id": 1}
        vector_store.add_document("Test document", metadata=metadata)

        assert vector_store.documents[0]["metadata"] == metadata

    def test_add_documents(self, vector_store):
        """Test adding multiple documents"""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        vector_store.add_documents(texts)

        assert len(vector_store.documents) == 3
        assert len(vector_store.vectors) == 3
        assert all(doc["text"] in texts for doc in vector_store.documents)

    def test_add_documents_with_metadata(self, vector_store):
        """Test adding multiple documents with metadata"""
        texts = ["Doc 1", "Doc 2"]
        metadata_list = [{"id": 1}, {"id": 2}]
        vector_store.add_documents(texts, metadata_list=metadata_list)

        assert len(vector_store.documents) == 2
        assert vector_store.documents[0]["metadata"]["id"] == 1
        assert vector_store.documents[1]["metadata"]["id"] == 2

    def test_search_empty_store(self, vector_store):
        """Test searching empty vector store"""
        results = vector_store.search("query", top_k=5)
        assert results == []

    def test_search(self, vector_store):
        """Test searching with documents"""
        vector_store.add_documents(["Python programming", "Machine learning", "Data science"])
        results = vector_store.search("programming", top_k=2)

        assert len(results) == 2
        assert "similarity" in results[0]
        assert "text" in results[0]
        assert "metadata" in results[0]

    def test_search_top_k(self, vector_store):
        """Test top_k parameter"""
        vector_store.add_documents([f"Document {i}" for i in range(10)])
        results = vector_store.search("query", top_k=3)

        assert len(results) == 3

    def test_search_top_k_exceeds_documents(self, vector_store):
        """Test top_k larger than available documents"""
        vector_store.add_documents(["Doc 1", "Doc 2"])
        results = vector_store.search("query", top_k=10)

        assert len(results) == 2

    def test_similarity_scores(self, vector_store):
        """Test that similarity scores are between 0 and 1"""
        vector_store.add_documents(["Test document"])
        results = vector_store.search("query")

        assert len(results) > 0
        assert 0 <= results[0]["similarity"] <= 1

    def test_sorted_by_similarity(self, vector_store):
        """Test that results are sorted by similarity (descending)"""
        vector_store.add_documents(["Doc 1", "Doc 2", "Doc 3"])
        results = vector_store.search("query", top_k=3)

        if len(results) > 1:
            similarities = [r["similarity"] for r in results]
            assert similarities == sorted(similarities, reverse=True)

    def test_add_document_invalid_text(self, vector_store):
        """Test that invalid text raises ValueError"""
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            vector_store.add_document("")

        with pytest.raises(ValueError, match="text must be a non-empty string"):
            vector_store.add_document("   ")

    def test_add_documents_invalid_list(self, vector_store):
        """Test that invalid texts list raises ValueError"""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            vector_store.add_documents([])

        with pytest.raises(TypeError, match="texts must be a list"):
            vector_store.add_documents("not a list")

    def test_add_documents_metadata_length_mismatch(self, vector_store):
        """Test that metadata length mismatch raises ValueError"""
        with pytest.raises(ValueError, match="metadata_list length"):
            vector_store.add_documents(["Doc 1", "Doc 2"], metadata_list=[{"id": 1}])

    def test_search_invalid_query(self, vector_store):
        """Test that invalid query raises ValueError"""
        vector_store.add_documents(["Test document"])

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            vector_store.search("", top_k=1)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            vector_store.search("   ", top_k=1)

    def test_search_invalid_top_k(self, vector_store):
        """Test that invalid top_k raises ValueError"""
        vector_store.add_documents(["Test document"])

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            vector_store.search("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            vector_store.search("query", top_k=-1)

