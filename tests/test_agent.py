"""
Integration tests for ClouderaAgent (with mocked API calls)
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from agents import ClouderaAgent, EmbeddingConfig, ClouderaEmbeddingClient


@pytest.fixture
def mock_config():
    """Create a test configuration"""
    return EmbeddingConfig(
        base_url="https://test-endpoint.com/v1",
        api_key="test-key",
        query_model="test-query-model",
        passage_model="test-passage-model",
        embedding_dim=1024
    )


@pytest.fixture
def mock_embedding_client(mock_config):
    """Create a mock embedding client"""
    client = Mock(spec=ClouderaEmbeddingClient)
    client.config = mock_config

    # Create deterministic embeddings for testing
    def create_embedding(text, model_type="query"):
        # Simple hash-based embedding for deterministic results
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        base = 0.1 if model_type == "query" else 0.2
        return [base + (hash_val % 100) / 10000.0] * 1024

    client.embed_query = Mock(side_effect=lambda text: create_embedding(text, "query"))
    client.embed_passage = Mock(side_effect=lambda text: create_embedding(text, "passage"))
    client.embed_batch = Mock(side_effect=lambda texts, use_passage=False:
                              [create_embedding(t, "passage" if use_passage else "query")
                               for t in texts])

    return client


@pytest.fixture
def agent(mock_config, mock_embedding_client):
    """Create an agent with mocked client"""
    with patch('agents.cloudera_agent.ClouderaEmbeddingClient', return_value=mock_embedding_client):
        return ClouderaAgent(mock_config)


class TestClouderaAgent:
    """Test ClouderaAgent functionality"""

    def test_agent_creation(self, agent):
        """Test agent can be created"""
        assert agent is not None
        stats = agent.get_stats()
        assert "embedding_dim" in stats
        assert "num_documents" in stats

    def test_add_knowledge_single(self, agent):
        """Test adding single document"""
        agent.add_knowledge(["Test document"])

        stats = agent.get_stats()
        assert stats["num_documents"] == 1

    def test_add_knowledge_multiple(self, agent):
        """Test adding multiple documents"""
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        agent.add_knowledge(docs)

        stats = agent.get_stats()
        assert stats["num_documents"] == 3

    def test_add_knowledge_with_metadata(self, agent):
        """Test adding documents with metadata"""
        docs = ["Doc 1", "Doc 2"]
        metadata = [{"id": 1}, {"id": 2}]
        agent.add_knowledge(docs, metadata_list=metadata)

        stats = agent.get_stats()
        assert stats["num_documents"] == 2

    def test_retrieve_context(self, agent):
        """Test retrieving context"""
        agent.add_knowledge(["Python is a programming language", "Machine learning uses algorithms"])
        results = agent.retrieve_context("What is Python?", top_k=1)

        # retrieve_context returns a list directly, not a dict
        assert isinstance(results, list)
        assert len(results) == 1
        assert "similarity" in results[0]
        assert "text" in results[0]

    def test_retrieve_context_empty(self, agent):
        """Test retrieving context from empty knowledge base"""
        results = agent.retrieve_context("query", top_k=5)

        # retrieve_context returns a list directly
        assert isinstance(results, list)
        assert len(results) == 0

    def test_answer_with_context(self, agent):
        """Test answer_with_context method"""
        agent.add_knowledge([
            "Python is a high-level programming language",
            "Machine learning is a subset of AI"
        ])
        result = agent.answer_with_context("What is Python?", top_k=2)

        assert "context" in result
        assert "context_text" in result
        assert "num_results" in result
        assert result["num_results"] > 0

    def test_get_stats(self, agent):
        """Test get_stats method"""
        agent.add_knowledge(["Doc 1", "Doc 2"])
        stats = agent.get_stats()

        assert stats["num_documents"] == 2
        assert stats["embedding_dim"] == 1024
        assert "query_model" in stats
        assert "passage_model" in stats

    def test_top_k_parameter(self, agent):
        """Test top_k parameter works correctly"""
        agent.add_knowledge([f"Document {i}" for i in range(10)])
        results = agent.retrieve_context("query", top_k=3)

        # retrieve_context returns a list directly
        assert isinstance(results, list)
        assert len(results) == 3

    def test_similarity_ordering(self, agent):
        """Test that results are ordered by similarity"""
        agent.add_knowledge([
            "Python programming language",
            "Machine learning algorithms",
            "Data science techniques"
        ])
        results = agent.retrieve_context("Python", top_k=3)

        # retrieve_context returns a list directly
        assert isinstance(results, list)
        if len(results) > 1:
            similarities = [r["similarity"] for r in results]
            assert similarities == sorted(similarities, reverse=True)

    def test_retrieve_context_invalid_query(self, agent):
        """Test that invalid query raises ValueError"""
        agent.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.retrieve_context("", top_k=1)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.retrieve_context("   ", top_k=1)

    def test_retrieve_context_invalid_top_k(self, agent):
        """Test that invalid top_k raises ValueError"""
        agent.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            agent.retrieve_context("query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            agent.retrieve_context("query", top_k=-1)

    def test_answer_with_context_invalid_query(self, agent):
        """Test that invalid query raises ValueError"""
        agent.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.answer_with_context("", top_k=1)

    def test_add_knowledge_empty_list(self, agent):
        """Test that empty list raises ValueError"""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            agent.add_knowledge([])

