"""
Integration tests for LLM endpoints in ClouderaAgent
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from agents import ClouderaAgent, EmbeddingConfig, LLMConfig, ClouderaEmbeddingClient
try:
    from openai import OpenAI, APIError
    HAS_OPENAI = True
except ImportError:
    # Fallback for testing without openai installed
    OpenAI = Mock
    APIError = Exception
    HAS_OPENAI = False


@pytest.fixture
def mock_embedding_config():
    """Create a test embedding configuration"""
    return EmbeddingConfig(
        base_url="https://test-endpoint.com/v1",
        api_key="test-key",
        query_model="test-query-model",
        passage_model="test-passage-model",
        embedding_dim=1024
    )


@pytest.fixture
def mock_llm_config():
    """Create a test LLM configuration"""
    return LLMConfig(
        base_url="https://test-llm-endpoint.com/v1",
        api_key="test-key",
        model="nvidia/llama-3.3-nemotron-super-49b-v1"
    )


@pytest.fixture
def mock_embedding_client(mock_embedding_config):
    """Create a mock embedding client"""
    client = Mock(spec=ClouderaEmbeddingClient)
    client.config = mock_embedding_config

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
def mock_llm_client(mock_llm_config):
    """Create a mock LLM client"""
    client = Mock(spec=OpenAI)

    # Mock chat completions response
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "This is a test LLM response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=mock_response)
    client.base_url = mock_llm_config.base_url

    return client


@pytest.fixture
def agent_with_llm(mock_embedding_config, mock_llm_config, mock_embedding_client):
    """Create an agent with LLM support"""
    with patch('agents.cloudera_agent.ClouderaEmbeddingClient', return_value=mock_embedding_client):
        with patch('agents.cloudera_agent.OpenAI') as mock_openai:
            # Configure the mock OpenAI to return our mock client
            mock_openai.return_value.chat.completions.create.return_value.choices = [
                Mock(message=Mock(content="Test LLM response"))
            ]
            agent = ClouderaAgent(mock_embedding_config, llm_config=mock_llm_config)
            # Replace the LLM client with our mock
            agent.llm_client = Mock()
            agent.llm_client.chat.completions.create.return_value.choices = [
                Mock(message=Mock(content="Test LLM response"))
            ]
            return agent


@pytest.fixture
def agent_without_llm(mock_embedding_config, mock_embedding_client):
    """Create an agent without LLM support"""
    with patch('agents.cloudera_agent.ClouderaEmbeddingClient', return_value=mock_embedding_client):
        return ClouderaAgent(mock_embedding_config)


class TestLLMConfig:
    """Test LLMConfig dataclass"""

    def test_llm_config_creation(self, mock_llm_config):
        """Test LLMConfig can be created"""
        assert mock_llm_config.base_url == "https://test-llm-endpoint.com/v1"
        assert mock_llm_config.api_key == "test-key"
        assert mock_llm_config.model == "nvidia/llama-3.3-nemotron-super-49b-v1"


class TestClouderaAgentWithLLM:
    """Test ClouderaAgent with LLM support"""

    def test_agent_creation_with_llm(self, agent_with_llm):
        """Test agent can be created with LLM config"""
        assert agent_with_llm is not None
        assert agent_with_llm.llm_config is not None
        assert agent_with_llm.llm_client is not None

        stats = agent_with_llm.get_stats()
        assert "llm_model" in stats
        assert "llm_endpoint" in stats
        assert stats["llm_model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"

    def test_agent_creation_without_llm(self, agent_without_llm):
        """Test agent can be created without LLM config"""
        assert agent_without_llm is not None
        assert agent_without_llm.llm_config is None
        assert agent_without_llm.llm_client is None

        stats = agent_without_llm.get_stats()
        assert "llm_model" not in stats
        assert "llm_endpoint" not in stats

    def test_answer_with_llm_requires_llm(self, agent_without_llm):
        """Test that answer_with_llm raises error if LLM not configured"""
        agent_without_llm.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="LLM not configured"):
            agent_without_llm.answer_with_llm("Test query")

    def test_answer_with_llm_with_context(self, agent_with_llm):
        """Test answer_with_llm with context retrieval (RAG)"""
        agent_with_llm.add_knowledge([
            "Python is a programming language",
            "Machine learning uses algorithms"
        ])

        result = agent_with_llm.answer_with_llm("What is Python?", top_k=2, use_context=True)

        assert "answer" in result
        assert "context" in result
        assert "query" in result
        assert "model" in result
        assert result["query"] == "What is Python?"
        assert result["model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
        assert len(result["context"]) > 0

    def test_answer_with_llm_without_context(self, agent_with_llm):
        """Test answer_with_llm without context (direct LLM query)"""
        result = agent_with_llm.answer_with_llm("What is Python?", use_context=False)

        assert "answer" in result
        assert "query" in result
        assert "model" in result
        assert len(result["context"]) == 0
        assert result["context_text"] == ""

    def test_answer_with_llm_temperature(self, agent_with_llm):
        """Test answer_with_llm with custom temperature"""
        agent_with_llm.add_knowledge(["Test document"])

        result = agent_with_llm.answer_with_llm(
            "Test query",
            temperature=0.5,
            use_context=True
        )

        assert result["temperature"] == 0.5
        # Verify temperature was passed to LLM
        agent_with_llm.llm_client.chat.completions.create.assert_called_once()
        call_kwargs = agent_with_llm.llm_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_answer_with_llm_max_tokens(self, agent_with_llm):
        """Test answer_with_llm with max_tokens parameter"""
        agent_with_llm.add_knowledge(["Test document"])

        result = agent_with_llm.answer_with_llm(
            "Test query",
            max_tokens=100,
            use_context=True
        )

        # Verify max_tokens was passed to LLM
        agent_with_llm.llm_client.chat.completions.create.assert_called_once()
        call_kwargs = agent_with_llm.llm_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    def test_answer_with_llm_invalid_query(self, agent_with_llm):
        """Test that invalid query raises ValueError"""
        agent_with_llm.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent_with_llm.answer_with_llm("", use_context=True)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent_with_llm.answer_with_llm("   ", use_context=True)

    def test_answer_with_llm_empty_knowledge_base(self, agent_with_llm):
        """Test answer_with_llm with empty knowledge base"""
        result = agent_with_llm.answer_with_llm("Test query", use_context=True)

        assert "answer" in result
        assert len(result["context"]) == 0
        assert result["context_text"] == ""

    def test_answer_with_llm_prompt_formatting(self, agent_with_llm):
        """Test that prompt is correctly formatted with context"""
        agent_with_llm.add_knowledge([
            "Python is a programming language",
            "It is known for simplicity"
        ])

        result = agent_with_llm.answer_with_llm("What is Python?", top_k=2, use_context=True)

        # Verify the prompt was formatted correctly
        agent_with_llm.llm_client.chat.completions.create.assert_called_once()
        call_args = agent_with_llm.llm_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Context:" in messages[0]["content"]
        assert "Question:" in messages[0]["content"]
        assert "What is Python?" in messages[0]["content"]

    def test_answer_with_llm_no_context_prompt(self, agent_with_llm):
        """Test that prompt without context is just the query"""
        result = agent_with_llm.answer_with_llm("What is Python?", use_context=False)

        # Verify the prompt is just the query
        agent_with_llm.llm_client.chat.completions.create.assert_called_once()
        call_args = agent_with_llm.llm_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Python?"
        assert "Context:" not in messages[0]["content"]

    def test_answer_with_llm_404_error(self, agent_with_llm):
        """Test handling of 404 errors from LLM endpoint"""
        # Import APIError from the same module the agent uses
        from agents.cloudera_agent import APIError

        # Create a proper exception class for 404 error that inherits from APIError
        class MockAPIError404(APIError):
            def __init__(self):
                # APIError constructor typically takes message, request, body
                # Try different constructor signatures
                try:
                    super().__init__(message="Not Found", request=None, body=None)
                except TypeError:
                    try:
                        super().__init__(message="Not Found")
                    except TypeError:
                        try:
                            super().__init__()
                        except TypeError:
                            pass
                self.status_code = 404
                self.message = "Not Found"

            def __str__(self):
                return "Error code: 404"

        error = MockAPIError404()
        agent_with_llm.llm_client.chat.completions.create.side_effect = error

        agent_with_llm.add_knowledge(["Test document"])

        with pytest.raises(ValueError, match="LLM endpoint not found"):
            agent_with_llm.answer_with_llm("Test query", use_context=True)

    def test_answer_with_llm_api_error(self, agent_with_llm):
        """Test handling of other API errors"""
        # Import APIError from the same module the agent uses
        from agents.cloudera_agent import APIError

        # Create a proper exception class for 500 error that inherits from APIError
        class MockAPIError500(APIError):
            def __init__(self):
                # APIError constructor typically takes message, request, body
                # Try different constructor signatures
                try:
                    super().__init__(message="Internal Server Error", request=None, body=None)
                except TypeError:
                    try:
                        super().__init__(message="Internal Server Error")
                    except TypeError:
                        try:
                            super().__init__()
                        except TypeError:
                            pass
                self.status_code = 500
                self.message = "Internal Server Error"

            def __str__(self):
                return "Error code: 500"

        error = MockAPIError500()
        agent_with_llm.llm_client.chat.completions.create.side_effect = error

        agent_with_llm.add_knowledge(["Test document"])

        # The error should be raised as-is (not converted to ValueError)
        with pytest.raises(APIError):
            agent_with_llm.answer_with_llm("Test query", use_context=True)

    def test_get_stats_with_llm(self, agent_with_llm):
        """Test get_stats includes LLM information"""
        agent_with_llm.add_knowledge(["Doc 1", "Doc 2"])
        stats = agent_with_llm.get_stats()

        assert stats["num_documents"] == 2
        assert stats["llm_model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
        assert stats["llm_endpoint"] == "https://test-llm-endpoint.com/v1"
        assert "embedding_dim" in stats
        assert "query_model" in stats

    def test_get_stats_without_llm(self, agent_without_llm):
        """Test get_stats without LLM information"""
        agent_without_llm.add_knowledge(["Doc 1"])
        stats = agent_without_llm.get_stats()

        assert stats["num_documents"] == 1
        assert "llm_model" not in stats
        assert "llm_endpoint" not in stats


class TestCreateClouderaAgentWithLLM:
    """Test create_cloudera_agent factory function with LLM support"""

    @patch('agents.cloudera_agent.ClouderaEmbeddingClient')
    @patch('agents.cloudera_agent.OpenAI')
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_create_agent_loads_llm_config(self, mock_open, mock_exists, mock_openai, mock_embedding_client):
        """Test that create_cloudera_agent loads LLM config from config-llm.json"""
        import json
        from agents import create_cloudera_agent
        from io import StringIO

        # Mock file existence
        def exists_side_effect(path):
            return path in ["config.json", "config-llm.json"]
        mock_exists.side_effect = exists_side_effect

        # Mock config files with proper context manager support
        def open_side_effect(path, mode='r'):
            if path == "config.json":
                file_content = json.dumps({
                    "endpoint": {"base_url": "https://test.com/v1"},
                    "models": {
                        "query_model": "test-query",
                        "passage_model": "test-passage"
                    },
                    "api_key": "test-key",
                    "embedding_dim": 1024
                })
            elif path == "config-llm.json":
                file_content = json.dumps({
                    "llm_endpoint": {
                        "base_url": "https://test-llm.com/v1",
                        "model": "nvidia/llama-3.3-nemotron-super-49b-v1"
                    },
                    "api_key": "test-key"
                })
            else:
                file_content = ""

            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=StringIO(file_content))
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.read = Mock(return_value=file_content)
            return mock_file

        mock_open.side_effect = open_side_effect

        # Mock OpenAI client
        mock_llm_client = Mock()
        mock_llm_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Test"))
        ]
        mock_openai.return_value = mock_llm_client

        # Provide embedding config parameters since we're mocking file operations
        # We need to set environment variables for models since they're required
        import os
        os.environ["CLOUDERA_QUERY_MODEL"] = "test-query"
        os.environ["CLOUDERA_PASSAGE_MODEL"] = "test-passage"

        agent = create_cloudera_agent(
            base_url="https://test.com/v1",
            api_key="test-key",
            use_llm=True
        )

        # Clean up
        os.environ.pop("CLOUDERA_QUERY_MODEL", None)
        os.environ.pop("CLOUDERA_PASSAGE_MODEL", None)

        assert agent is not None
        assert agent.llm_config is not None
        assert agent.llm_config.model == "nvidia/llama-3.3-nemotron-super-49b-v1"

    @patch('agents.cloudera_agent.ClouderaEmbeddingClient')
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_create_agent_without_llm_config(self, mock_open, mock_exists, mock_embedding_client):
        """Test that create_cloudera_agent works without LLM config"""
        import json
        from agents import create_cloudera_agent
        from io import StringIO

        # Mock file existence
        def exists_side_effect(path):
            return path == "config.json"
        mock_exists.side_effect = exists_side_effect

        # Mock config.json with proper context manager support
        def open_side_effect(path, mode='r'):
            if path == "config.json":
                file_content = json.dumps({
                    "endpoint": {"base_url": "https://test.com/v1"},
                    "models": {
                        "query_model": "test-query",
                        "passage_model": "test-passage"
                    },
                    "api_key": "test-key",
                    "embedding_dim": 1024
                })
            else:
                file_content = ""

            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=StringIO(file_content))
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.read = Mock(return_value=file_content)
            return mock_file

        mock_open.side_effect = open_side_effect

        # Provide embedding config parameters since we're mocking file operations
        # We need to set environment variables for models since they're required
        import os
        os.environ["CLOUDERA_QUERY_MODEL"] = "test-query"
        os.environ["CLOUDERA_PASSAGE_MODEL"] = "test-passage"

        agent = create_cloudera_agent(
            base_url="https://test.com/v1",
            api_key="test-key",
            use_llm=True
        )

        # Clean up
        os.environ.pop("CLOUDERA_QUERY_MODEL", None)
        os.environ.pop("CLOUDERA_PASSAGE_MODEL", None)

        assert agent is not None
        assert agent.llm_config is None  # No LLM config found

    @patch('agents.cloudera_agent.ClouderaEmbeddingClient')
    @patch('agents.cloudera_agent.OpenAI')
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_create_agent_llm_in_config_json(self, mock_open, mock_exists, mock_openai, mock_embedding_client):
        """Test that create_cloudera_agent loads LLM config from config.json if present"""
        import json
        from agents import create_cloudera_agent
        from io import StringIO

        # Mock file existence
        def exists_side_effect(path):
            return path == "config.json"
        mock_exists.side_effect = exists_side_effect

        # Mock config.json with LLM endpoint and proper context manager support
        def open_side_effect(path, mode='r'):
            if path == "config.json":
                file_content = json.dumps({
                    "endpoint": {"base_url": "https://test.com/v1"},
                    "models": {
                        "query_model": "test-query",
                        "passage_model": "test-passage"
                    },
                    "llm_endpoint": {
                        "base_url": "https://test-llm.com/v1",
                        "model": "nvidia/llama-3.3-nemotron-super-49b-v1"
                    },
                    "api_key": "test-key",
                    "embedding_dim": 1024
                })
            else:
                file_content = ""

            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=StringIO(file_content))
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.read = Mock(return_value=file_content)
            return mock_file

        mock_open.side_effect = open_side_effect

        # Mock OpenAI client
        mock_llm_client = Mock()
        mock_llm_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content="Test"))
        ]
        mock_openai.return_value = mock_llm_client

        # Provide embedding config parameters since we're mocking file operations
        # We need to set environment variables for models since they're required
        import os
        os.environ["CLOUDERA_QUERY_MODEL"] = "test-query"
        os.environ["CLOUDERA_PASSAGE_MODEL"] = "test-passage"

        agent = create_cloudera_agent(
            base_url="https://test.com/v1",
            api_key="test-key",
            use_llm=True
        )

        # Clean up
        os.environ.pop("CLOUDERA_QUERY_MODEL", None)
        os.environ.pop("CLOUDERA_PASSAGE_MODEL", None)

        assert agent is not None
        assert agent.llm_config is not None
        assert agent.llm_config.model == "nvidia/llama-3.3-nemotron-super-49b-v1"

    @patch('agents.cloudera_agent.ClouderaEmbeddingClient')
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    def test_create_agent_use_llm_false(self, mock_open, mock_exists, mock_embedding_client):
        """Test that create_cloudera_agent skips LLM when use_llm=False"""
        import json
        from agents import create_cloudera_agent
        from io import StringIO

        # Mock file existence
        def exists_side_effect(path):
            return path in ["config.json", "config-llm.json"]
        mock_exists.side_effect = exists_side_effect

        # Mock config.json with proper context manager support
        def open_side_effect(path, mode='r'):
            if path == "config.json":
                file_content = json.dumps({
                    "endpoint": {"base_url": "https://test.com/v1"},
                    "models": {
                        "query_model": "test-query",
                        "passage_model": "test-passage"
                    },
                    "api_key": "test-key",
                    "embedding_dim": 1024
                })
            else:
                file_content = ""

            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=StringIO(file_content))
            mock_file.__exit__ = Mock(return_value=None)
            mock_file.read = Mock(return_value=file_content)
            return mock_file

        mock_open.side_effect = open_side_effect

        # Provide embedding config parameters since we're mocking file operations
        # We need to set environment variables for models since they're required
        import os
        os.environ["CLOUDERA_QUERY_MODEL"] = "test-query"
        os.environ["CLOUDERA_PASSAGE_MODEL"] = "test-passage"

        agent = create_cloudera_agent(
            base_url="https://test.com/v1",
            api_key="test-key",
            use_llm=False
        )

        # Clean up
        os.environ.pop("CLOUDERA_QUERY_MODEL", None)
        os.environ.pop("CLOUDERA_PASSAGE_MODEL", None)

        assert agent is not None
        assert agent.llm_config is None  # LLM disabled

