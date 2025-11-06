"""
Unit tests for configuration management
"""
import pytest
import os
import json
from pathlib import Path
from agents import EmbeddingConfig, create_cloudera_agent


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass"""
    
    def test_config_creation(self):
        """Test creating a valid config"""
        config = EmbeddingConfig(
            base_url="https://test-endpoint.com/v1",
            api_key="test-key",
            query_model="test-query-model",
            passage_model="test-passage-model",
            embedding_dim=1024
        )
        assert config.base_url == "https://test-endpoint.com/v1"
        assert config.api_key == "test-key"
        assert config.query_model == "test-query-model"
        assert config.passage_model == "test-passage-model"
        assert config.embedding_dim == 1024
    
    def test_config_default_dimension(self):
        """Test default embedding dimension"""
        config = EmbeddingConfig(
            base_url="https://test-endpoint.com/v1",
            api_key="test-key",
            query_model="test-query-model",
            passage_model="test-passage-model"
        )
        assert config.embedding_dim == 1024


class TestCreateAgent:
    """Test agent creation with different configurations"""
    
    def test_create_agent_missing_endpoint(self, tmp_path, monkeypatch):
        """Test that missing endpoint raises ValueError"""
        # Remove config.json if it exists
        config_path = tmp_path / "config.json"
        if config_path.exists():
            config_path.unlink()
        
        # Clear environment variables and config file
        monkeypatch.delenv("CLOUDERA_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CLOUDERA_ENDPOINT", raising=False)
        
        # Mock config.json to not exist
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(ValueError, match="Endpoint URL required"):
                create_cloudera_agent()
        finally:
            os.chdir(original_cwd)
    
    def test_create_agent_missing_api_key(self, tmp_path, monkeypatch):
        """Test that missing API key raises ValueError"""
        # Remove config.json if it exists
        config_path = tmp_path / "config.json"
        if config_path.exists():
            config_path.unlink()
        
        # Set endpoint but not API key
        monkeypatch.setenv("CLOUDERA_EMBEDDING_URL", "https://test-endpoint.com/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        # Mock /tmp/jwt to not exist
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(ValueError, match="API key required"):
                create_cloudera_agent()
        finally:
            os.chdir(original_cwd)
    
    def test_create_agent_missing_model_ids(self, tmp_path, monkeypatch):
        """Test that missing model IDs raises ValueError"""
        # Remove config.json if it exists
        config_path = tmp_path / "config.json"
        if config_path.exists():
            config_path.unlink()
        
        # Set endpoint and API key but not model IDs
        monkeypatch.setenv("CLOUDERA_EMBEDDING_URL", "https://test-endpoint.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("CLOUDERA_QUERY_MODEL", raising=False)
        monkeypatch.delenv("CLOUDERA_PASSAGE_MODEL", raising=False)
        
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with pytest.raises(ValueError, match="Query model ID required"):
                create_cloudera_agent()
        finally:
            os.chdir(original_cwd)

