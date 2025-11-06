"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import json
from pathlib import Path


@pytest.fixture(scope="session")
def test_config_path(tmp_path_factory):
    """Create a temporary config.json for testing"""
    config_path = tmp_path_factory.mktemp("test_config") / "config.json"
    config_data = {
        "endpoint": {
            "base_url": "https://test-endpoint.com/v1"
        },
        "models": {
            "query_model": "test-query-model",
            "passage_model": "test-passage-model"
        },
        "api_key": "test-api-key",
        "embedding_dim": 1024
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test"""
    # Store original values
    original_env = {}
    for key in ["CLOUDERA_EMBEDDING_URL", "OPENAI_API_KEY", 
                "CLOUDERA_QUERY_MODEL", "CLOUDERA_PASSAGE_MODEL"]:
        original_env[key] = os.environ.get(key)
    
    yield
    
    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

