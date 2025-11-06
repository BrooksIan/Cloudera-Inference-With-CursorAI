# Testing Guide

## Test Types

This project uses **pytest** for Python testing and bash scripts for integration/end-to-end tests.

### 1. Unit Tests (`test_config.py`, `test_vector_store.py`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (milliseconds)
- **Dependencies**: None (uses mocks)
- **When to run**: Before every commit, in CI/CD

**Examples:**
- Configuration validation
- Vector store operations
- Data structure validation

### 2. Integration Tests (`test_agent.py`)
- **Purpose**: Test component interactions with mocked external APIs
- **Speed**: Fast (seconds)
- **Dependencies**: Mocked OpenAI client
- **When to run**: Before commits, in CI/CD

**Examples:**
- Agent creation and configuration
- End-to-end RAG workflows
- Error handling

### 3. End-to-End Tests (Bash scripts)
- **Purpose**: Test against real Cloudera endpoints
- **Speed**: Slow (minutes)
- **Dependencies**: Real endpoint, VPN, API keys
- **When to run**: Pre-deployment, manual validation

**Examples:**
- `test_cloudera_endpoint.sh` - Basic connectivity
- `test_security_compliance.sh` - Security validation

## Running Tests

### Run All Tests (Recommended)

Execute all test suites at once:

```bash
./tests/test_all.sh
```

This script runs:
1. All pytest unit/integration tests
2. Basic endpoint test (`test_cloudera_endpoint.sh`)
3. Security & compliance test (`test_security_compliance.sh`)

**Output:**
- Color-coded test results
- Summary of all test suites
- Exit code: `0` (all pass) or `1` (any fail)

### Run Python Tests Only

```bash
# Run all pytest tests
pytest tests/

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestEmbeddingConfig::test_config_creation

# Run with coverage
pytest --cov=agents --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run in verbose mode
pytest -v
```

### Run Bash Test Scripts

#### Basic Endpoint Test

```bash
./tests/test_cloudera_endpoint.sh
```

Tests: connectivity, authentication, model access, embedding generation.

#### Security & Compliance Test

```bash
./tests/test_security_compliance.sh
```

Validates:
- Prerequisites and dependencies
- Configuration (endpoint, API key, model IDs)
- Network security (HTTPS, DNS, TLS/SSL)
- Authentication & authorization
- Model access control
- Data sovereignty
- Performance & reliability
- Error handling
- Framework integration

**Output:**
- Test report: `security_test_report_YYYYMMDD_HHMMSS.txt`
- Execution log: `security_test_log_YYYYMMDD_HHMMSS.txt`
- Exit code: `0` (ready) or `1` (not ready)

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_config.py       # Configuration tests
├── test_vector_store.py # Vector store tests
├── test_agent.py        # Agent integration tests
├── test_all.sh          # Run all tests
├── test_cloudera_endpoint.sh
└── test_security_compliance.sh
```

## Best Practices

1. **Use Mocks for External APIs**: Never hit real endpoints in unit/integration tests
2. **Test Edge Cases**: Empty inputs, None values, invalid configurations
3. **Test Error Handling**: Verify proper exceptions are raised
4. **Keep Tests Fast**: Unit tests should run in < 1 second total
5. **Use Fixtures**: Share common test data via `conftest.py`
6. **Parametrize**: Test multiple scenarios with `@pytest.mark.parametrize`

## Example Test

```python
def test_add_document(vector_store):
    """Test adding a single document"""
    vector_store.add_document("Test document")
    
    assert len(vector_store.documents) == 1
    assert len(vector_store.vectors) == 1
    assert vector_store.documents[0]["text"] == "Test document"
```

## CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    ./tests/test_all.sh
```

Or for pytest only:

```yaml
- name: Run pytest tests
  run: |
    pip install -r requirements.txt
    pytest --cov=agents --cov-report=xml
```

## Test Configuration

Tests use `pytest.ini` for configuration:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
```

## Troubleshooting Tests

### Tests fail with "ModuleNotFoundError"
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

### Tests fail with "config.json not found"
- Unit/integration tests use mocks and don't need real config
- Bash tests look for `config.json` in parent directory
- Create `config.json` from `config.json.example` if running bash tests

### Tests fail with authentication errors
- Bash tests require real endpoint and API key
- Ensure VPN is connected (if required)
- Verify `config.json` has valid credentials

### Pytest tests fail with import errors
- Ensure you're in the project root directory
- Check that `agents/` directory is in Python path
- Verify `agents/__init__.py` exists
