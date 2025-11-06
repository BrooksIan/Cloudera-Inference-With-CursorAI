# Examples

This directory contains example scripts demonstrating how to use the Cloudera Inference With CursorAI framework.

## Overview

The examples showcase different use cases and complexity levels:
- **Basic usage**: Simple RAG operations with a knowledge base
- **Developer-focused**: Code generation and pattern search for software developers
- **Helper scripts**: Automated setup and execution

## Example Scripts

### 1. `example_agent_usage.py`

**Purpose**: Basic example demonstrating core RAG (Retrieval Augmented Generation) functionality.

**What it demonstrates**:
- Creating a Cloudera agent
- Adding documents to a knowledge base
- Querying the knowledge base with semantic search
- Retrieving relevant context based on queries

**Key Features**:
- Simple knowledge base setup
- Multiple query examples
- Similarity score display
- Basic logging configuration

**Usage**:
```bash
# From project root
python examples/example_agent_usage.py

# Or using the helper script
./examples/run_example.sh
```

**Example Output**:
- Creates an agent using configuration from `config.json` or environment variables
- Adds 7 sample documents about Python, machine learning, embeddings, etc.
- Runs 4 example queries and displays relevant results with similarity scores

**Best For**:
- First-time users learning the framework
- Understanding basic RAG concepts
- Quick testing of your configuration

---

### 2. `example_developer_usage.py`

**Purpose**: Advanced example for software developers showing code generation and pattern search.

**What it demonstrates**:
- Building a knowledge base with Python code examples
- Searching for code patterns and snippets
- Generating Python code from natural language queries
- Interactive search mode
- Comprehensive error handling

**Key Features**:
- **Code Generation**: Returns ready-to-use Python code snippets
- **Pattern Search**: Find code patterns (list comprehensions, decorators, async/await, etc.)
- **API Best Practices**: Search for REST API design patterns
- **Testing Patterns**: Find unit testing and mocking examples
- **Interactive Mode**: Search your knowledge base interactively
- **Error Handling**: Detailed error messages with troubleshooting steps

**Usage**:
```bash
# Run with example queries
python examples/example_developer_usage.py

# Run in interactive mode
python examples/example_developer_usage.py --interactive
# or
python examples/example_developer_usage.py -i
```

**Knowledge Base Contents**:
- Python code examples (list comprehensions, dictionary comprehensions, error handling, context managers, async/await, generators)
- REST API design patterns and best practices
- Testing patterns (unit tests, mocking, test organization)
- Code organization and best practices

**Example Queries**:
- "How do I use list comprehensions in Python?"
- "Show me Python code for dictionary comprehensions"
- "Generate Python code for error handling with try/except"
- "How do I use context managers in Python?"
- "Show me Python code for async/await functions"

**Output Format**:
The script returns Python code in markdown code blocks that can be copied directly into your projects:
```python
# Generated code example
squares = [x**2 for x in range(10)]
# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

**Best For**:
- Software developers building code search tools
- Creating developer documentation search systems
- Building internal code knowledge bases
- Learning Python patterns and best practices

---

### 3. `run_example.sh`

**Purpose**: Helper script to run the basic example with proper environment setup.

**What it does**:
- Activates the virtual environment
- Checks for API key configuration
- Runs `example_agent_usage.py` with proper setup

**Usage**:
```bash
# From project root
./examples/run_example.sh
```

**Prerequisites**:
- Virtual environment must exist (`venv/` directory)
- API key must be set in `OPENAI_API_KEY` environment variable or `/tmp/jwt` file

**Best For**:
- Quick testing without manual environment activation
- Automated example execution
- CI/CD pipelines

---

## Quick Start

### Prerequisites

1. **Configuration**: Ensure `config.json` is set up or environment variables are configured
   - See main [README.md](../README.md) for configuration instructions

2. **Virtual Environment**: Activate your virtual environment
   ```bash
   source venv/bin/activate
   ```

3. **Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

### Running Examples

**Option 1: Basic Example (Recommended for first-time users)**
```bash
python examples/example_agent_usage.py
```

**Option 2: Developer Example**
```bash
python examples/example_developer_usage.py
```

**Option 3: Interactive Developer Example**
```bash
python examples/example_developer_usage.py --interactive
```

**Option 4: Using Helper Script**
```bash
./examples/run_example.sh
```

## Understanding the Examples

### Basic Example Flow

1. **Create Agent**: Initialize the Cloudera agent with your configuration
2. **Add Knowledge**: Populate the knowledge base with documents
3. **Query**: Search the knowledge base with natural language queries
4. **Retrieve**: Get relevant context with similarity scores

### Developer Example Flow

1. **Create Agent**: Initialize the Cloudera agent
2. **Build Knowledge Base**: Add Python code examples, patterns, and best practices
3. **Query Code**: Search for specific code patterns or examples
4. **Extract Code**: Retrieve ready-to-use Python code snippets
5. **Interactive Search**: (Optional) Search interactively for code patterns

## Customizing Examples

### Modify Knowledge Base

Edit the `knowledge_base` list in `example_agent_usage.py` or the functions in `example_developer_usage.py` to add your own documents:

```python
knowledge_base = [
    "Your custom document 1...",
    "Your custom document 2...",
]
agent.add_knowledge(knowledge_base)
```

### Add Custom Queries

Modify the `queries` list to test with your own questions:

```python
queries = [
    "Your question 1?",
    "Your question 2?",
]
```

### Adjust Search Parameters

Modify the `top_k` parameter to retrieve more or fewer results:

```python
result = agent.answer_with_context(query, top_k=5)  # Get top 5 results
```

## Troubleshooting

### Import Errors

If you see import errors:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Verify installation: `python -c 'from agents import create_cloudera_agent'`

### Configuration Errors

If you see configuration errors:
1. Check `config.json` exists and is properly formatted
2. Verify environment variables are set (if using them)
3. See main [README.md](../README.md) for configuration details

### API Errors

**404 Not Found**:
- Verify endpoint URL in Cloudera AI Platform console
- Check model IDs match your endpoint configuration

**401 Authentication Failed**:
- Verify API key is valid and not expired
- Update `config.json` or environment variables with new token

**Connection Errors**:
- Check network connectivity
- Verify VPN is connected (if required)
- Check firewall rules

## Next Steps

After running the examples:

1. **Explore the Code**: Review the example scripts to understand the API
2. **Customize**: Modify examples to match your use case
3. **Build Your Own**: Create your own scripts using the patterns shown
4. **Read Documentation**: See main [README.md](../README.md) for comprehensive documentation
5. **Check Best Practices**: See [docs/DEVELOPER_RECOMMENDATIONS.md](../docs/DEVELOPER_RECOMMENDATIONS.md) for development tips

## Additional Resources

- **Main Documentation**: [README.md](../README.md)
- **Developer Guide**: [docs/DEVELOPER_RECOMMENDATIONS.md](../docs/DEVELOPER_RECOMMENDATIONS.md)
- **Vector Store Guide**: [docs/VECTOR_STORE_SELECTION.md](../docs/VECTOR_STORE_SELECTION.md)
- **CLI Tool**: `cloudera-agent --help`

## Example Use Cases

### Use Case 1: Documentation Search
Use the basic example pattern to build a documentation search system:
- Add your documentation as knowledge base
- Query with natural language questions
- Retrieve relevant documentation sections

### Use Case 2: Code Knowledge Base
Use the developer example pattern to build an internal code knowledge base:
- Add code examples and patterns
- Search for specific implementations
- Retrieve ready-to-use code snippets

### Use Case 3: Q&A System
Combine examples with an LLM to build a Q&A system:
- Use `answer_with_context()` to retrieve relevant context
- Send context to an LLM for answer generation
- Build a complete RAG pipeline

## Contributing

When adding new examples:
1. Follow the existing code style and structure
2. Include comprehensive docstrings
3. Add error handling
4. Update this README with example description
5. Test with various configurations

