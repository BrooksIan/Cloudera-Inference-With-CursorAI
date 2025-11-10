# Cursor Integration Guide for Cloudera Agents

This guide provides step-by-step instructions for integrating Cloudera agents with Cursor IDE for a seamless development experience.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Cursor Configuration](#cursor-configuration)
3. [Using Agents in Your Code](#using-agents-in-your-code)
4. [Advanced Integration Patterns](#advanced-integration-patterns)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Quick Setup

### Prerequisites

1. **Cloudera Endpoint**: You need a Cloudera embedding endpoint URL
2. **API Key**: JWT token from Cloudera
3. **Model IDs**: Query and passage model IDs

### Step 1: Configure Your Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create config.json from template
cp config.json.example config.json

# 3. Edit config.json with your Cloudera endpoint details
# See config.json.example for structure
```

### Step 2: Verify Your Configuration

```bash
# Test your configuration
cloudera-agent health

# Or use Python
python3 -c "from agents import create_cloudera_agent; agent = create_cloudera_agent(); print(agent.health_check())"
```

---

## Cursor Configuration

### Option 0: Use Workspace Profile (Recommended for This Project)

**Best for:** Project-specific configuration that doesn't affect global Cursor settings

1. **Create Workspace Profile**
   ```bash
   # Generate workspace file with Cloudera settings
   python3 scripts/create_cursor_workspace.py
   ```

2. **Open Workspace in Cursor**
   - **File → Open Workspace from File...** → Select `ModelTesting.code-workspace`
   - Or double-click the `.code-workspace` file

3. **Benefits of Workspace Profile:**
   - ✅ Project-specific settings (doesn't affect other projects)
   - ✅ Automatically configured from `config.json`
   - ✅ Can be committed to version control (with placeholder API key)
   - ✅ Team members get same configuration
   - ✅ Easy to switch between projects

4. **Update Workspace Profile**
   ```bash
   # Re-run script after updating config.json
   python3 scripts/create_cursor_workspace.py
   ```

**Note:** The workspace file includes your API key. If committing to version control, consider:
- Using environment variables in the workspace file
- Or keeping the workspace file in `.gitignore` and providing a template

### Option 1: Configure via Cursor Settings UI

1. **Open Cursor Settings**
   - **Mac**: `Cmd + ,` or `Cursor → Settings`
   - **Windows/Linux**: `Ctrl + ,` or `Cursor → Settings`

2. **Navigate to AI Settings**
   - Go to: **Settings → Features → AI** or **Settings → AI**

3. **Disable Other Providers**
   - Turn OFF: OpenAI, Anthropic, Claude, GPT-4, etc.
   - This ensures Cursor only uses your Cloudera endpoint

4. **Enable Custom Endpoint**
   - Find: **"Custom OpenAI-compatible endpoint"** or **"OpenAI Compatible API"**
   - Enable this option

5. **Configure Endpoint**
   - **Base URL**: Your Cloudera endpoint URL
     ```
     https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint/v1
     ```
   - **API Key**: Your Cloudera API key (JWT token)
   - **Model**: Your model ID (e.g., `nvidia/nv-embedqa-e5-v5-query`)

6. **Save and Restart**
   - Save settings
   - Restart Cursor to apply changes

### Option 2: Configure via Cursor Settings JSON

1. **Open Settings JSON**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type: "Preferences: Open User Settings (JSON)"
   - Press Enter

2. **Add Configuration**

```json
{
  "cursor.ai.enabled": true,
  "cursor.ai.provider": "custom",
  "cursor.ai.customEndpoint": {
    "baseUrl": "https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint/v1",
    "apiKey": "your-api-key-here",
    "model": "nvidia/nv-embedqa-e5-v5-query"
  },
  "cursor.ai.openai.enabled": false,
  "cursor.ai.anthropic.enabled": false
}
```

**Note**: Replace with your actual endpoint URL, API key, and model ID.

### Option 3: Use Helper Script (Recommended)

We provide a helper script to configure Cursor automatically:

```bash
# Run the configuration helper
python3 scripts/configure_cursor.py

# Or manually create the configuration
python3 -c "
import json
import os

# Load your config.json
with open('config.json', 'r') as f:
    config = json.load(f)

# Create Cursor settings
cursor_settings = {
    'cursor.ai.enabled': True,
    'cursor.ai.provider': 'custom',
    'cursor.ai.customEndpoint': {
        'baseUrl': config['endpoint']['base_url'],
        'apiKey': config['api_key'],
        'model': config['models']['query_model']
    },
    'cursor.ai.openai.enabled': False,
    'cursor.ai.anthropic.enabled': False
}

# Print instructions
print('Add this to your Cursor settings.json:')
print(json.dumps(cursor_settings, indent=2))
"
```

---

## Using Agents in Your Code

### Basic Usage in Cursor

Once Cursor is configured, you can use Cloudera agents directly in your code:

```python
# In any Python file in your project
from agents import create_cloudera_agent

# Create agent (automatically uses config.json)
agent = create_cloudera_agent()

# Add your knowledge base
agent.add_knowledge([
    "Python is a high-level programming language",
    "Machine learning uses algorithms to learn from data",
    "RAG combines retrieval with generation for better answers"
])

# Query your knowledge base
result = agent.answer_with_context("What is Python?", top_k=3)
print(result['context_text'])
```

### Integration with Cursor's AI Features

#### 1. Code Completion Context

Use agents to provide context for Cursor's autocomplete:

```python
# Create a context-aware helper
from agents import create_cloudera_agent

class CodeContextHelper:
    def __init__(self):
        self.agent = create_cloudera_agent()
        # Load your codebase documentation
        self.agent.add_knowledge([
            "Our API uses REST endpoints",
            "Authentication requires Bearer tokens",
            "Error handling uses try/except blocks"
        ])
    
    def get_context(self, query: str):
        """Get relevant context for code completion"""
        result = self.agent.answer_with_context(query, top_k=3)
        return result['context_text']

# Use in your code
helper = CodeContextHelper()
context = helper.get_context("How do I authenticate API requests?")
# Use context to inform Cursor's suggestions
```

#### 2. Documentation Search

Query your internal documentation directly from Cursor:

```python
from agents import create_cloudera_agent
import os

# Initialize agent with your documentation
agent = create_cloudera_agent()

# Load documentation (you can load from files)
docs = []
for root, dirs, files in os.walk('docs'):
    for file in files:
        if file.endswith('.md'):
            with open(os.path.join(root, file), 'r') as f:
                docs.append(f.read())

agent.add_knowledge(docs)

# Now you can query your docs from Cursor
def search_docs(query: str):
    result = agent.answer_with_context(query, top_k=5)
    return result['context']

# Use in Cursor chat or code
docs = search_docs("How do I configure the endpoint?")
```

#### 3. Code Example Retrieval

Store and retrieve code examples:

```python
from agents import create_cloudera_agent

agent = create_cloudera_agent()

# Add code examples with metadata
code_examples = [
    {
        "text": """
def authenticate_api(api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    return headers
""",
        "metadata": {"type": "authentication", "language": "python"}
    },
    {
        "text": """
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    return wrapper
""",
        "metadata": {"type": "decorator", "language": "python"}
    }
]

# Add examples
agent.add_knowledge(
    [ex["text"] for ex in code_examples],
    metadata_list=[ex["metadata"] for ex in code_examples]
)

# Retrieve relevant examples
def get_code_example(pattern: str):
    result = agent.answer_with_context(f"code example for {pattern}", top_k=3)
    return result['context']

# Use in Cursor
examples = get_code_example("authentication")
```

---

## Advanced Integration Patterns

### Pattern 1: Cursor Chat Integration

Create a helper that works with Cursor's chat feature:

```python
# cursor_helper.py
from agents import create_cloudera_agent
from typing import Optional

class CursorAgentHelper:
    """Helper class for using Cloudera agents with Cursor"""
    
    def __init__(self):
        self.agent = create_cloudera_agent()
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load your project's knowledge base"""
        # Load from files, APIs, databases, etc.
        knowledge = [
            "Project uses Python 3.12",
            "We use pytest for testing",
            "API endpoints follow REST conventions"
        ]
        self.agent.add_knowledge(knowledge)
    
    def answer(self, question: str, use_llm: bool = False) -> str:
        """Answer a question using RAG"""
        if use_llm and self.agent.llm_client:
            result = self.agent.answer_with_llm(question, top_k=3)
            return result['answer']
        else:
            result = self.agent.answer_with_context(question, top_k=3)
            return result['context_text']
    
    def get_code_suggestions(self, intent: str) -> list:
        """Get code suggestions based on intent"""
        result = self.agent.answer_with_context(
            f"code example for {intent}", 
            top_k=5
        )
        return [ctx['text'] for ctx in result['context']]

# Use in your code
helper = CursorAgentHelper()

# In Cursor chat, you can reference:
# "Use helper.answer('How do I authenticate?')"
# "Get code suggestions with helper.get_code_suggestions('error handling')"
```

### Pattern 2: Project-Specific Knowledge Base

Load your entire project as a knowledge base:

```python
# project_knowledge.py
from agents import create_cloudera_agent
import os
from pathlib import Path

def load_project_knowledge(project_root: str = ".") -> None:
    """Load entire project as knowledge base"""
    agent = create_cloudera_agent()
    
    knowledge = []
    metadata = []
    
    # Load Python files
    for py_file in Path(project_root).rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        with open(py_file, 'r') as f:
            content = f.read()
            knowledge.append(content)
            metadata.append({
                "type": "code",
                "file": str(py_file),
                "language": "python"
            })
    
    # Load documentation
    for doc_file in Path(project_root).rglob("*.md"):
        with open(doc_file, 'r') as f:
            content = f.read()
            knowledge.append(content)
            metadata.append({
                "type": "documentation",
                "file": str(doc_file)
            })
    
    agent.add_knowledge(knowledge, metadata_list=metadata)
    return agent

# Use in your project
agent = load_project_knowledge()

# Now you can query your entire project
result = agent.answer_with_context(
    "How does authentication work in this project?",
    top_k=5
)
```

### Pattern 3: Real-Time Code Assistance

Create a live code assistant that works with Cursor:

```python
# code_assistant.py
from agents import create_cloudera_agent
import inspect

class CodeAssistant:
    """Real-time code assistant for Cursor"""
    
    def __init__(self):
        self.agent = create_cloudera_agent()
        self._load_patterns()
    
    def _load_patterns(self):
        """Load common code patterns"""
        patterns = [
            "Error handling: Use try/except with specific exceptions",
            "Type hints: Always use type hints for function parameters",
            "Logging: Use logger.info() for important events",
            "Testing: Write unit tests for all public functions"
        ]
        self.agent.add_knowledge(patterns)
    
    def suggest_improvements(self, code: str) -> list:
        """Suggest improvements for code"""
        result = self.agent.answer_with_context(
            f"best practices for: {code[:200]}",
            top_k=3
        )
        return [ctx['text'] for ctx in result['context']]
    
    def explain_code(self, code: str) -> str:
        """Explain what code does"""
        result = self.agent.answer_with_context(
            f"explain this code: {code}",
            top_k=2
        )
        return result['context_text']

# Use in Cursor
assistant = CodeAssistant()

# Get suggestions for your code
code = """
def process_data(data):
    return data.upper()
"""

suggestions = assistant.suggest_improvements(code)
explanation = assistant.explain_code(code)
```

---

## Troubleshooting

### Issue: Cursor Not Using Cloudera Endpoint

**Symptoms:**
- Cursor still uses OpenAI/Anthropic
- Network requests go to wrong endpoint

**Solutions:**

1. **Verify Settings**
   ```bash
   # Check Cursor settings
   # Mac: ~/Library/Application Support/Cursor/User/settings.json
   # Windows: %APPDATA%\Cursor\User\settings.json
   # Linux: ~/.config/Cursor/User/settings.json
   ```

2. **Disable All Other Providers**
   - Make sure OpenAI is disabled
   - Make sure Anthropic is disabled
   - Only custom endpoint should be enabled

3. **Restart Cursor**
   - Fully quit and restart Cursor
   - Settings changes require restart

4. **Check Network Tab**
   - Open Cursor DevTools (Help → Toggle Developer Tools)
   - Check Network tab for API calls
   - Verify requests go to your Cloudera endpoint

### Issue: Authentication Errors

**Symptoms:**
- 401 Unauthorized errors
- "Invalid API key" messages

**Solutions:**

1. **Verify API Key**
   ```bash
   # Test API key
   cloudera-agent health
   ```

2. **Check API Key Expiration**
   - JWT tokens expire
   - Get new token from Cloudera
   - Update config.json or Cursor settings

3. **Verify Endpoint URL**
   - URL must end with `/v1`
   - Check for typos
   - Verify endpoint is active in Cloudera AI Platform

### Issue: Model Not Found

**Symptoms:**
- 404 Not Found errors
- "Model not found" messages

**Solutions:**

1. **Verify Model ID**
   ```bash
   # Check available models
   curl -H "Authorization: Bearer YOUR_API_KEY" \
        https://your-endpoint.com/v1/models
   ```

2. **Check Model ID Format**
   - Must match exactly (case-sensitive)
   - Common format: `nvidia/nv-embedqa-e5-v5-query`
   - Verify in Cloudera AI Platform

3. **Verify Endpoint Type**
   - Must be an embedding endpoint
   - LLM endpoints don't support `/embeddings` path

### Issue: Slow Performance

**Symptoms:**
- Slow autocomplete
- Delayed responses

**Solutions:**

1. **Check Network Latency**
   ```bash
   # Test endpoint latency
   time curl -H "Authorization: Bearer YOUR_API_KEY" \
             -X POST https://your-endpoint.com/v1/embeddings \
             -d '{"model":"your-model","input":"test"}'
   ```

2. **Optimize Batch Size**
   - Use batch operations when possible
   - Reduce `top_k` if not needed

3. **Cache Results**
   - Cache frequently used queries
   - Store embeddings locally for static content

---

## Best Practices

### 1. Configuration Management

**✅ DO:**
- Use `config.json` for development
- Use environment variables for production
- Keep `config.json` in `.gitignore`

**❌ DON'T:**
- Commit `config.json` to version control
- Hardcode API keys in code
- Share API keys in documentation

### 2. Knowledge Base Management

**✅ DO:**
- Load knowledge base once at startup
- Use metadata for filtering
- Update knowledge base incrementally

**❌ DON'T:**
- Reload knowledge base on every query
- Store entire codebase in memory
- Mix different types of documents without metadata

### 3. Performance Optimization

**✅ DO:**
- Use batch operations for multiple documents
- Cache frequently used queries
- Set appropriate `top_k` values (3-5 for most cases)

**❌ DON'T:**
- Process documents one at a time
- Query with very large `top_k` values
- Ignore similarity scores

### 4. Error Handling

**✅ DO:**
- Handle API errors gracefully
- Log errors for debugging
- Provide user-friendly error messages

**❌ DON'T:**
- Ignore exceptions
- Expose sensitive error details
- Fail silently

### 5. Testing

**✅ DO:**
- Test agent functionality before using in Cursor
- Verify configuration with health check
- Test with sample queries

**❌ DON'T:**
- Skip testing
- Assume configuration is correct
- Ignore test failures

---

## Quick Reference

### Common Commands

```bash
# Health check
cloudera-agent health

# Test query
cloudera-agent query "your question"

# Interactive mode
cloudera-agent interactive

# Add documents and query
cloudera-agent add "document text" --query "search query"
```

### Common Code Patterns

```python
# Basic usage
from agents import create_cloudera_agent
agent = create_cloudera_agent()
agent.add_knowledge(["your docs"])
result = agent.answer_with_context("query", top_k=3)

# With LLM
result = agent.answer_with_llm("query", top_k=3, use_context=True)

# Get stats
stats = agent.get_stats()
```

### Configuration Files

- **config.json**: Main configuration (gitignored)
- **config.json.example**: Template (safe to commit)
- **Cursor settings**: `~/.config/Cursor/User/settings.json` (Mac/Linux)

---

## Next Steps

1. ✅ Configure Cursor with your Cloudera endpoint
2. ✅ Test with `cloudera-agent health`
3. ✅ Load your knowledge base
4. ✅ Start using agents in your code
5. ✅ Explore advanced patterns

For more information, see:
- [README.md](../README.md) - Main documentation
- [DEVELOPER_RECOMMENDATIONS.md](DEVELOPER_RECOMMENDATIONS.md) - Developer tips
- [VECTOR_STORE_SELECTION.md](VECTOR_STORE_SELECTION.md) - Vector store options

---

**Happy coding with Cursor and Cloudera Agents! 🚀**

