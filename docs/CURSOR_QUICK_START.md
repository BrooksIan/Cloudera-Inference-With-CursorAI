# Cursor Quick Start Guide

**Quick setup for using Cloudera Agents in Cursor IDE**

## 🚀 3-Step Setup

### Step 1: Configure Your Project

```bash
# Install dependencies
pip install -r requirements.txt

# Create config.json
cp config.json.example config.json

# Edit config.json with your Cloudera endpoint details
# - endpoint.base_url: Your Cloudera endpoint URL
# - models.query_model: Your query model ID
# - models.passage_model: Your passage model ID
# - api_key: Your Cloudera API key (JWT token)
```

### Step 2: Configure Cursor IDE

**Option A: Use Workspace Profile (Recommended)**
```bash
# Create workspace profile with Cloudera settings
python3 scripts/create_cursor_workspace.py

# Open workspace in Cursor
# File → Open Workspace from File... → ModelTesting.code-workspace
```
✅ **Benefits:** Project-specific, doesn't affect global settings, easy to share

**Option B: Use Helper Script (Global Settings)**
```bash
python3 scripts/configure_cursor.py
```
⚠️ **Note:** This affects global Cursor settings (all projects)

**Option C: Manual Configuration**
1. Open Cursor Settings: `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux)
2. Go to: **Settings → Features → AI**
3. Disable: OpenAI, Anthropic, and other providers
4. Enable: "Custom OpenAI-compatible endpoint"
5. Configure:
   - **Base URL**: Your Cloudera endpoint URL (from `config.json`)
   - **API Key**: Your Cloudera API key (from `config.json`)
   - **Model**: Your query model ID (from `config.json`)
6. Save and restart Cursor

### Step 3: Verify & Use

```bash
# Test your configuration
cloudera-agent health

# Or test in Python
python3 -c "from agents import create_cloudera_agent; agent = create_cloudera_agent(); print('✅ Configuration OK!')"
```

## 💻 Using Agents in Your Code

```python
from agents import create_cloudera_agent

# Create agent (uses config.json automatically)
agent = create_cloudera_agent()

# Add your knowledge base
agent.add_knowledge([
    "Your documentation here...",
    "Code examples...",
    "API references...",
])

# Query your knowledge base
result = agent.answer_with_context("How do I use this feature?", top_k=3)
print(result['context_text'])
```

## 🎯 Common Use Cases

### 1. Documentation Search
```python
agent = create_cloudera_agent()
agent.add_knowledge(["Your docs here..."])
result = agent.answer_with_context("How do I configure X?", top_k=3)
```

### 2. Code Example Retrieval
```python
agent = create_cloudera_agent()
agent.add_knowledge(["Code examples here..."])
result = agent.answer_with_context("code example for authentication", top_k=3)
```

### 3. RAG with LLM
```python
agent = create_cloudera_agent()
agent.add_knowledge(["Your knowledge base..."])
result = agent.answer_with_llm("What is X?", top_k=3, use_context=True)
print(result['answer'])
```

## 🔧 Troubleshooting

### Cursor Not Using Cloudera Endpoint?
- ✅ Verify settings: Settings → Features → AI
- ✅ Disable other providers (OpenAI, Anthropic)
- ✅ Restart Cursor after configuration changes

### Authentication Errors?
- ✅ Test API key: `cloudera-agent health`
- ✅ Check API key expiration (JWT tokens expire)
- ✅ Verify endpoint URL is correct

### Model Not Found?
- ✅ Verify model ID matches your Cloudera deployment
- ✅ Check endpoint supports embeddings (not just LLM)
- ✅ Ensure model ID is case-sensitive and exact

## 📚 More Information

- **Detailed Guide**: [docs/CURSOR_INTEGRATION_GUIDE.md](docs/CURSOR_INTEGRATION_GUIDE.md)
- **Main README**: [README.md](README.md)
- **Developer Tips**: [docs/DEVELOPER_RECOMMENDATIONS.md](docs/DEVELOPER_RECOMMENDATIONS.md)

## ✅ Quick Checklist

- [ ] `config.json` created and configured
- [ ] Cursor settings configured with Cloudera endpoint
- [ ] Other providers disabled in Cursor
- [ ] Cursor restarted
- [ ] Health check passes: `cloudera-agent health`
- [ ] Can use agents in code: `from agents import create_cloudera_agent`

---

**Ready to code with Cloudera Agents in Cursor! 🚀**

