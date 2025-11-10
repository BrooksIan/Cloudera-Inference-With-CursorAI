# Cursor Workspace Profile Guide

This guide explains how to use Cursor workspace profiles for project-specific Cloudera agent configuration.

## What is a Workspace Profile?

A **workspace profile** is a `.code-workspace` file that contains:
- Project-specific settings
- Cursor AI configuration
- Extension recommendations
- File exclusions and formatting rules

**Benefits:**
- ✅ Project-specific (doesn't affect other projects)
- ✅ Automatically configured from `config.json`
- ✅ Easy to share with team members
- ✅ Can be version controlled (with care for API keys)

## Quick Start

### Step 1: Create Workspace Profile

```bash
# Generate workspace file with Cloudera settings
python3 scripts/create_cursor_workspace.py
```

This script will:
1. Read your `config.json`
2. Generate Cursor AI settings
3. Create or update `ModelTesting.code-workspace`
4. Include all necessary project settings

### Step 2: Open Workspace in Cursor

**Method 1: From Cursor Menu**
1. **File → Open Workspace from File...**
2. Select `ModelTesting.code-workspace`
3. Click "Open"

**Method 2: Double-Click**
- Double-click `ModelTesting.code-workspace` in Finder/File Explorer
- It will open in Cursor automatically

**Method 3: Command Line**
```bash
# Mac
open ModelTesting.code-workspace

# Linux
cursor ModelTesting.code-workspace

# Windows
cursor ModelTesting.code-workspace
```

### Step 3: Verify Configuration

1. Open Cursor Settings: `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux)
2. Go to: **Settings → Features → AI**
3. Verify:
   - Custom endpoint is enabled
   - Base URL matches your Cloudera endpoint
   - Model ID is correct
   - Other providers are disabled

## Workspace File Structure

The workspace file (`ModelTesting.code-workspace`) contains:

```json
{
  "folders": [
    {
      "path": ".",
      "name": "Cloudera Inference With CusorAI"
    }
  ],
  "settings": {
    // Cursor AI Configuration
    "cursor.ai.enabled": true,
    "cursor.ai.provider": "custom",
    "cursor.ai.customEndpoint": {
      "baseUrl": "https://your-endpoint.com/v1",
      "apiKey": "your-api-key",
      "model": "nvidia/nv-embedqa-e5-v5-query"
    },
    "cursor.ai.openai.enabled": false,
    "cursor.ai.anthropic.enabled": false,
    
    // Python Settings
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    
    // Editor Settings
    "editor.formatOnSave": true,
    "files.eol": "\n",
    "files.trimTrailingWhitespace": true
  },
  "extensions": {
    "recommendations": [
      "ms-python.python",
      "ms-python.vscode-pylance"
    ]
  }
}
```

## Updating Workspace Profile

### After Updating config.json

```bash
# Re-run the script to update workspace settings
python3 scripts/create_cursor_workspace.py
```

The script will:
- Read updated `config.json`
- Update Cursor AI settings in workspace file
- Preserve other workspace settings

### Manual Updates

You can also edit `ModelTesting.code-workspace` directly:

1. Open the file in Cursor or any text editor
2. Update the `cursor.ai.customEndpoint` settings
3. Save the file
4. Reload the workspace in Cursor

## Version Control Considerations

### Option 1: Commit Workspace File (Recommended for Teams)

**Pros:**
- Team members get same configuration
- Easy onboarding
- Consistent development environment

**Cons:**
- API key is in version control (security risk)

**Solution:** Use placeholder API key
```json
"apiKey": "YOUR_API_KEY_HERE"
```

Then team members:
1. Clone the repository
2. Create `config.json` with their API key
3. Run `python3 scripts/create_cursor_workspace.py` to update workspace

### Option 2: Don't Commit Workspace File

**Pros:**
- No API key in version control
- More secure

**Cons:**
- Each team member must create their own workspace file
- Less consistent across team

**Solution:** Add to `.gitignore`
```gitignore
# Workspace files (optional - depends on your preference)
*.code-workspace
```

## Multiple Workspace Profiles

You can create multiple workspace profiles for different configurations:

```bash
# Create workspace for development
python3 scripts/create_cursor_workspace.py dev.code-workspace config-dev.json

# Create workspace for production
python3 scripts/create_cursor_workspace.py prod.code-workspace config-prod.json
```

Then open the appropriate workspace file for your use case.

## Troubleshooting

### Workspace Settings Not Applied

**Issue:** Cursor not using workspace settings

**Solutions:**
1. **Reload Workspace**
   - Close and reopen the workspace file
   - Or: **File → Reload Window**

2. **Check Settings Priority**
   - Workspace settings override user settings
   - Verify workspace is open (check bottom-left of Cursor)

3. **Verify Workspace File**
   - Check JSON syntax is valid
   - Verify settings are in `settings` section

### API Key Not Working

**Issue:** Authentication errors with workspace API key

**Solutions:**
1. **Check API Key Expiration**
   - JWT tokens expire
   - Update `config.json` with new token
   - Re-run `create_cursor_workspace.py`

2. **Verify API Key Format**
   - Should be a JWT token
   - No extra spaces or quotes

3. **Test API Key**
   ```bash
   cloudera-agent health
   ```

### Workspace File Not Found

**Issue:** Script can't find workspace file

**Solutions:**
1. **Specify Workspace File**
   ```bash
   python3 scripts/create_cursor_workspace.py my-workspace.code-workspace
   ```

2. **Check Current Directory**
   - Run script from project root
   - Or specify full path to workspace file

## Best Practices

### 1. Keep Workspace File Updated

```bash
# Update workspace after config.json changes
python3 scripts/create_cursor_workspace.py
```

### 2. Use Environment Variables (Advanced)

For more security, you can use environment variables in workspace:

```json
"cursor.ai.customEndpoint": {
  "baseUrl": "${env:CLOUDERA_ENDPOINT_URL}",
  "apiKey": "${env:CLOUDERA_API_KEY}",
  "model": "${env:CLOUDERA_MODEL_ID}"
}
```

Then set environment variables before opening Cursor.

### 3. Share Workspace Template

Create a template workspace file without API key:

```json
"apiKey": "YOUR_API_KEY_HERE"
```

Team members can:
1. Clone repository
2. Create `config.json`
3. Run `create_cursor_workspace.py` to generate workspace with their API key

### 4. Document Workspace Usage

Add to your project README:
```markdown
## Development Setup

1. Create config.json from config.json.example
2. Generate workspace profile:
   ```bash
   python3 scripts/create_cursor_workspace.py
   ```
3. Open ModelTesting.code-workspace in Cursor
```

## Comparison: Workspace vs Global Settings

| Feature | Workspace Profile | Global Settings |
|---------|------------------|-----------------|
| **Scope** | Project-specific | All projects |
| **Sharing** | Easy (commit file) | Manual (each user) |
| **Isolation** | ✅ Doesn't affect other projects | ❌ Affects all projects |
| **Setup** | One-time per project | One-time per user |
| **Updates** | Re-run script | Manual update |
| **Version Control** | ✅ Can commit | ❌ User-specific |

**Recommendation:** Use workspace profiles for project-specific configuration.

## Next Steps

1. ✅ Create workspace profile: `python3 scripts/create_cursor_workspace.py`
2. ✅ Open workspace in Cursor
3. ✅ Verify configuration
4. ✅ Start using Cloudera agents!

For more information:
- [CURSOR_INTEGRATION_GUIDE.md](CURSOR_INTEGRATION_GUIDE.md) - Complete integration guide
- [CURSOR_QUICK_START.md](../CURSOR_QUICK_START.md) - Quick setup guide
- [README.md](../README.md) - Main documentation

---

**Happy coding with Cursor workspace profiles! 🚀**

