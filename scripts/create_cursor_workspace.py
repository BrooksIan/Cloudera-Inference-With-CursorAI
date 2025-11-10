#!/usr/bin/env python3
"""
Create or update Cursor workspace file with Cloudera endpoint settings.

This script reads config.json and creates/updates the .code-workspace file
with Cursor AI settings specific to this project.
"""

import json
import os
import sys
from pathlib import Path


def load_config(config_path: str = None, prefer_llm: bool = True) -> dict:
    """Load configuration from config-llm.json or config.json

    Args:
        config_path: Explicit config file path, or None to auto-detect
        prefer_llm: If True, prefer config-llm.json for LLM configuration
    """
    # Auto-detect config file if not provided
    if config_path is None:
        if prefer_llm and os.path.exists("config-llm.json"):
            config_path = "config-llm.json"
            print("   Found config-llm.json (for agent window)")
        elif os.path.exists("config.json"):
            config_path = "config.json"
            print("   Found config.json")
        else:
            print("❌ Error: No config file found")
            print("   Please create config-llm.json or config.json")
            sys.exit(1)

    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found")
        if prefer_llm:
            print(f"   For agent window, create config-llm.json from config-llm.json.example")
        print(f"   Or create config.json from config.json.example")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # If using config.json and prefer_llm, try to merge with config-llm.json
        if config_path == "config.json" and prefer_llm and os.path.exists("config-llm.json"):
            try:
                with open("config-llm.json", 'r') as f:
                    llm_config = json.load(f)
                    # Merge LLM config into main config
                    config.update(llm_config)
                    print("   Merged config-llm.json into config.json")
            except Exception:
                pass  # Ignore errors merging LLM config

        return config
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading {config_path}: {e}")
        sys.exit(1)


def load_workspace(workspace_path: str = "ModelTesting.code-workspace") -> dict:
    """Load existing workspace file or create new structure"""
    if os.path.exists(workspace_path):
        try:
            with open(workspace_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Invalid JSON in {workspace_path}: {e}")
            print("   Creating new workspace file...")
            return create_default_workspace()
    else:
        return create_default_workspace()


def create_default_workspace() -> dict:
    """Create default workspace structure"""
    return {
        "folders": [
            {
                "path": ".",
                "name": "Cloudera Inference With CursorAI"
            }
        ],
        "settings": {
            "files.exclude": {
                "**/.git": True,
                "**/.DS_Store": True,
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/venv": True
            },
            "files.associations": {
                "*.sh": "shellscript"
            },
            "editor.formatOnSave": True,
            "editor.defaultFormatter": "ms-python.python",
            "[shellscript]": {
                "editor.defaultFormatter": "mkhl.shfmt",
                "editor.formatOnSave": True
            },
            "[python]": {
                "editor.defaultFormatter": "ms-python.python",
                "editor.formatOnSave": True
            },
            "files.eol": "\n",
            "files.insertFinalNewline": True,
            "files.trimTrailingWhitespace": True,
            "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
            "python.terminal.activateEnvironment": True
        },
        "extensions": {
            "recommendations": [
                "mkhl.shfmt",
                "timonwong.shellcheck",
                "ms-python.python",
                "ms-python.vscode-pylance"
            ]
        }
    }


def generate_cursor_settings(config: dict, use_llm: bool = True) -> dict:
    """Generate Cursor AI settings from config.json or config-llm.json

    Args:
        config: Configuration dictionary from config.json or config-llm.json
        use_llm: If True, prefer LLM config for chat/completions (agent window)
                 If False, use embedding config
    """
    # Try LLM config first (for agent window/chat)
    llm_endpoint = config.get('llm_endpoint', {})
    llm_base_url = llm_endpoint.get('base_url', '')
    llm_model = llm_endpoint.get('model', '')

    # Fall back to embedding config if LLM not available
    endpoint = config.get('endpoint', {})
    models = config.get('models', {})
    embedding_base_url = endpoint.get('base_url') or endpoint.get('base_endpoint', '')
    query_model = models.get('query_model', '')

    api_key = config.get('api_key', '')

    # Determine which config to use
    if use_llm and llm_base_url and llm_model:
        # Use LLM config for agent window (chat/completions)
        base_url = llm_base_url
        model = llm_model
        config_type = "LLM (for agent window/chat)"
    elif embedding_base_url and query_model:
        # Use embedding config as fallback
        base_url = embedding_base_url
        model = query_model
        config_type = "Embedding (fallback)"
        if use_llm:
            print("⚠️  Warning: LLM config not found, using embedding config")
            print("   For agent window, create config-llm.json with llm_endpoint")
    else:
        print("❌ Error: No valid endpoint configuration found")
        if use_llm:
            print("   Expected: llm_endpoint.base_url and llm_endpoint.model")
        print("   Or: endpoint.base_url and models.query_model")
        sys.exit(1)

    if not api_key:
        print("⚠️  Warning: api_key not found in config")
        print("   You'll need to add it manually to the workspace file")
        api_key = "YOUR_API_KEY_HERE"

    # Ensure base_url ends with /v1
    if not base_url.endswith('/v1'):
        if base_url.endswith('/'):
            base_url = base_url + 'v1'
        else:
            base_url = base_url + '/v1'

    print(f"   Using: {config_type}")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {model}")

    # Cursor AI settings for workspace
    cursor_settings = {
        "cursor.ai.enabled": True,
        "cursor.ai.provider": "custom",
        "cursor.ai.customEndpoint": {
            "baseUrl": base_url,
            "apiKey": api_key,
            "model": model
        },
        "cursor.ai.openai.enabled": False,
        "cursor.ai.anthropic.enabled": False
    }

    return cursor_settings


def update_workspace(workspace: dict, cursor_settings: dict) -> dict:
    """Update workspace settings with Cursor AI configuration"""
    # Merge Cursor settings into workspace settings
    if "settings" not in workspace:
        workspace["settings"] = {}

    workspace["settings"].update(cursor_settings)

    return workspace


def save_workspace(workspace: dict, workspace_path: str = "ModelTesting.code-workspace"):
    """Save workspace file"""
    try:
        with open(workspace_path, 'w') as f:
            json.dump(workspace, f, indent=2)
        print(f"✅ Successfully updated {workspace_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving {workspace_path}: {e}")
        return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("CURSOR WORKSPACE PROFILE CREATOR")
    print("="*70 + "\n")

    # Determine workspace file name and config preference
    workspace_path = sys.argv[1] if len(sys.argv) > 1 else "ModelTesting.code-workspace"
    use_llm = True  # Default to LLM config for agent window

    # Check for --embedding flag
    if len(sys.argv) > 1 and "--embedding" in sys.argv:
        use_llm = False
        sys.argv.remove("--embedding")

    print(f"📁 Workspace file: {workspace_path}")
    if use_llm:
        print("📁 Config: Auto-detecting (preferring config-llm.json for agent window)")
    else:
        print("📁 Config: Using config.json (embedding)")
    print()

    # Load config
    print("🔍 Loading configuration...")
    config = load_config(prefer_llm=use_llm)
    print("✅ Configuration loaded")

    # Load or create workspace
    print(f"\n🔍 Loading workspace file...")
    workspace = load_workspace(workspace_path)
    if os.path.exists(workspace_path):
        print("✅ Existing workspace file loaded")
    else:
        print("✅ New workspace structure created")

    # Generate Cursor settings
    print("\n🔧 Generating Cursor AI settings...")
    cursor_settings = generate_cursor_settings(config, use_llm=use_llm)
    print("✅ Cursor AI settings generated")

    # Update workspace
    print("\n🔧 Updating workspace with Cursor AI settings...")
    workspace = update_workspace(workspace, cursor_settings)
    print("✅ Workspace updated")

    # Save workspace
    print(f"\n💾 Saving workspace file...")
    if save_workspace(workspace, workspace_path):
        print("\n" + "="*70)
        print("✅ WORKSPACE PROFILE CREATED SUCCESSFULLY")
        print("="*70 + "\n")

        if use_llm:
            print("📋 Configured for Agent Window (Chat/Completions)")
            print("   Using LLM endpoint for Cursor's agent window")
        else:
            print("📋 Configured for Embeddings (RAG/Retrieval)")
            print("   Using embedding endpoint")
        print()

        print("📋 Next Steps:")
        print("1. Open the workspace file in Cursor:")
        print(f"   File → Open Workspace from File... → {workspace_path}")
        print()
        print("2. Or double-click the workspace file to open it in Cursor")
        print()
        print("3. The workspace will automatically use your Cloudera endpoint")
        print("   for all AI features in this project")
        print()
        print("4. Verify the configuration:")
        print("   - Settings → Features → AI")
        print("   - Should show your Cloudera endpoint")
        print()
        print("5. Test the agent window:")
        print("   - Open Cursor's agent window (Cmd+L or Ctrl+L)")
        print("   - Ask a question and verify it uses your Cloudera LLM")
        print()

        print("="*70 + "\n")
    else:
        print("\n❌ Failed to save workspace file")
        sys.exit(1)


if __name__ == "__main__":
    main()

