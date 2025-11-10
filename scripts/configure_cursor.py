#!/usr/bin/env python3
"""
Helper script to configure Cursor IDE with Cloudera endpoint settings.

This script reads your config.json and generates Cursor settings configuration.
"""

import json
import os
import sys
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from config.json"""
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found")
        print(f"   Please create {config_path} from config.json.example")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading {config_path}: {e}")
        sys.exit(1)


def get_cursor_settings_path() -> Path:
    """Get path to Cursor settings.json file"""
    system = os.name
    
    if system == 'nt':  # Windows
        settings_path = Path(os.getenv('APPDATA')) / 'Cursor' / 'User' / 'settings.json'
    elif sys.platform == 'darwin':  # macOS
        settings_path = Path.home() / 'Library' / 'Application Support' / 'Cursor' / 'User' / 'settings.json'
    else:  # Linux
        settings_path = Path.home() / '.config' / 'Cursor' / 'User' / 'settings.json'
    
    return settings_path


def generate_cursor_settings(config: dict) -> dict:
    """Generate Cursor settings from config.json"""
    endpoint = config.get('endpoint', {})
    models = config.get('models', {})
    api_key = config.get('api_key', '')
    
    base_url = endpoint.get('base_url') or endpoint.get('base_endpoint', '')
    query_model = models.get('query_model', '')
    
    if not base_url:
        print("❌ Error: endpoint.base_url not found in config.json")
        sys.exit(1)
    
    if not query_model:
        print("❌ Error: models.query_model not found in config.json")
        sys.exit(1)
    
    if not api_key:
        print("⚠️  Warning: api_key not found in config.json")
        print("   You'll need to add it manually to Cursor settings")
    
    # Ensure base_url ends with /v1
    if not base_url.endswith('/v1'):
        if base_url.endswith('/'):
            base_url = base_url + 'v1'
        else:
            base_url = base_url + '/v1'
    
    cursor_settings = {
        "cursor.ai.enabled": True,
        "cursor.ai.provider": "custom",
        "cursor.ai.customEndpoint": {
            "baseUrl": base_url,
            "apiKey": api_key if api_key else "YOUR_API_KEY_HERE",
            "model": query_model
        },
        "cursor.ai.openai.enabled": False,
        "cursor.ai.anthropic.enabled": False
    }
    
    return cursor_settings


def print_instructions(cursor_settings: dict, settings_path: Path):
    """Print instructions for configuring Cursor"""
    print("\n" + "="*70)
    print("CURSOR CONFIGURATION INSTRUCTIONS")
    print("="*70 + "\n")
    
    print("📋 Option 1: Manual Configuration (Recommended)")
    print("-" * 70)
    print("1. Open Cursor Settings:")
    print("   - Press Cmd+, (Mac) or Ctrl+, (Windows/Linux)")
    print("   - Or: Cursor → Settings → Features → AI")
    print()
    print("2. Disable other providers:")
    print("   - Turn OFF: OpenAI, Anthropic, Claude, etc.")
    print()
    print("3. Enable custom endpoint:")
    print("   - Enable 'Custom OpenAI-compatible endpoint'")
    print()
    print("4. Configure endpoint:")
    print(f"   - Base URL: {cursor_settings['cursor.ai.customEndpoint']['baseUrl']}")
    print(f"   - API Key: {cursor_settings['cursor.ai.customEndpoint']['apiKey']}")
    print(f"   - Model: {cursor_settings['cursor.ai.customEndpoint']['model']}")
    print()
    print("5. Save and restart Cursor")
    print()
    
    print("📋 Option 2: Edit settings.json directly")
    print("-" * 70)
    print(f"1. Open: {settings_path}")
    print()
    print("2. Add or update these settings:")
    print()
    print(json.dumps(cursor_settings, indent=2))
    print()
    print("3. Save and restart Cursor")
    print()
    
    print("📋 Option 3: Use this script to update settings.json")
    print("-" * 70)
    print("⚠️  This will modify your Cursor settings.json file")
    response = input("Do you want to automatically update settings.json? (y/N): ")
    
    if response.lower() == 'y':
        update_settings_file(cursor_settings, settings_path)
    else:
        print("\n✅ Configuration ready. Follow Option 1 or 2 above.")


def update_settings_file(cursor_settings: dict, settings_path: Path):
    """Update Cursor settings.json file"""
    # Create directory if it doesn't exist
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing settings or create new
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                existing_settings = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Warning: {settings_path} contains invalid JSON")
            print("   Creating backup and starting fresh...")
            backup_path = settings_path.with_suffix('.json.backup')
            settings_path.rename(backup_path)
            existing_settings = {}
    else:
        existing_settings = {}
    
    # Merge settings
    existing_settings.update(cursor_settings)
    
    # Write updated settings
    try:
        with open(settings_path, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        print(f"\n✅ Successfully updated {settings_path}")
        print("   Please restart Cursor for changes to take effect")
    except Exception as e:
        print(f"\n❌ Error updating {settings_path}: {e}")
        print("   Please use Option 1 or 2 instead")


def verify_configuration(config: dict):
    """Verify configuration is complete"""
    print("🔍 Verifying configuration...")
    
    issues = []
    
    if not config.get('endpoint', {}).get('base_url'):
        issues.append("Missing endpoint.base_url")
    
    if not config.get('models', {}).get('query_model'):
        issues.append("Missing models.query_model")
    
    if not config.get('api_key'):
        issues.append("Missing api_key (will need to add manually)")
    
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    else:
        print("✅ Configuration looks good!")
        print()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("CURSOR CONFIGURATION HELPER")
    print("="*70 + "\n")
    
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = load_config(config_path)
    
    # Verify configuration
    verify_configuration(config)
    
    # Generate Cursor settings
    cursor_settings = generate_cursor_settings(config)
    
    # Get settings path
    settings_path = get_cursor_settings_path()
    
    # Print instructions
    print_instructions(cursor_settings, settings_path)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Configure Cursor using one of the options above")
    print("2. Restart Cursor")
    print("3. Test with: cloudera-agent health")
    print("4. Start using Cloudera agents in your code!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

