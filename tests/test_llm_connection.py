#!/usr/bin/env python3
"""
Quick test script to verify Cloudera LLM connection
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from agents import create_cloudera_agent

    print("=" * 70)
    print("Testing Cloudera LLM Connection")
    print("=" * 70)
    print()

    # Create agent
    print("1. Creating Cloudera agent...")
    agent = create_cloudera_agent()
    print("   ✅ Agent created successfully")
    print()

    # Check if LLM is configured
    if agent.llm_client is None:
        print("⚠️  Warning: LLM not configured")
        print("   The agent was created but LLM client is not available")
        print("   Make sure config-llm.json exists with llm_endpoint configuration")
        sys.exit(1)

    print("2. LLM client is configured")
    print(f"   Model: {agent.llm_config.model}")
    print(f"   Base URL: {agent.llm_config.base_url}")
    print()

    # Test simple query
    print("3. Testing LLM with a simple query...")
    test_query = "What is Python? Answer in one sentence."

    try:
        result = agent.answer_with_llm(
            test_query,
            top_k=0,  # No context needed for simple test
            use_context=False,
            temperature=0.7,
            max_tokens=100
        )

        print("   ✅ LLM responded successfully!")
        print()
        print("   Question:", test_query)
        print("   Answer:", result.get('answer', 'No answer returned'))
        print()
        print("=" * 70)
        print("✅ Cloudera LLM is working correctly!")
        print("=" * 70)

    except Exception as e:
        print(f"   ❌ Error testing LLM: {e}")
        print()
        print("   This might indicate:")
        print("   - Network connectivity issues")
        print("   - Invalid API key")
        print("   - Incorrect endpoint URL")
        print("   - Model not available")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Error importing agents: {e}")
    print("   Make sure you're in the virtual environment:")
    print("   source venv/bin/activate")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

