#!/bin/bash
# Helper script to run the example agent with proper environment setup

set -e

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run 'python3 -m venv venv' first."
    exit 1
fi

source venv/bin/activate

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set in environment"
    if [ -f "/tmp/jwt" ]; then
        echo "Found /tmp/jwt file, will use that"
    else
        echo "Error: No API key found. Set OPENAI_API_KEY or ensure /tmp/jwt exists"
        exit 1
    fi
else
    echo "Using OPENAI_API_KEY from environment"
fi

# Run the example
python3 examples/example_agent_usage.py

