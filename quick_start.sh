#!/bin/bash
# Quick start script for Cloudera Inference With CursorAI
# This script helps developers get started quickly

set -e

echo "=========================================="
echo "Cloudera Inference With CursorAI"
echo "Quick Start Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e .
else
    pip install -r requirements.txt
fi
echo "✓ Dependencies installed"

# Check for config.json
if [ ! -f "config.json" ]; then
    echo ""
    echo "⚠️  config.json not found"
    if [ -f "config.json.example" ]; then
        echo "Creating config.json from example..."
        cp config.json.example config.json
        echo "✓ config.json created from example"
        echo ""
        echo "⚠️  IMPORTANT: Edit config.json with your Cloudera endpoint details!"
        echo "   See README.md for configuration instructions"
    else
        echo "⚠️  config.json.example not found. Please create config.json manually."
    fi
else
    echo "✓ config.json found"
fi

# Run health check if config exists
if [ -f "config.json" ]; then
    echo ""
    echo "Running health check..."
    if python3 -c "from agents import create_cloudera_agent; agent = create_cloudera_agent(); print('✓ Agent created successfully')" 2>/dev/null; then
        echo "✓ Configuration appears valid"
    else
        echo "⚠️  Configuration validation failed. Please check config.json"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config.json with your Cloudera endpoint details"
echo "  2. Run: source venv/bin/activate"
echo "  3. Test: python examples/example_agent_usage.py"
echo "  4. Or use CLI: cloudera-agent health"
echo ""
echo "For more information, see README.md"

