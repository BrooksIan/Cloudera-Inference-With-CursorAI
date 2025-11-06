#!/bin/bash

# Test Cloudera-hosted OpenAI endpoint
# This script tests the Cloudera endpoint to verify it's working and identify the model

set -e

# Configuration - Priority: Environment variables > config.json > defaults
# Load from config.json if it exists (look in parent directory)
if [ -f "../config.json" ]; then
    CONFIG_FILE="../config.json"
elif [ -f "config.json" ]; then
    CONFIG_FILE="config.json"
else
    CONFIG_FILE=""
fi

if [ -n "$CONFIG_FILE" ]; then
    CONFIG_ENDPOINT=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('endpoint', {}).get('base_url', '') or data.get('endpoint', {}).get('base_endpoint', ''))" 2>/dev/null || echo "")
    CONFIG_API_KEY=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('api_key', ''))" 2>/dev/null || echo "")
    CONFIG_QUERY_MODEL=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('models', {}).get('query_model', ''))" 2>/dev/null || echo "")
    CONFIG_PASSAGE_MODEL=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data.get('models', {}).get('passage_model', ''))" 2>/dev/null || echo "")
else
    CONFIG_ENDPOINT=""
    CONFIG_API_KEY=""
    CONFIG_QUERY_MODEL=""
    CONFIG_PASSAGE_MODEL=""
fi

# Use environment variables if set, otherwise fall back to config.json
CLOUDERA_ENDPOINT="${CLOUDERA_ENDPOINT:-${CONFIG_ENDPOINT:-https://your-cloudera-endpoint.com}}"
EMBEDDING_ENDPOINT="${CLOUDERA_EMBEDDING_URL:-${CONFIG_ENDPOINT:-${CLOUDERA_ENDPOINT}/namespaces/serving-default/endpoints/your-endpoint/v1}}"

# Get model IDs from environment or config file (required, no defaults)
QUERY_MODEL="${CLOUDERA_QUERY_MODEL:-${CONFIG_QUERY_MODEL}}"
PASSAGE_MODEL="${CLOUDERA_PASSAGE_MODEL:-${CONFIG_PASSAGE_MODEL}}"

# Validate model IDs are set
if [ -z "$QUERY_MODEL" ]; then
    echo -e "${RED}✗ Error: Query model ID not configured${NC}"
    echo "Set CLOUDERA_QUERY_MODEL environment variable or configure models.query_model in config.json"
    exit 1
fi

if [ -z "$PASSAGE_MODEL" ]; then
    echo -e "${RED}✗ Error: Passage model ID not configured${NC}"
    echo "Set CLOUDERA_PASSAGE_MODEL environment variable or configure models.passage_model in config.json"
    exit 1
fi

MODEL_ID="${QUERY_MODEL}"

# Try to get API key - Priority: Environment > config.json > /tmp/jwt
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -n "$CONFIG_API_KEY" ]; then
        API_KEY="$CONFIG_API_KEY"
    elif [ -f "/tmp/jwt" ]; then
        API_KEY=$(python3 -c "import json; print(json.load(open('/tmp/jwt'))['access_token'])" 2>/dev/null || echo "")
    else
        API_KEY=""
    fi
else
    API_KEY="${OPENAI_API_KEY}"
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_header "Cloudera Endpoint Test"

# Check if endpoint is configured
if [ "$CLOUDERA_ENDPOINT" = "https://your-cloudera-endpoint.com" ] || [ "$EMBEDDING_ENDPOINT" = "https://your-cloudera-endpoint.com/namespaces/serving-default/endpoints/your-endpoint/v1" ]; then
    echo -e "${RED}✗ Error: Endpoint URL not configured${NC}"
    echo ""
    echo "Please set your Cloudera endpoint URL:"
    echo "  Option 1: Configure config.json (recommended)"
    echo "    Edit config.json and set the 'endpoint.base_url' field"
    echo ""
    echo "  Option 2: Environment variable"
    echo "    export CLOUDERA_EMBEDDING_URL='https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint/v1'"
    echo ""
    echo "  Option 3: Base endpoint"
    echo "    export CLOUDERA_ENDPOINT='https://your-endpoint.com'"
    echo ""
    echo "See README.md for instructions on finding your endpoint URL."
    exit 1
fi

echo "Configuration:"
echo "  Base Endpoint: $CLOUDERA_ENDPOINT"
echo "  Embedding Endpoint: $EMBEDDING_ENDPOINT"
echo "  Model ID: $MODEL_ID"
if [ -n "$API_KEY" ]; then
    echo "  API Key: ${API_KEY:0:10}...${API_KEY: -4} (masked)"
    # Determine source
    if [ -n "$CONFIG_FILE" ] && [ "$API_KEY" = "$CONFIG_API_KEY" ]; then
        echo "  API Key Source: config.json"
    elif [ -f "/tmp/jwt" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "  API Key Source: /tmp/jwt"
    else
        echo "  API Key Source: OPENAI_API_KEY environment variable"
    fi
else
    echo "  API Key: Not set"
    echo "    Set OPENAI_API_KEY environment variable, configure config.json, or"
    echo "    Ensure /tmp/jwt exists with access_token"
fi
echo ""

# Test 1: Check endpoint accessibility
print_header "1. Endpoint Accessibility Test"
if curl -s -I "$CLOUDERA_ENDPOINT" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Endpoint is accessible${NC}"
    RESPONSE=$(curl -s -I "$CLOUDERA_ENDPOINT" 2>&1 | head -5)
    echo "  Response headers:"
    echo "$RESPONSE" | sed 's/^/    /'
else
    echo -e "${RED}✗ Endpoint is not accessible${NC}"
    exit 1
fi

# Test 2: Test Embedding Endpoint (requires API key)
print_header "2. Test Embedding Endpoint"
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}⚠ API key not set. Skipping embedding test.${NC}"
    echo "  To test with API key, run:"
    echo "    export OPENAI_API_KEY='your-api-key'"
    echo "    ./test_cloudera_endpoint.sh"
    echo "  Or ensure /tmp/jwt exists with access_token"
else
    echo "  Endpoint: $EMBEDDING_ENDPOINT"
    echo "  Model: $MODEL_ID"
    echo "  Sending test embedding request..."
    TEST_PAYLOAD='{
        "input": "This is a test sentence for embedding",
        "model": "'"$MODEL_ID"'"
    }'

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$TEST_PAYLOAD" \
        "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo -e "${GREEN}✓ Embedding request successful${NC}"
        echo "  Response:"
        echo "$BODY" | python3 -m json.tool 2>/dev/null | head -50 || echo "$BODY" | head -50

        # Extract embedding length
        EMBEDDING_LEN=$(echo "$BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "unknown")
        if [ "$EMBEDDING_LEN" != "unknown" ] && [ -n "$EMBEDDING_LEN" ]; then
            echo ""
            echo -e "${GREEN}Embedding dimension: $EMBEDDING_LEN${NC}"
        fi
    else
        echo -e "${RED}✗ Embedding request failed (HTTP $HTTP_CODE)${NC}"
        echo "  Response: $BODY" | head -20
    fi
fi

# Test 3: Test Embedding with Passage Model
print_header "3. Test Embedding (Passage Model)"
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}⚠ API key not set. Skipping passage embedding test.${NC}"
else
    PASSAGE_MODEL_ID="${PASSAGE_MODEL}"
    echo "  Model: $PASSAGE_MODEL_ID"
    echo "  Sending test passage embedding request..."
    TEST_PAYLOAD='{
        "input": "This is a test passage for embedding into the vector store",
        "model": "'"$PASSAGE_MODEL_ID"'"
    }'

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$TEST_PAYLOAD" \
        "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo -e "${GREEN}✓ Passage embedding request successful${NC}"
        echo "  Response:"
        echo "$BODY" | python3 -m json.tool 2>/dev/null | head -30 || echo "$BODY" | head -30
    else
        echo -e "${RED}✗ Passage embedding request failed (HTTP $HTTP_CODE)${NC}"
        echo "  Response: $BODY" | head -20
    fi
fi

# Test 4: Check endpoint health/status
print_header "4. Endpoint Health Check"
echo "  Checking endpoint status..."
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X GET \
    "${CLOUDERA_ENDPOINT}/health" 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    echo -e "${GREEN}✓ Health endpoint accessible${NC}"
    echo "  Response: $BODY"
else
    echo -e "${YELLOW}⚠ Health endpoint not available (HTTP $HTTP_CODE)${NC}"
fi

# Summary
print_header "Test Summary"
echo "Base Endpoint: $CLOUDERA_ENDPOINT"
echo "Embedding Endpoint: $EMBEDDING_ENDPOINT"
echo "Model: $MODEL_ID"
echo "Status: Accessible (behind Istio Envoy proxy)"
echo ""
echo "To fully test the endpoint with API key:"
echo "  1. Set OPENAI_API_KEY environment variable, or"
echo "  2. Ensure /tmp/jwt exists with access_token (for workbench)"
echo "  3. Run: ./test_cloudera_endpoint.sh"
echo ""
echo "Note: The endpoint requires authentication for embedding requests."

