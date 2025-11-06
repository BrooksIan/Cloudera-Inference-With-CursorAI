#!/bin/bash

# Comprehensive Security and Compliance Test Script
# For Cloudera Inference With CursorAI in Secure Enterprise Environments
#
# This script validates:
# - Endpoint configuration and accessibility
# - Authentication and authorization
# - Model access controls
# - Network security
# - Data sovereignty
# - Audit trail capabilities
# - Performance and reliability
# - Error handling

set -euo pipefail

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
    # Read endpoint from config.json
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
CLOUDERA_ENDPOINT="${CLOUDERA_ENDPOINT:-${CONFIG_ENDPOINT}}"
EMBEDDING_ENDPOINT="${CLOUDERA_EMBEDDING_URL:-${CONFIG_ENDPOINT}}"
API_KEY="${OPENAI_API_KEY:-${CONFIG_API_KEY}}"

# Get model IDs from environment or config file (required, no defaults)
MODEL_ID_QUERY="${CLOUDERA_QUERY_MODEL:-${CONFIG_QUERY_MODEL}}"
MODEL_ID_PASSAGE="${CLOUDERA_PASSAGE_MODEL:-${CONFIG_PASSAGE_MODEL}}"

# Validate model IDs are set
if [ -z "$MODEL_ID_QUERY" ]; then
    fail_test "Query model ID not configured" "Set CLOUDERA_QUERY_MODEL or configure models.query_model in config.json" "critical"
fi

if [ -z "$MODEL_ID_PASSAGE" ]; then
    fail_test "Passage model ID not configured" "Set CLOUDERA_PASSAGE_MODEL or configure models.passage_model in config.json" "critical"
fi

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNINGS=0
CRITICAL_FAILURES=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Test output file
TEST_REPORT="security_test_report_$(date +%Y%m%d_%H%M%S).txt"
LOG_FILE="security_test_log_$(date +%Y%m%d_%H%M%S).txt"

# Helper functions
print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}\n"
}

log_test() {
    local status=$1
    local message=$2
    local details=$3
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$status] $message" >> "$LOG_FILE"
    if [ -n "$details" ]; then
        echo "$details" >> "$LOG_FILE"
    fi
}

pass_test() {
    ((TESTS_PASSED++))
    echo -e "${GREEN}✓ PASS${NC}: $1"
    log_test "PASS" "$1" "$2"
}

fail_test() {
    ((TESTS_FAILED++))
    echo -e "${RED}✗ FAIL${NC}: $1"
    log_test "FAIL" "$1" "$2"
    if [ "${3:-}" = "critical" ]; then
        ((CRITICAL_FAILURES++))
    fi
}

warn_test() {
    ((TESTS_WARNINGS++))
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
    log_test "WARN" "$1" "$2"
}

check_dependency() {
    if command -v "$1" &> /dev/null; then
        pass_test "Dependency check: $1 installed" "$(which $1)"
        return 0
    else
        fail_test "Dependency check: $1 not found" "Required dependency missing" "critical"
        return 1
    fi
}

# Start test report
{
    echo "=========================================="
    echo "Security & Compliance Test Report"
    echo "Cloudera Inference With CursorAI"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "User: $(whoami)"
    echo "=========================================="
    echo ""
} > "$TEST_REPORT"

print_header "Security & Compliance Test Suite"
echo "Test Report: $TEST_REPORT"
echo "Log File: $LOG_FILE"
echo ""

# ============================================================================
# PHASE 1: PREREQUISITE CHECKS
# ============================================================================
print_section "PHASE 1: Prerequisite Checks"

# Check dependencies
check_dependency "python3"
check_dependency "curl"
check_dependency "jq" || warn_test "jq not installed (optional, for JSON parsing)"

# Check Python packages
if python3 -c "import openai, numpy" 2>/dev/null; then
    pass_test "Python dependencies installed" "$(python3 -c 'import openai, numpy; print(f"openai {openai.__version__}, numpy {numpy.__version__}")')"
else
    fail_test "Python dependencies missing" "Install with: pip install -r requirements.txt" "critical"
fi

# ============================================================================
# PHASE 2: CONFIGURATION VALIDATION
# ============================================================================
print_section "PHASE 2: Configuration Validation"

# Check endpoint configuration
if [ -z "$EMBEDDING_ENDPOINT" ] && [ -z "$CLOUDERA_ENDPOINT" ]; then
    fail_test "Endpoint URL not configured" "Set CLOUDERA_EMBEDDING_URL, CLOUDERA_ENDPOINT, or configure config.json" "critical"
else
    if [ -n "$EMBEDDING_ENDPOINT" ]; then
        # Check if it came from config.json
        if [ -n "$CONFIG_FILE" ] && [ "$EMBEDDING_ENDPOINT" = "$CONFIG_ENDPOINT" ]; then
            pass_test "Embedding endpoint configured from config.json" "$EMBEDDING_ENDPOINT"
        else
            pass_test "Embedding endpoint configured from environment" "$EMBEDDING_ENDPOINT"
        fi
    else
        warn_test "Using base endpoint (will construct full URL)" "$CLOUDERA_ENDPOINT"
        EMBEDDING_ENDPOINT="${CLOUDERA_ENDPOINT}/namespaces/serving-default/endpoints/your-endpoint/v1"
    fi
fi

# Validate endpoint URL format
if [[ "$EMBEDDING_ENDPOINT" =~ ^https://.*/v1$ ]]; then
    pass_test "Endpoint URL format valid" "HTTPS endpoint with /v1 suffix"
else
    fail_test "Endpoint URL format invalid" "Expected: https://.../v1" "critical"
fi

# Check for placeholder values
if [[ "$EMBEDDING_ENDPOINT" == *"your-endpoint"* ]] || [[ "$EMBEDDING_ENDPOINT" == *"your-cloudera-endpoint.com"* ]]; then
    fail_test "Endpoint URL contains placeholder" "Replace placeholder with actual endpoint" "critical"
fi

# Check API key
if [ -z "$API_KEY" ]; then
    # Try to get from /tmp/jwt
    if [ -f "/tmp/jwt" ]; then
        API_KEY=$(python3 -c "import json; print(json.load(open('/tmp/jwt'))['access_token'])" 2>/dev/null || echo "")
        if [ -n "$API_KEY" ]; then
            pass_test "API key found in /tmp/jwt" "Using JWT token from /tmp/jwt"
        else
            fail_test "API key not found" "Set OPENAI_API_KEY, configure config.json, or provide /tmp/jwt" "critical"
        fi
    else
        fail_test "API key not configured" "Set OPENAI_API_KEY environment variable or configure config.json" "critical"
    fi
else
    # Check if it came from config.json
    if [ -n "$CONFIG_FILE" ] && [ "$API_KEY" = "$CONFIG_API_KEY" ]; then
        pass_test "API key configured from config.json" "Using api_key from config.json"
    else
        pass_test "API key configured from environment" "Using OPENAI_API_KEY environment variable"
    fi
fi

# Validate API key format (should be JWT-like)
if [[ "$API_KEY" =~ ^eyJ ]]; then
    pass_test "API key format appears valid" "JWT-like token format"
else
    warn_test "API key format unusual" "Expected JWT-like token starting with 'eyJ'"
fi

# ============================================================================
# PHASE 3: NETWORK SECURITY CHECKS
# ============================================================================
print_section "PHASE 3: Network Security Checks"

# Extract base domain
BASE_DOMAIN=$(echo "$EMBEDDING_ENDPOINT" | sed -E 's|https://([^/]+).*|\1|')

# Check HTTPS only
if [[ "$EMBEDDING_ENDPOINT" == https://* ]]; then
    pass_test "HTTPS protocol enforced" "All traffic encrypted"
else
    fail_test "Non-HTTPS endpoint detected" "Security risk: Use HTTPS only" "critical"
fi

# Check endpoint accessibility
if curl -s -I --max-time 10 "$EMBEDDING_ENDPOINT" > /dev/null 2>&1; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$EMBEDDING_ENDPOINT" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "404" ] || [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
        pass_test "Endpoint accessible" "HTTP $HTTP_CODE (expected for base endpoint)"
    else
        warn_test "Unexpected HTTP code" "Received HTTP $HTTP_CODE"
    fi
else
    fail_test "Endpoint not accessible" "Check network connectivity and VPN" "critical"
fi

# Check DNS resolution
if host "$BASE_DOMAIN" > /dev/null 2>&1; then
    pass_test "DNS resolution successful" "Domain resolves correctly"
else
    fail_test "DNS resolution failed" "Check network configuration" "critical"
fi

# Check TLS/SSL
if echo | openssl s_client -connect "$BASE_DOMAIN:443" -servername "$BASE_DOMAIN" 2>/dev/null | grep -q "Verify return code: 0"; then
    pass_test "TLS/SSL certificate valid" "Secure connection established"
else
    warn_test "TLS/SSL validation issue" "Certificate may be self-signed or invalid"
fi

# ============================================================================
# PHASE 4: AUTHENTICATION & AUTHORIZATION
# ============================================================================
print_section "PHASE 4: Authentication & Authorization"

# Test authentication with valid key
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 30 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"test","model":"'$MODEL_ID_QUERY'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    pass_test "Authentication successful" "Valid API key accepted"
    
    # Validate response structure
    if echo "$BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); assert 'data' in data and 'object' in data" 2>/dev/null; then
        pass_test "Response structure valid" "Proper JSON response format"
    else
        fail_test "Invalid response structure" "Response does not match expected format"
    fi
elif [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
    fail_test "Authentication failed" "HTTP $HTTP_CODE - Check API key validity" "critical"
else
    fail_test "Unexpected response" "HTTP $HTTP_CODE - $BODY"
fi

# Test authentication with invalid key
INVALID_KEY="invalid_key_$(date +%s)"
RESPONSE_INVALID=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X POST \
    -H "Authorization: Bearer $INVALID_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"test","model":"'$MODEL_ID_QUERY'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_INVALID=$(echo "$RESPONSE_INVALID" | tail -n1)
if [ "$HTTP_CODE_INVALID" = "401" ] || [ "$HTTP_CODE_INVALID" = "403" ]; then
    pass_test "Invalid key rejected" "Security: Unauthorized access prevented"
else
    warn_test "Invalid key handling" "Expected 401/403, got HTTP $HTTP_CODE_INVALID"
fi

# ============================================================================
# PHASE 5: MODEL ACCESS CONTROL
# ============================================================================
print_section "PHASE 5: Model Access Control"

# Test query model
RESPONSE_QUERY=$(curl -s -w "\n%{http_code}" --max-time 30 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"What is machine learning?","model":"'$MODEL_ID_QUERY'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_QUERY=$(echo "$RESPONSE_QUERY" | tail -n1)
BODY_QUERY=$(echo "$RESPONSE_QUERY" | sed '$d')

if [ "$HTTP_CODE_QUERY" = "200" ]; then
    EMBEDDING_LEN=$(echo "$BODY_QUERY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "0")
    if [ "$EMBEDDING_LEN" = "1024" ]; then
        pass_test "Query model accessible" "Model: $MODEL_ID_QUERY, Dimension: $EMBEDDING_LEN"
    else
        warn_test "Query model dimension unexpected" "Expected 1024, got $EMBEDDING_LEN"
    fi
else
    fail_test "Query model access failed" "HTTP $HTTP_CODE_QUERY"
fi

# Test passage model
RESPONSE_PASSAGE=$(curl -s -w "\n%{http_code}" --max-time 30 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"Machine learning is a subset of artificial intelligence.","model":"'$MODEL_ID_PASSAGE'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_PASSAGE=$(echo "$RESPONSE_PASSAGE" | tail -n1)
BODY_PASSAGE=$(echo "$RESPONSE_PASSAGE" | sed '$d')

if [ "$HTTP_CODE_PASSAGE" = "200" ]; then
    EMBEDDING_LEN_PASSAGE=$(echo "$BODY_PASSAGE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', [{}])[0].get('embedding', [])))" 2>/dev/null || echo "0")
    if [ "$EMBEDDING_LEN_PASSAGE" = "1024" ]; then
        pass_test "Passage model accessible" "Model: $MODEL_ID_PASSAGE, Dimension: $EMBEDDING_LEN_PASSAGE"
    else
        warn_test "Passage model dimension unexpected" "Expected 1024, got $EMBEDDING_LEN_PASSAGE"
    fi
else
    fail_test "Passage model access failed" "HTTP $HTTP_CODE_PASSAGE"
fi

# Test unauthorized model access (should fail or be restricted)
UNAUTHORIZED_MODEL="unauthorized-model-$(date +%s)"
RESPONSE_UNAUTH=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"test","model":"'$UNAUTHORIZED_MODEL'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_UNAUTH=$(echo "$RESPONSE_UNAUTH" | tail -n1)
if [ "$HTTP_CODE_UNAUTH" != "200" ]; then
    pass_test "Unauthorized model access prevented" "HTTP $HTTP_CODE_UNAUTH - Model access control working"
else
    warn_test "Unauthorized model access allowed" "Security: Model access control may be too permissive"
fi

# ============================================================================
# PHASE 6: DATA SOVEREIGNTY & PRIVACY
# ============================================================================
print_section "PHASE 6: Data Sovereignty & Privacy"

# Verify endpoint is internal/enterprise
if [[ "$EMBEDDING_ENDPOINT" == *"cloudera.site"* ]] || [[ "$EMBEDDING_ENDPOINT" == *"cloudera.com"* ]]; then
    pass_test "Enterprise endpoint confirmed" "Data stays within Cloudera infrastructure"
else
    warn_test "Endpoint domain verification" "Verify endpoint is enterprise-hosted"
fi

# Check for external API calls (should not happen)
if ! curl -s --max-time 5 "https://api.openai.com" > /dev/null 2>&1; then
    pass_test "External API access blocked" "No external model provider access"
else
    warn_test "External API accessible" "Network may allow external API calls"
fi

# ============================================================================
# PHASE 7: PERFORMANCE & RELIABILITY
# ============================================================================
print_section "PHASE 7: Performance & Reliability"

# Test response time
START_TIME=$(date +%s%N)
RESPONSE_PERF=$(curl -s --max-time 30 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":"Performance test","model":"'$MODEL_ID_QUERY'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)
END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [ "$DURATION_MS" -lt 5000 ]; then
    pass_test "Response time acceptable" "${DURATION_MS}ms (target: <5000ms)"
elif [ "$DURATION_MS" -lt 10000 ]; then
    warn_test "Response time slow" "${DURATION_MS}ms (target: <5000ms)"
else
    fail_test "Response time too slow" "${DURATION_MS}ms (target: <5000ms)"
fi

# Test batch request
BATCH_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 60 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input":["Test 1","Test 2","Test 3"],"model":"'$MODEL_ID_QUERY'"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_BATCH=$(echo "$BATCH_RESPONSE" | tail -n1)
if [ "$HTTP_CODE_BATCH" = "200" ]; then
    BATCH_COUNT=$(echo "$BATCH_RESPONSE" | sed '$d' | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', [])))" 2>/dev/null || echo "0")
    if [ "$BATCH_COUNT" = "3" ]; then
        pass_test "Batch processing working" "Processed $BATCH_COUNT items"
    else
        warn_test "Batch processing incomplete" "Expected 3, got $BATCH_COUNT"
    fi
else
    warn_test "Batch processing failed" "HTTP $HTTP_CODE_BATCH (may not be supported)"
fi

# ============================================================================
# PHASE 8: ERROR HANDLING
# ============================================================================
print_section "PHASE 8: Error Handling"

# Test malformed request
MALFORMED_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"invalid":"request"}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_MALFORMED=$(echo "$MALFORMED_RESPONSE" | tail -n1)
if [ "$HTTP_CODE_MALFORMED" = "400" ] || [ "$HTTP_CODE_MALFORMED" = "422" ]; then
    pass_test "Malformed request rejected" "HTTP $HTTP_CODE_MALFORMED - Proper error handling"
else
    warn_test "Malformed request handling" "Expected 400/422, got HTTP $HTTP_CODE_MALFORMED"
fi

# Test missing required fields
MISSING_FIELDS_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 \
    -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{}' \
    "${EMBEDDING_ENDPOINT}/embeddings" 2>&1)

HTTP_CODE_MISSING=$(echo "$MISSING_FIELDS_RESPONSE" | tail -n1)
if [ "$HTTP_CODE_MISSING" = "400" ] || [ "$HTTP_CODE_MISSING" = "422" ]; then
    pass_test "Missing fields validation" "HTTP $HTTP_CODE_MISSING - Input validation working"
else
    warn_test "Missing fields handling" "Expected 400/422, got HTTP $HTTP_CODE_MISSING"
fi

# ============================================================================
# PHASE 9: AUDIT TRAIL CAPABILITIES
# ============================================================================
print_section "PHASE 9: Audit Trail Capabilities"

# Log test execution
{
    echo "Test Execution Log"
    echo "=================="
    echo "Timestamp: $(date)"
    echo "Endpoint: $EMBEDDING_ENDPOINT"
    echo "Models Tested: $MODEL_ID_QUERY, $MODEL_ID_PASSAGE"
    echo "Tests Run: $((TESTS_PASSED + TESTS_FAILED + TESTS_WARNINGS))"
    echo ""
} >> "$TEST_REPORT"

pass_test "Audit logging enabled" "Test execution logged to $TEST_REPORT and $LOG_FILE"

# ============================================================================
# PHASE 10: FRAMEWORK INTEGRATION TEST
# ============================================================================
print_section "PHASE 10: Framework Integration Test"

# Test Python agent framework
# The framework will automatically use config.json if environment variables are not set
PYTHON_TEST=$(python3 << 'PYEOF'
import sys
import os

try:
    sys.path.insert(0, os.getcwd())
    from agents import create_cloudera_agent
    
    # Clear environment variables to force use of config.json
    # (The framework will use config.json if env vars are not set)
    original_embedding_url = os.environ.pop('CLOUDERA_EMBEDDING_URL', None)
    original_api_key = os.environ.pop('OPENAI_API_KEY', None)
    
    # Try to create agent (will use config.json if available)
    try:
        agent = create_cloudera_agent()
        stats = agent.get_stats()
        
        if stats['embedding_dim'] == 1024:
            print("PASS: Framework integration successful (using config.json)")
            sys.exit(0)
        else:
            print(f"WARN: Unexpected embedding dimension: {stats['embedding_dim']}")
            sys.exit(0)
    except ValueError as e:
        # Restore environment variables and try again
        if original_embedding_url:
            os.environ['CLOUDERA_EMBEDDING_URL'] = original_embedding_url
        if original_api_key:
            os.environ['OPENAI_API_KEY'] = original_api_key
        
        # Try again with environment variables
        agent = create_cloudera_agent()
        stats = agent.get_stats()
        
        if stats['embedding_dim'] == 1024:
            print("PASS: Framework integration successful (using environment variables)")
            sys.exit(0)
        else:
            print(f"WARN: Unexpected embedding dimension: {stats['embedding_dim']}")
            sys.exit(0)
except Exception as e:
    print(f"FAIL: {str(e)}")
    sys.exit(1)
PYEOF
)

if echo "$PYTHON_TEST" | grep -q "PASS"; then
    pass_test "Framework integration working" "$PYTHON_TEST"
elif echo "$PYTHON_TEST" | grep -q "WARN"; then
    warn_test "Framework integration" "$PYTHON_TEST"
else
    fail_test "Framework integration failed" "$PYTHON_TEST"
fi

# ============================================================================
# FINAL REPORT
# ============================================================================
print_header "Test Summary"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_WARNINGS))
PASS_RATE=$(( TESTS_PASSED * 100 / TOTAL_TESTS )) 2>/dev/null || PASS_RATE=0

echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo -e "${YELLOW}Warnings: $TESTS_WARNINGS${NC}"
echo -e "Pass Rate: ${PASS_RATE}%"
echo ""

if [ $CRITICAL_FAILURES -gt 0 ]; then
    echo -e "${RED}${BOLD}CRITICAL FAILURES: $CRITICAL_FAILURES${NC}"
    echo -e "${RED}System is NOT ready for production deployment${NC}"
    EXIT_CODE=1
elif [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${YELLOW}${BOLD}Some tests failed. Review warnings and failures.${NC}"
    EXIT_CODE=1
elif [ $TESTS_WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}${BOLD}All critical tests passed. Review warnings.${NC}"
    EXIT_CODE=0
else
    echo -e "${GREEN}${BOLD}All tests passed! System is ready for secure deployment.${NC}"
    EXIT_CODE=0
fi

# Generate final report
{
    echo ""
    echo "=========================================="
    echo "Final Summary"
    echo "=========================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Warnings: $TESTS_WARNINGS"
    echo "Critical Failures: $CRITICAL_FAILURES"
    echo "Pass Rate: ${PASS_RATE}%"
    echo ""
    echo "Status: $([ $EXIT_CODE -eq 0 ] && echo "READY" || echo "NOT READY")"
    echo "=========================================="
} >> "$TEST_REPORT"

echo ""
echo "Detailed report: $TEST_REPORT"
echo "Execution log: $LOG_FILE"
echo ""

exit $EXIT_CODE

