#!/bin/bash

# Run All Tests
# Executes all test suites: pytest unit/integration tests and bash test scripts

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Test results
PYTEST_PASSED=0
PYTEST_FAILED=0
BASH_TESTS_PASSED=0
BASH_TESTS_FAILED=0
TOTAL_TESTS=0
TOTAL_PASSED=0
TOTAL_FAILED=0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}\n"
}

# ============================================================================
# PHASE 1: PYTEST TESTS
# ============================================================================
print_header "Running All Tests"

print_section "Phase 1: Python Unit & Integration Tests (pytest)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Error: Virtual environment not found. Run 'python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt'${NC}"
    exit 1
fi

# Check if pytest is installed (after activating venv)
if ! "$PYTHON_CMD" -m pytest --version &> /dev/null; then
    echo -e "${YELLOW}⚠ Warning: pytest not found. Installing...${NC}"
    "$PYTHON_CMD" -m pip install -q pytest pytest-cov
fi

# Run pytest tests using python3 -m pytest (more reliable)
echo "Running pytest tests..."
if "$PYTHON_CMD" -m pytest tests/ -v --tb=short 2>&1; then
    PYTEST_PASSED=1
    echo -e "${GREEN}✓ Pytest tests passed${NC}"
else
    PYTEST_FAILED=1
    echo -e "${RED}✗ Pytest tests failed${NC}"
fi

# Get pytest test count
PYTEST_OUTPUT=$("$PYTHON_CMD" -m pytest tests/ --collect-only -q 2>/dev/null | tail -1 || echo "0 test")
if [[ $PYTEST_OUTPUT =~ ([0-9]+)\ test ]]; then
    TOTAL_TESTS=$((TOTAL_TESTS + ${BASH_REMATCH[1]}))
fi

if [ $PYTEST_PASSED -eq 1 ]; then
    TOTAL_PASSED=$((TOTAL_PASSED + 1))
else
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
fi

# ============================================================================
# PHASE 2: BASIC ENDPOINT TEST
# ============================================================================
print_section "Phase 2: Basic Endpoint Test"

if [ -f "$SCRIPT_DIR/test_cloudera_endpoint.sh" ]; then
    echo "Running basic endpoint test..."
    if "$SCRIPT_DIR/test_cloudera_endpoint.sh" 2>&1; then
        BASH_TESTS_PASSED=$((BASH_TESTS_PASSED + 1))
        echo -e "${GREEN}✓ Basic endpoint test passed${NC}"
    else
        BASH_TESTS_FAILED=$((BASH_TESTS_FAILED + 1))
        echo -e "${RED}✗ Basic endpoint test failed${NC}"
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
else
    echo -e "${YELLOW}⚠ Warning: test_cloudera_endpoint.sh not found${NC}"
fi

# ============================================================================
# PHASE 3: SECURITY & COMPLIANCE TEST
# ============================================================================
print_section "Phase 3: Security & Compliance Test"

if [ -f "$SCRIPT_DIR/test_security_compliance.sh" ]; then
    echo "Running security & compliance test..."
    if "$SCRIPT_DIR/test_security_compliance.sh" 2>&1; then
        BASH_TESTS_PASSED=$((BASH_TESTS_PASSED + 1))
        echo -e "${GREEN}✓ Security & compliance test passed${NC}"
    else
        BASH_TESTS_FAILED=$((BASH_TESTS_FAILED + 1))
        echo -e "${RED}✗ Security & compliance test failed${NC}"
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
else
    echo -e "${YELLOW}⚠ Warning: test_security_compliance.sh not found${NC}"
fi

# ============================================================================
# SUMMARY
# ============================================================================
print_header "Test Summary"

echo "Python Tests (pytest):"
if [ $PYTEST_PASSED -eq 1 ]; then
    echo -e "  ${GREEN}✓ PASSED${NC}"
else
    echo -e "  ${RED}✗ FAILED${NC}"
fi

echo ""
echo "Bash Test Scripts:"
echo "  Basic Endpoint Test: $([ $BASH_TESTS_PASSED -ge 1 ] && echo -e "${GREEN}✓ PASSED${NC}" || echo -e "${RED}✗ FAILED${NC}")"
echo "  Security & Compliance Test: $([ $BASH_TESTS_PASSED -ge 2 ] && echo -e "${GREEN}✓ PASSED${NC}" || [ $BASH_TESTS_FAILED -ge 2 ] && echo -e "${RED}✗ FAILED${NC}" || echo -e "${YELLOW}⚠ SKIPPED${NC}")"

echo ""
echo "Overall Results:"
echo "  Total Test Suites: $((PYTEST_PASSED + PYTEST_FAILED + BASH_TESTS_PASSED + BASH_TESTS_FAILED))"
echo "  Passed: $((PYTEST_PASSED + BASH_TESTS_PASSED))"
echo "  Failed: $((PYTEST_FAILED + BASH_TESTS_FAILED))"

# Determine exit code
if [ $PYTEST_FAILED -eq 0 ] && [ $BASH_TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}All tests passed! ✓${NC}\n"
    exit 0
else
    echo -e "\n${RED}${BOLD}Some tests failed. Review output above. ✗${NC}\n"
    exit 1
fi

