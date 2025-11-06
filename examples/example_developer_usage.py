#!/usr/bin/env python3
"""
Example: Using Cloudera Inference With CursorAI for Software Developers

This example demonstrates how software developers can use the framework to:
- Search code documentation and examples
- Find relevant code snippets and patterns
- Query API documentation and best practices
- Build a knowledge base from code comments and documentation
"""

import logging
import sys

# Check for required dependencies
try:
    from agents import create_cloudera_agent
except ImportError as e:
    print("="*60)
    print("ERROR: Missing required dependencies")
    print("="*60)
    print(f"\nImport error: {e}")
    print("\nSetup Instructions:")
    print("\n1. Create virtual environment (if it doesn't exist):")
    print("   python3 -m venv venv")
    print("\n2. Activate virtual environment:")
    print("   source venv/bin/activate")
    print("   # You should see (venv) in your prompt")
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("   # Make sure you're using pip from the venv, not system pip")
    print("\n4. Verify installation:")
    print("   python -c 'import openai; print(\"Dependencies OK\")'")
    print("\n5. Run the script:")
    print("   python example_developer_usage.py")
    print("\nNote: If you see 'externally-managed-environment' error,")
    print("      make sure the virtual environment is activated first!")
    print("="*60)
    sys.exit(1)

# Configure logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def setup_developer_knowledge_base(agent):
    """Add code-related documentation and examples to the knowledge base"""

    # Python code examples and patterns - with actual code snippets
    python_examples = [
        # List comprehensions - with code
        """List comprehension syntax: [expression for item in iterable if condition]

Python code example:
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# Result: [0, 4, 16, 36, 64]

# Nested list comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]
# Result: [[0, 0, 0], [0, 1, 2], [0, 2, 4]]
```""",

        # Dictionary comprehensions - with code
        """Dictionary comprehension: {key: value for item in iterable}

Python code example:
```python
# Basic dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With condition
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# Result: {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# From list of tuples
items = [('a', 1), ('b', 2), ('c', 3)]
result = {k: v*2 for k, v in items}
# Result: {'a': 2, 'b': 4, 'c': 6}
```""",

        # Error handling - with code
        """Python error handling uses try/except blocks. Always catch specific exceptions.

Python code example:
```python
def safe_divide(a: float, b: float) -> float:
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return 0.0
    except TypeError as e:
        print(f"Error: Invalid types - {e}")
        raise
    finally:
        print("Division operation completed")

# Usage
result = safe_divide(10, 2)  # Returns 5.0
result = safe_divide(10, 0)  # Returns 0.0, prints error
```""",

        # Context managers - with code
        """Context managers use 'with' statement for resource management. Ensures proper cleanup.

Python code example:
```python
# File handling with context manager
with open('data.txt', 'r') as f:
    content = f.read()
    # File automatically closed after block

# Custom context manager
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Elapsed time: {elapsed:.2f} seconds")

# Usage
with timer():
    # Your code here
    time.sleep(1)
```""",

        # Decorators - with code
        """Decorators modify function behavior. Common decorators: @staticmethod, @classmethod, @property.

Python code example:
```python
from functools import wraps, lru_cache

# Simple decorator
def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Property decorator
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

# Caching decorator
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```""",

        # Type hints - with code
        '''Type hints improve code readability. Use Optional[Type] for nullable values.

Python code example:
```python
from typing import Optional, List, Dict, Union

def process_data(data: List[str]) -> Dict[str, int]:
    """Process list of strings and return count dictionary"""
    return {item: len(item) for item in data}

def get_user(id: int) -> Optional[Dict[str, str]]:
    """Get user by ID, returns None if not found"""
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(id)

def parse_value(value: Union[str, int]) -> int:
    """Parse string or int to int"""
    if isinstance(value, str):
        return int(value)
    return value
```''',

        # Async/await - with code
        '''Async functions use 'async def' and 'await' keywords. Use asyncio.run() to execute.

Python code example:
```python
import asyncio
import aiohttp

async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def fetch_multiple(urls: List[str]) -> List[dict]:
    """Fetch multiple URLs concurrently"""
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)

# Usage
async def main():
    urls = ['https://api.example.com/data1', 'https://api.example.com/data2']
    results = await fetch_multiple(urls)
    return results

# Run async function
results = asyncio.run(main())
```''',

        # Generators - with code
        '''Generators use 'yield' instead of 'return'. Memory efficient for large datasets.

Python code example:
```python
def count_up_to(max: int):
    """Generator that counts up to max"""
    count = 1
    while count <= max:
        yield count
        count += 1

# Usage
for num in count_up_to(5):
    print(num)  # Prints 1, 2, 3, 4, 5

# Generator expression
squares = (x**2 for x in range(10))
for square in squares:
    print(square)

# Infinite generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
for _ in range(10):
    print(next(fib))
```''',
    ]

    # API and framework patterns
    api_patterns = [
        # REST API design
        "REST API best practices: Use proper HTTP methods (GET, POST, PUT, DELETE), "
        "return appropriate status codes (200 OK, 201 Created, 404 Not Found, 500 Error), "
        "use consistent URL patterns (/api/v1/resources/:id).",

        # Error handling in APIs
        "API error handling: Return structured error responses with status codes. "
        "Example: {'error': {'code': 'VALIDATION_ERROR', 'message': 'Invalid input'}} with 400 status.",

        # Authentication
        "API authentication: Use Bearer tokens in Authorization header. "
        "Example: Authorization: Bearer <token>. Validate tokens on every request.",

        # Rate limiting
        "Rate limiting: Implement per-user or per-IP limits. "
        "Return 429 Too Many Requests with Retry-After header when limit exceeded.",
    ]

    # Testing patterns
    testing_patterns = [
        # Unit testing
        "Unit testing: Test individual functions in isolation. "
        "Use pytest: def test_function(): assert function() == expected. "
        "Use fixtures for setup: @pytest.fixture def data(): return test_data.",

        # Mocking
        "Mocking: Replace dependencies with test doubles. "
        "Example: @patch('module.external_api') def test_with_mock(mock_api): mock_api.return_value = 'test'.",

        # Test organization
        "Test organization: Group related tests in classes, use descriptive names. "
        "Example: class TestUserAuthentication: def test_valid_login(): ... def test_invalid_password(): ...",
    ]

    # Best practices
    best_practices = [
        # Code organization
        "Code organization: Follow PEP 8 style guide, use meaningful variable names, "
        "keep functions small and focused (single responsibility principle), "
        "add docstrings to classes and functions.",

        # Version control
        "Version control best practices: Write clear commit messages, "
        "use feature branches, review code before merging, "
        "keep commits atomic (one logical change per commit).",

        # Code review
        "Code review checklist: Check for bugs, verify tests pass, "
        "ensure code follows style guide, verify error handling, "
        "check for security vulnerabilities, verify documentation.",

        # Performance
        "Performance optimization: Profile before optimizing, "
        "use appropriate data structures (dicts for lookups, sets for membership), "
        "avoid premature optimization, cache expensive computations.",
    ]

    # Combine all knowledge
    all_docs = python_examples + api_patterns + testing_patterns + best_practices

    logger.info(f"Adding {len(all_docs)} developer knowledge documents...")
    agent.add_knowledge(all_docs)

    return len(all_docs)


def search_code_examples(agent, query: str, top_k: int = 3):
    """Search for code examples and patterns, returning Python code"""
    logger.info(f"Searching for: {query}")

    try:
        result = agent.answer_with_context(query, top_k=top_k)

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Found {result['num_results']} relevant code examples:")
        print(f"{'='*60}\n")

        # Extract and format Python code from results
        code_examples = []
        for i, context in enumerate(result['context'], 1):
            print(f"[{i}] Similarity: {context['similarity']:.3f}")
            text = context['text']

            # Check if result contains code blocks
            if '```python' in text or '```' in text:
                # Extract code block
                if '```python' in text:
                    code_start = text.find('```python') + 9
                else:
                    code_start = text.find('```') + 3
                code_end = text.find('```', code_start)
                if code_end > code_start:
                    code = text[code_start:code_end].strip()
                    code_examples.append(code)
                    print(f"    Generated Python Code:")
                    print(f"    {'-'*56}")
                    # Print code with indentation
                    for line in code.split('\n'):
                        print(f"    {line}")
                    print(f"    {'-'*56}\n")
                else:
                    print(f"    {text}\n")
            else:
                print(f"    {text}\n")

        # Return result with extracted code
        result['code_examples'] = code_examples

        # If code examples were found, show how to use them
        if code_examples:
            print(f"\n{'='*60}")
            print("Generated Python Code Ready to Use:")
            print(f"{'='*60}")
            for i, code in enumerate(code_examples, 1):
                print(f"\nCode Example {i}:")
                print("-" * 60)
                print(code)
                print("-" * 60)

        return result

    except ValueError as e:
        logger.error(f"Invalid query: {e}")
        print(f"Error: {e}")
        return None
    except Exception as e:
        error_str = str(e)
        error_type = str(type(e).__name__)

        # Check for 404 errors (endpoint not found)
        if "404" in error_str or "NotFoundError" in error_type or "Not Found" in error_str:
            logger.error(f"Endpoint not found (404): {e}")
            print("\n" + "="*60)
            print("ERROR: Endpoint Not Found (404)")
            print("="*60)
            print("\nThe API endpoint URL or model name is incorrect.")
            print("\nTo fix this:")
            print("  1. Verify your endpoint URL in Cloudera AI Platform console")
            print("  2. Check that the endpoint name in config.json matches exactly")
            print("  3. Verify the model IDs are correct for your endpoint")
            print("  4. Update config.json with the correct endpoint URL and model names")
            print("\nExample config.json structure:")
            print("  {")
            print("    \"endpoint\": {")
            print("      \"base_url\": \"https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint-name/v1\"")
            print("    },")
            print("    \"models\": {")
            print("      \"query_model\": \"your-model-id\",")
            print("      \"passage_model\": \"your-model-id\"")
            print("    }")
            print("  }")
            print("="*60)
        # Check for authentication errors
        elif "401" in error_str or "Token has expired" in error_str or "AuthenticationError" in error_type:
            logger.error(f"Authentication error: {e}")
            print("\n" + "="*60)
            print("ERROR: Authentication Failed")
            print("="*60)
            print("\nThe API token has expired or is invalid.")
            print("\nPlease update your API token in config.json or environment variables.")
            print("="*60)
        else:
            logger.exception(f"Unexpected error: {e}")
            print(f"Error: {e}")

        return None


def interactive_search(agent):
    """Interactive search mode for developers"""
    print("\n" + "="*60)
    print("Developer Knowledge Base - Interactive Search")
    print("="*60)
    print("Enter queries to search code examples, patterns, and best practices.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                print("Please enter a query.")
                continue

            search_code_examples(agent, query, top_k=3)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    """Main function demonstrating developer usage"""
    print("="*60)
    print("Cloudera Inference With CursorAI - Developer Example")
    print("="*60)

    try:
        # Create agent
        print("\n1. Creating Cloudera Agent...")
        agent = create_cloudera_agent()
        print("   ✓ Agent created successfully!")

        # Display agent stats
        stats = agent.get_stats()
        print(f"   Models: {stats['query_model']}, {stats['passage_model']}")
        print(f"   Embedding dimension: {stats['embedding_dim']}\n")

        # Setup knowledge base
        print("2. Setting up developer knowledge base...")
        num_docs = setup_developer_knowledge_base(agent)
        print(f"   ✓ Added {num_docs} documents to knowledge base\n")

        # Example queries
        print("3. Running example queries...\n")

        example_queries = [
            "How do I use list comprehensions in Python?",
            "Show me Python code for dictionary comprehensions",
            "Generate Python code for error handling with try/except",
            "How do I use context managers in Python?",
            "Show me Python code for async/await functions",
        ]

        # Run queries and collect generated code
        all_generated_code = []
        for query in example_queries:
            result = search_code_examples(agent, query, top_k=2)
            if result and 'code_examples' in result:
                all_generated_code.extend(result['code_examples'])

        # Show summary of generated code
        if all_generated_code:
            print(f"\n{'='*60}")
            print(f"Summary: Generated {len(all_generated_code)} Python code examples")
            print(f"{'='*60}")
            print("\nYou can now use these code examples in your projects!")
            print("Copy the code from above and paste it into your Python files.")

        # Interactive mode
        print("\n4. Starting interactive search mode...")
        print("   (You can also run this script with --interactive flag)\n")

        if '--interactive' in sys.argv or '-i' in sys.argv:
            interactive_search(agent)
        else:
            print("   Run with --interactive flag for interactive mode:")
            print("   python example_developer_usage.py --interactive")

        print("\n" + "="*60)
        print("Example completed successfully!")
        print("="*60)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  - config.json is configured, OR")
        print("  - Environment variables are set (CLOUDERA_EMBEDDING_URL, OPENAI_API_KEY, etc.)")
        sys.exit(1)
    except Exception as e:
        error_str = str(e)
        error_type = str(type(e).__name__)

        # Check for 404 errors (endpoint not found)
        if "404" in error_str or "NotFoundError" in error_type or "Not Found" in error_str:
            logger.error(f"Endpoint not found (404): {e}")
            print("\n" + "="*60)
            print("ERROR: Endpoint Not Found (404)")
            print("="*60)
            print("\nThe API endpoint URL or model name is incorrect.")
            print("\nTo fix this:")
            print("  1. Verify your endpoint URL in Cloudera AI Platform console")
            print("  2. Check that the endpoint name in config.json matches exactly")
            print("  3. Verify the model IDs are correct for your endpoint")
            print("  4. Update config.json with the correct endpoint URL and model names")
            print("\nExample config.json structure:")
            print("  {")
            print("    \"endpoint\": {")
            print("      \"base_url\": \"https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint-name/v1\"")
            print("    },")
            print("    \"models\": {")
            print("      \"query_model\": \"your-model-id\",")
            print("      \"passage_model\": \"your-model-id\"")
            print("    }")
            print("  }")
            print("="*60)
        # Check for authentication errors
        elif "401" in error_str or "Token has expired" in error_str or "AuthenticationError" in error_type:
            logger.error(f"Authentication error: {e}")
            print("\n" + "="*60)
            print("ERROR: Authentication Failed")
            print("="*60)
            print("\nThe API token has expired or is invalid.")
            print("\nTo fix this:")
            print("  1. Get a new API token from Cloudera AI Platform")
            print("  2. Update config.json with the new token:")
            print("     - Edit config.json")
            print("     - Replace the 'api_key' value with your new token")
            print("\n  3. Or update environment variable:")
            print("     export OPENAI_API_KEY='your-new-token'")
            print("\n  4. Run the script again")
            print("="*60)
        else:
            logger.exception(f"Unexpected error: {e}")
            print(f"\nUnexpected error: {e}")
            print("\nCheck the logs above for more details.")

        sys.exit(1)


if __name__ == "__main__":
    main()

