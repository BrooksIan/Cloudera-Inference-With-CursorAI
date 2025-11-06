#!/usr/bin/env python3
"""
Example: Using Cloudera-hosted Language Models (LLMs)

This example demonstrates how to use a Cloudera-hosted language model
like nvidia/llama-3.3-nemotron-super-49b-v1 for text generation and chat completions.

Note: Language models use the /chat/completions endpoint, not /embeddings.
"""

import os
import json
import logging
from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from typing import Optional, List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory (parent of examples directory)
PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from config.json, config-llm.json, or environment variables"""
    config_data = {}

    # Try to load from config file
    if config_file is None:
        # First try config-llm.json (dedicated LLM config) in project root
        config_llm_path = PROJECT_ROOT / "config-llm.json"
        config_json_path = PROJECT_ROOT / "config.json"

        if config_llm_path.exists():
            config_file = str(config_llm_path)
        # Fall back to config.json
        elif config_json_path.exists():
            config_file = str(config_json_path)
        else:
            logger.info("No config file found, using environment variables")
            return config_data
    else:
        # If config_file is provided, resolve it relative to project root if it's not absolute
        if not os.path.isabs(config_file):
            config_file_path = PROJECT_ROOT / config_file
            if config_file_path.exists():
                config_file = str(config_file_path)

    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {config_file}: {e}")
    else:
        logger.info(f"Config file {config_file} not found, using environment variables")

    return config_data


def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    config_file: Optional[str] = None
) -> Tuple[OpenAI, str]:
    """Create an OpenAI-compatible client for Cloudera-hosted LLM

    Args:
        base_url: Cloudera endpoint URL (ending in /v1)
        api_key: API key (JWT token)
        model: Model ID (e.g., nvidia/llama-3.3-nemotron-super-49b-v1)
        config_file: Path to config.json file

    Returns:
        OpenAI client configured for Cloudera endpoint
    """
    config_data = load_config(config_file)

    # Get base_url from parameter, env var, or config file
    if not base_url:
        base_url = os.getenv("CLOUDERA_LLM_URL") or \
                  config_data.get("llm_endpoint", {}).get("base_url") or \
                  config_data.get("endpoint", {}).get("base_url")

    if not base_url:
        raise ValueError(
            "Endpoint URL required. Set CLOUDERA_LLM_URL environment variable, "
            "provide config.json file with llm_endpoint.base_url, or pass base_url parameter.\n"
            "Example: export CLOUDERA_LLM_URL='https://your-endpoint.com/namespaces/serving-default/endpoints/your-llm-endpoint/v1'"
        )

    # Ensure base_url ends with /v1
    if not base_url.endswith("/v1"):
        if base_url.endswith("/"):
            base_url = base_url + "v1"
        else:
            base_url = base_url + "/v1"

    # Get API key from parameter, env var, or config file
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or config_data.get("api_key")

        # Fall back to /tmp/jwt (for workbench)
        if not api_key and os.path.exists("/tmp/jwt"):
            try:
                with open("/tmp/jwt", "r") as f:
                    jwt_data = json.load(f)
                    api_key = jwt_data.get("access_token")
            except Exception as e:
                logger.warning(f"Failed to read /tmp/jwt: {e}")

    if not api_key:
        raise ValueError(
            "API key required. Set OPENAI_API_KEY environment variable, "
            "provide config.json file, or provide /tmp/jwt file"
        )

    # Get model from parameter, env var, or config file
    if not model:
        model = os.getenv("CLOUDERA_LLM_MODEL") or \
               config_data.get("llm_endpoint", {}).get("model") or \
               config_data.get("models", {}).get("llm_model")

    if not model:
        raise ValueError(
            "Model ID required. Set CLOUDERA_LLM_MODEL environment variable or "
            "configure llm_endpoint.model in config.json"
        )

    # Create OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0  # LLMs may take longer than embeddings
    )

    logger.info(f"Initialized LLM client with endpoint: {base_url}")
    logger.info(f"Using model: {model}")

    return client, model


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """Generate chat completion using Cloudera-hosted LLM

    Args:
        client: OpenAI client configured for Cloudera endpoint
        model: Model ID
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create

    Returns:
        Generated text response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    except APIError as e:
        error_str = str(e)
        # Check for 404 errors (endpoint not found)
        if "404" in error_str or hasattr(e, 'status_code') and e.status_code == 404:
            logger.error(f"Endpoint not found (404): {e}")
            logger.error(f"Current endpoint URL: {client.base_url}")
            logger.error("\nThis usually means:")
            logger.error("  1. The endpoint name in your config file is incorrect")
            logger.error("  2. The endpoint doesn't exist or isn't deployed yet")
            logger.error("  3. The endpoint URL has a typo")
            logger.error("\nTo fix this:")
            logger.error("  1. Log into Cloudera AI Platform")
            logger.error("  2. Navigate to Deployments → Model Endpoints")
            logger.error("  3. Find the endpoint that hosts your LLM model")
            logger.error(f"  4. Copy the exact endpoint name and update config-llm.json")
            logger.error("  5. Replace 'your-llm-endpoint' in the base_url with the actual endpoint name")
            raise ValueError(
                f"Endpoint not found (404). Please check your config-llm.json file.\n"
                f"Current endpoint: {client.base_url}\n"
                f"Make sure the endpoint name matches exactly what's in Cloudera AI Platform."
            ) from e
        logger.error(f"API error: {e}")
        raise
    except (APIConnectionError, APITimeoutError) as e:
        logger.error(f"Connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def simple_query(client: OpenAI, model: str, prompt: str, **kwargs) -> str:
    """Simple query function for single prompts

    Args:
        client: OpenAI client configured for Cloudera endpoint
        model: Model ID
        prompt: User prompt/question
        **kwargs: Additional parameters for chat completion

    Returns:
        Generated text response
    """
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(client, model, messages, **kwargs)


def rag_with_llm(
    embedding_agent,
    llm_client: OpenAI,
    llm_model: str,
    question: str,
    top_k: int = 3,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Complete RAG pipeline: Retrieve context using embeddings, then generate answer with LLM

    Args:
        embedding_agent: ClouderaAgent instance for retrieval
        llm_client: OpenAI client for LLM
        llm_model: Model ID for LLM
        question: User question
        top_k: Number of context chunks to retrieve
        temperature: LLM temperature

    Returns:
        Dict with answer, sources, and similarities
    """
    # 1. Retrieve relevant context using embeddings
    logger.info(f"Retrieving context for: {question}")
    result = embedding_agent.answer_with_context(question, top_k=top_k)
    context = result['context_text']

    # 2. Format prompt with context
    prompt = f"""Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

    # 3. Generate answer with LLM
    logger.info("Generating answer with LLM...")
    messages = [{"role": "user", "content": prompt}]
    answer = chat_completion(
        llm_client,
        llm_model,
        messages,
        temperature=temperature
    )

    return {
        'answer': answer,
        'sources': [ctx['text'] for ctx in result['context']],
        'similarities': [ctx['similarity'] for ctx in result['context']],
        'context_text': context
    }


def main():
    """Example usage of Cloudera-hosted LLM"""
    print("="*60)
    print("Cloudera LLM Usage Example")
    print("="*60)

    try:
        # Create LLM client
        print("\n1. Creating LLM client...")
        client, model = create_llm_client()
        print(f"   ✓ Client created successfully!")
        print(f"   Model: {model}")
        print(f"   Endpoint: {client.base_url}")

        # Example 1: Simple query
        print("\n2. Example 1: Simple Query")
        print("-" * 60)
        question = "What is Python programming language?"
        print(f"Question: {question}")
        answer = simple_query(client, model, question, temperature=0.7)
        print(f"Answer: {answer}")

        # Example 2: Chat conversation
        print("\n3. Example 2: Chat Conversation")
        print("-" * 60)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the key features of Python?"},
        ]
        response1 = chat_completion(client, model, messages, temperature=0.7)
        print(f"User: {messages[1]['content']}")
        print(f"Assistant: {response1}")

        # Continue conversation
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "Can you give me a code example?"})
        response2 = chat_completion(client, model, messages, temperature=0.7)
        print(f"\nUser: {messages[-1]['content']}")
        print(f"Assistant: {response2}")

        # Example 3: RAG with LLM (if embedding agent is available)
        print("\n4. Example 3: RAG with LLM")
        print("-" * 60)
        try:
            from agents import create_cloudera_agent

            # Create embedding agent for retrieval
            embedding_agent = create_cloudera_agent()

            # Add some knowledge
            embedding_agent.add_knowledge([
                "Python is a high-level programming language known for its simplicity.",
                "Python supports multiple programming paradigms including object-oriented and functional programming.",
                "Python has a large standard library and active community.",
            ])

            # RAG query
            rag_question = "What are the main characteristics of Python?"
            result = rag_with_llm(embedding_agent, client, model, rag_question)

            print(f"Question: {rag_question}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources used:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source[:100]}...")
        except ImportError:
            print("   (Skipping RAG example - embedding agent not available)")
        except Exception as e:
            print(f"   (Skipping RAG example - {e})")

        print("\n" + "="*60)
        print("Example completed successfully!")
        print("="*60)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  - config.json includes llm_endpoint configuration, OR")
        print("  - Environment variables are set (CLOUDERA_LLM_URL, CLOUDERA_LLM_MODEL, OPENAI_API_KEY)")
        print("\nExample config.json structure:")
        print("  {")
        print("    \"llm_endpoint\": {")
        print("      \"base_url\": \"https://your-endpoint.com/namespaces/serving-default/endpoints/your-llm-endpoint/v1\",")
        print("      \"model\": \"nvidia/llama-3.3-nemotron-super-49b-v1\"")
        print("    },")
        print("    \"api_key\": \"your-api-key\"")
        print("  }")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

