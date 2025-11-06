"""
Cloudera Inference With CursorAI
Uses Cloudera-hosted embedding models for RAG (Retrieval Augmented Generation) agents.
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional, Any
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import numpy as np
from dataclasses import dataclass

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for Cloudera embedding endpoint"""
    base_url: str
    api_key: str
    query_model: str
    passage_model: str
    embedding_dim: int = 1024


@dataclass
class LLMConfig:
    """Configuration for Cloudera LLM endpoint"""
    base_url: str
    api_key: str
    model: str


class ClouderaEmbeddingClient:
    """Client for Cloudera embedding endpoint"""

    def __init__(self, config: EmbeddingConfig, max_retries: int = 3, retry_delay: float = 1.0):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=30.0  # 30 second timeout
        )
        logger.info(f"Initialized ClouderaEmbeddingClient with endpoint: {config.base_url}")

    def _validate_text(self, text: str, method_name: str) -> None:
        """Validate input text"""
        if not isinstance(text, str):
            raise TypeError(f"{method_name}: text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"{method_name}: text cannot be empty")
        if len(text) > 100000:  # Reasonable limit
            raise ValueError(f"{method_name}: text too long (max 100000 characters)")

    def _retry_api_call(self, func, *args, **kwargs):
        """Retry API call with exponential backoff"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API connection error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"API connection failed after {self.max_retries} attempts: {e}")
            except APIError as e:
                # Don't retry on API errors (4xx, 5xx) unless it's a rate limit
                if e.status_code == 429:  # Rate limit
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Rate limit error after {self.max_retries} attempts: {e}")
                        raise
                else:
                    logger.error(f"API error (status {e.status_code}): {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during API call: {e}")
                raise

        if last_exception:
            raise last_exception

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query using the query model

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is invalid
            APIError: If API call fails
        """
        self._validate_text(text, "embed_query")
        logger.debug(f"Generating query embedding for text (length: {len(text)})")

        def _call():
            response = self.client.embeddings.create(
                input=text,
                model=self.config.query_model
            )
            return response.data[0].embedding

        embedding = self._retry_api_call(_call)
        logger.debug(f"Successfully generated query embedding (dimension: {len(embedding)})")
        return embedding

    def embed_passage(self, text: str) -> List[float]:
        """Generate embedding for a passage using the passage model

        Args:
            text: Passage text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is invalid
            APIError: If API call fails
        """
        self._validate_text(text, "embed_passage")
        logger.debug(f"Generating passage embedding for text (length: {len(text)})")

        def _call():
            response = self.client.embeddings.create(
                input=text,
                model=self.config.passage_model
            )
            return response.data[0].embedding

        embedding = self._retry_api_call(_call)
        logger.debug(f"Successfully generated passage embedding (dimension: {len(embedding)})")
        return embedding

    def embed_batch(self, texts: List[str], use_passage: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            use_passage: If True, use passage model; otherwise use query model

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is invalid
            APIError: If API call fails
        """
        if not isinstance(texts, list):
            raise TypeError(f"embed_batch: texts must be a list, got {type(texts).__name__}")
        if not texts:
            raise ValueError("embed_batch: texts list cannot be empty")

        model = self.config.passage_model if use_passage else self.config.query_model
        model_type = "passage" if use_passage else "query"
        logger.info(f"Generating {len(texts)} {model_type} embeddings using model: {model}")

        embeddings = []
        for i, text in enumerate(texts):
            try:
                self._validate_text(text, f"embed_batch[{i}]")
                def _call():
                    response = self.client.embeddings.create(
                        input=text,
                        model=model
                    )
                    return response.data[0].embedding

                embedding = self._retry_api_call(_call)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")
            except Exception as e:
                logger.error(f"Failed to embed text at index {i}: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration"""

    def __init__(self, embedding_client: ClouderaEmbeddingClient):
        self.embedding_client = embedding_client
        self.documents: List[Dict[str, Any]] = []
        self.vectors: List[List[float]] = []
        logger.info("Initialized SimpleVectorStore")

    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """Add a document to the vector store

        Args:
            text: Document text
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("add_document: text must be a non-empty string")

        logger.debug(f"Adding document to vector store (text length: {len(text)})")
        embedding = self.embedding_client.embed_passage(text)
        self.vectors.append(embedding)
        self.documents.append({
            "text": text,
            "metadata": metadata or {}
        })
        logger.debug(f"Document added. Total documents: {len(self.documents)}")

    def add_documents(self, texts: List[str], metadata_list: Optional[List[Dict]] = None):
        """Add multiple documents to the vector store

        Args:
            texts: List of document texts
            metadata_list: Optional list of metadata dictionaries

        Raises:
            ValueError: If texts list is invalid or lengths don't match
        """
        if not isinstance(texts, list):
            raise TypeError(f"add_documents: texts must be a list, got {type(texts).__name__}")
        if not texts:
            raise ValueError("add_documents: texts list cannot be empty")

        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        elif len(metadata_list) != len(texts):
            raise ValueError(
                f"add_documents: metadata_list length ({len(metadata_list)}) "
                f"must match texts length ({len(texts)})"
            )

        logger.info(f"Adding {len(texts)} documents to vector store")
        embeddings = self.embedding_client.embed_batch(texts, use_passage=True)

        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            self.vectors.append(embedding)
            self.documents.append({
                "text": text,
                "metadata": metadata
            })

        logger.info(f"Successfully added {len(texts)} documents. Total documents: {len(self.documents)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing text, metadata, and similarity score

        Raises:
            ValueError: If query is invalid or top_k is invalid
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search: query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"search: top_k must be a positive integer, got {top_k}")

        if not self.vectors:
            logger.debug("Search called on empty vector store")
            return []

        logger.debug(f"Searching for query (length: {len(query)}, top_k: {top_k})")
        query_embedding = self.embedding_client.embed_query(query)
        query_vector = np.array(query_embedding)

        # Calculate cosine similarities
        similarities = []
        for vector in self.vectors:
            doc_vector = np.array(vector)
            norm_product = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            if norm_product == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vector, doc_vector) / norm_product
            similarities.append(similarity)

        # Get top k results
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "similarity": float(similarities[idx])
            })

        logger.debug(f"Search completed. Found {len(results)} results")
        return results


class ClouderaAgent:
    """Agent that uses Cloudera embeddings for RAG and optionally LLMs for generation"""

    def __init__(
        self,
        config: EmbeddingConfig,
        llm_config: Optional[LLMConfig] = None,
        vector_store: Optional[SimpleVectorStore] = None
    ):
        self.config = config
        self.embedding_client = ClouderaEmbeddingClient(config)
        self.vector_store = vector_store or SimpleVectorStore(self.embedding_client)

        # Optional LLM client for generation
        self.llm_config = llm_config
        self.llm_client = None
        if llm_config:
            self.llm_client = OpenAI(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                timeout=60.0  # LLMs may take longer than embeddings
            )
            logger.info(f"Initialized ClouderaAgent with LLM: {llm_config.model}")

        logger.info(f"Initialized ClouderaAgent with models: query={config.query_model}, passage={config.passage_model}")

    def add_knowledge(self, texts: List[str], metadata_list: Optional[List[Dict]] = None):
        """Add knowledge base documents to the agent

        Args:
            texts: List of document texts to add
            metadata_list: Optional list of metadata dictionaries

        Raises:
            ValueError: If texts list is invalid
        """
        logger.info(f"Adding {len(texts)} documents to knowledge base")
        self.vector_store.add_documents(texts, metadata_list)
        logger.info(f"Knowledge base now contains {len(self.vector_store.documents)} documents")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of context dictionaries with text, metadata, and similarity

        Raises:
            ValueError: If query is invalid
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("retrieve_context: query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"retrieve_context: top_k must be a positive integer, got {top_k}")

        logger.debug(f"Retrieving context for query (top_k: {top_k})")
        results = self.vector_store.search(query, top_k=top_k)
        logger.debug(f"Retrieved {len(results)} context results")
        return results

    def answer_with_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a query using retrieved context

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            Dictionary containing query, context results, context text, and num_results

        Raises:
            ValueError: If query is invalid
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("answer_with_context: query must be a non-empty string")

        logger.info(f"Processing query with context retrieval (top_k: {top_k})")
        # Retrieve relevant context
        context_results = self.retrieve_context(query, top_k)

        # Build context string
        context_text = "\n\n".join([
            f"[Context {i+1}]: {result['text']}"
            for i, result in enumerate(context_results)
        ])

        result = {
            "query": query,
            "context": context_results,
            "context_text": context_text,
            "num_results": len(context_results)
        }

        logger.info(f"Query processed. Retrieved {len(context_results)} context results")
        return result

    def answer_with_llm(
        self,
        query: str,
        top_k: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """Answer a query using RAG (Retrieval Augmented Generation) with LLM

        Args:
            query: Query text
            top_k: Number of top context results to retrieve
            temperature: LLM temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            use_context: If True, retrieve context first; if False, use LLM directly

        Returns:
            Dictionary containing query, answer, context results, and metadata

        Raises:
            ValueError: If query is invalid or LLM is not configured
        """
        if not self.llm_client:
            raise ValueError(
                "LLM not configured. Provide llm_config when creating ClouderaAgent, "
                "or use answer_with_context() for retrieval-only mode."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("answer_with_llm: query must be a non-empty string")

        logger.info(f"Processing query with RAG (top_k: {top_k}, use_context: {use_context})")

        # Retrieve context if requested
        context_results = []
        context_text = ""
        if use_context:
            context_results = self.retrieve_context(query, top_k)
            context_text = "\n\n".join([
                f"[Context {i+1}]: {result['text']}"
                for i, result in enumerate(context_results)
            ])

        # Build prompt
        if context_text:
            prompt = f"""Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        else:
            prompt = query

        # Generate answer with LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message.content
        except APIError as e:
            error_str = str(e)
            if "404" in error_str or (hasattr(e, 'status_code') and e.status_code == 404):
                logger.error(f"LLM endpoint not found (404): {e}")
                logger.error(f"Current endpoint: {self.llm_client.base_url}")
                raise ValueError(
                    f"LLM endpoint not found (404). Please check your LLM configuration.\n"
                    f"Current endpoint: {self.llm_client.base_url}\n"
                    f"Make sure the endpoint name matches exactly what's in Cloudera AI Platform."
                ) from e
            raise

        result = {
            "query": query,
            "answer": answer,
            "context": context_results,
            "context_text": context_text,
            "num_results": len(context_results),
            "model": self.llm_config.model,
            "temperature": temperature
        }

        logger.info(f"Query processed. Generated answer using LLM: {self.llm_config.model}")
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's knowledge base

        Returns:
            Dictionary containing statistics about the agent
        """
        stats = {
            "num_documents": len(self.vector_store.documents),
            "embedding_dim": self.config.embedding_dim,
            "query_model": self.config.query_model,
            "passage_model": self.config.passage_model
        }
        if self.llm_config:
            stats["llm_model"] = self.llm_config.model
            stats["llm_endpoint"] = self.llm_config.base_url
        logger.debug(f"Agent stats: {stats}")
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the endpoint and return status

        Returns:
            Dictionary containing health check results with 'status' (bool) and 'details' (dict)

        Raises:
            Exception: If health check fails
        """
        results = {
            "status": True,
            "details": {}
        }

        try:
            # Test query embedding
            query_embedding = self.embedding_client.embed_query("health check test")
            results["details"]["query_embedding"] = {
                "status": "ok",
                "dimension": len(query_embedding)
            }

            # Test passage embedding
            passage_embedding = self.embedding_client.embed_passage("health check test passage")
            results["details"]["passage_embedding"] = {
                "status": "ok",
                "dimension": len(passage_embedding)
            }

            # Test batch embedding
            batch_embeddings = self.embedding_client.embed_batch(["test 1", "test 2"], use_passage=True)
            results["details"]["batch_embedding"] = {
                "status": "ok",
                "count": len(batch_embeddings)
            }

            # Get configuration info
            stats = self.get_stats()
            results["details"]["configuration"] = stats

            logger.info("Health check passed")
            return results

        except Exception as e:
            results["status"] = False
            results["details"]["error"] = str(e)
            logger.error(f"Health check failed: {e}")
            raise


def create_cloudera_agent(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    config_file: Optional[str] = None,
    llm_config_file: Optional[str] = None,
    use_llm: bool = True
) -> ClouderaAgent:
    """Factory function to create a Cloudera agent with default configuration

    Args:
        base_url: Embedding endpoint URL (overrides config file and environment)
        api_key: API key for authentication (overrides config file and environment)
        config_file: Path to JSON configuration file for embeddings (default: config.json)
        llm_config_file: Path to JSON configuration file for LLM (default: config-llm.json)
        use_llm: If True, attempt to load LLM configuration (default: True)

    Configuration priority:
        1. Function parameters (base_url, api_key)
        2. Environment variables (CLOUDERA_EMBEDDING_URL, OPENAI_API_KEY)
        3. Config file (config.json)
        4. /tmp/jwt file (for workbench environments)
    """

    # Try to load from config file if not provided
    if config_file is None:
        config_file = "config.json"

    config_data = {}
    if os.path.exists(config_file):
        try:
            logger.debug(f"Loading configuration from {config_file}")
            with open(config_file, "r") as f:
                config_data = json.load(f)
            logger.debug(f"Successfully loaded configuration from {config_file}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON in {config_file}: {e}. Continuing with environment variables.")
        except Exception as e:
            logger.warning(f"Failed to read config file {config_file}: {e}. Continuing with environment variables.")

    # Get configuration from parameters, environment, or config file
    if base_url is None:
        # Try environment variable first
        base_url = os.getenv("CLOUDERA_EMBEDDING_URL")

        # Fall back to config file
        if not base_url and config_data:
            base_url = config_data.get("endpoint", {}).get("base_url") or \
                      config_data.get("endpoint", {}).get("base_endpoint")

    if not base_url:
        raise ValueError(
            "Endpoint URL required. Set CLOUDERA_EMBEDDING_URL environment variable, "
            "provide config.json file, or pass base_url parameter.\n"
            "Example: export CLOUDERA_EMBEDDING_URL='https://your-endpoint.com/namespaces/serving-default/endpoints/your-endpoint/v1'"
        )

    if api_key is None:
        # Try environment variable first
        api_key = os.getenv("OPENAI_API_KEY")

        # Fall back to config file
        if not api_key and config_data:
            api_key = config_data.get("api_key")

        # Fall back to /tmp/jwt (for workbench)
        if not api_key and os.path.exists("/tmp/jwt"):
            try:
                logger.debug("Attempting to load API key from /tmp/jwt")
                with open("/tmp/jwt", "r") as f:
                    jwt_data = json.load(f)
                    api_key = jwt_data.get("access_token")
                if api_key:
                    logger.debug("Successfully loaded API key from /tmp/jwt")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON in /tmp/jwt: {e}")
            except Exception as e:
                logger.warning(f"Failed to read /tmp/jwt: {e}")

    if not api_key:
        raise ValueError(
            "API key required. Set OPENAI_API_KEY environment variable, "
            "provide config.json file, or provide /tmp/jwt file"
        )

    # Get model IDs from config file or environment (required, no defaults)
    query_model = os.getenv("CLOUDERA_QUERY_MODEL") or \
                  config_data.get("models", {}).get("query_model")

    passage_model = os.getenv("CLOUDERA_PASSAGE_MODEL") or \
                    config_data.get("models", {}).get("passage_model")

    if not query_model:
        raise ValueError(
            "Query model ID required. Set CLOUDERA_QUERY_MODEL environment variable or "
            "configure models.query_model in config.json"
        )

    if not passage_model:
        raise ValueError(
            "Passage model ID required. Set CLOUDERA_PASSAGE_MODEL environment variable or "
            "configure models.passage_model in config.json"
        )

    embedding_dim = int(os.getenv("CLOUDERA_EMBEDDING_DIM",
                          config_data.get("embedding_dim", 1024)))

    config = EmbeddingConfig(
        base_url=base_url,
        api_key=api_key,
        query_model=query_model,
        passage_model=passage_model,
        embedding_dim=embedding_dim
    )

    # Try to load LLM configuration if requested
    llm_config = None
    if use_llm:
        # Determine LLM config file
        if llm_config_file is None:
            # Try config-llm.json first, then check config.json for llm_endpoint
            if os.path.exists("config-llm.json"):
                llm_config_file = "config-llm.json"
            elif config_data.get("llm_endpoint"):
                # LLM config is in the same config.json file
                llm_config_file = config_file
            else:
                llm_config_file = None

        if llm_config_file and os.path.exists(llm_config_file):
            try:
                logger.debug(f"Loading LLM configuration from {llm_config_file}")
                with open(llm_config_file, "r") as f:
                    llm_config_data = json.load(f)

                # Get LLM endpoint URL
                llm_base_url = os.getenv("CLOUDERA_LLM_URL") or \
                              llm_config_data.get("llm_endpoint", {}).get("base_url")

                # Get LLM model
                llm_model = os.getenv("CLOUDERA_LLM_MODEL") or \
                           llm_config_data.get("llm_endpoint", {}).get("model")

                # Get API key (can be shared with embeddings or separate)
                llm_api_key = os.getenv("OPENAI_API_KEY") or \
                             llm_config_data.get("api_key") or \
                             api_key  # Fall back to embedding API key

                if llm_base_url and llm_model and llm_api_key:
                    # Ensure base_url ends with /v1
                    if not llm_base_url.endswith("/v1"):
                        if llm_base_url.endswith("/"):
                            llm_base_url = llm_base_url + "v1"
                        else:
                            llm_base_url = llm_base_url + "/v1"

                    llm_config = LLMConfig(
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                        model=llm_model
                    )
                    logger.info(f"Loaded LLM configuration: model={llm_model}, endpoint={llm_base_url}")
                else:
                    logger.warning("LLM configuration incomplete, skipping LLM setup")
            except Exception as e:
                logger.warning(f"Failed to load LLM configuration from {llm_config_file}: {e}")
        else:
            logger.debug("No LLM configuration file found, agent will use retrieval-only mode")

    logger.info(f"Creating ClouderaAgent with endpoint: {base_url}, models: {query_model}, {passage_model}")
    return ClouderaAgent(config, llm_config=llm_config)

