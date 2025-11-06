#!/usr/bin/env python3
"""
Command-line interface for Cloudera Inference With CursorAI
Provides quick testing, health checks, and utility functions
"""

import argparse
import sys
import logging
from typing import Optional
from .cloudera_agent import create_cloudera_agent, ClouderaAgent, EmbeddingConfig
from .cloudera_agent import ClouderaEmbeddingClient

logger = logging.getLogger(__name__)


def health_check(agent: Optional[ClouderaAgent] = None) -> bool:
    """Perform a health check on the Cloudera endpoint

    Args:
        agent: Optional agent instance. If not provided, creates a new one.

    Returns:
        True if health check passes, False otherwise
    """
    try:
        if agent is None:
            print("Creating agent for health check...")
            agent = create_cloudera_agent()

        print("Performing health check...")

        # Test query embedding
        print("  ✓ Testing query embedding...")
        query_embedding = agent.embedding_client.embed_query("test query")
        if not query_embedding or len(query_embedding) == 0:
            print("  ✗ Query embedding failed: empty result")
            return False
        print(f"  ✓ Query embedding successful (dimension: {len(query_embedding)})")

        # Test passage embedding
        print("  ✓ Testing passage embedding...")
        passage_embedding = agent.embedding_client.embed_passage("test passage")
        if not passage_embedding or len(passage_embedding) == 0:
            print("  ✗ Passage embedding failed: empty result")
            return False
        print(f"  ✓ Passage embedding successful (dimension: {len(passage_embedding)})")

        # Test batch embedding
        print("  ✓ Testing batch embedding...")
        batch_embeddings = agent.embedding_client.embed_batch(["test 1", "test 2"], use_passage=True)
        if not batch_embeddings or len(batch_embeddings) != 2:
            print("  ✗ Batch embedding failed: incorrect result count")
            return False
        print(f"  ✓ Batch embedding successful ({len(batch_embeddings)} embeddings)")

        # Get stats
        stats = agent.get_stats()
        print("\nConfiguration:")
        print(f"  Endpoint: {agent.config.base_url}")
        print(f"  Query Model: {stats['query_model']}")
        print(f"  Passage Model: {stats['passage_model']}")
        print(f"  Embedding Dimension: {stats['embedding_dim']}")
        print(f"  Documents in Knowledge Base: {stats['num_documents']}")

        print("\n✓ Health check passed!")
        return True

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that config.json exists and is properly formatted")
        print("  2. Verify environment variables are set correctly")
        print("  3. See README.md for configuration instructions")
        return False
    except Exception as e:
        print(f"\n✗ Health check failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify endpoint URL is correct")
        print("  2. Check API key is valid and not expired")
        print("  3. Ensure network connectivity to Cloudera endpoint")
        print("  4. Verify model IDs match your endpoint configuration")
        return False


def test_query(agent: Optional[ClouderaAgent], query: str, top_k: int = 3):
    """Test a query against the knowledge base

    Args:
        agent: Optional agent instance. If not provided, creates a new one.
        query: Query text to test
        top_k: Number of results to return
    """
    try:
        if agent is None:
            agent = create_cloudera_agent()

        print(f"Query: {query}")
        print(f"Top K: {top_k}\n")

        result = agent.answer_with_context(query, top_k=top_k)

        print(f"Found {result['num_results']} results:\n")
        for i, context in enumerate(result['context'], 1):
            print(f"[{i}] Similarity: {context['similarity']:.4f}")
            print(f"    Text: {context['text'][:100]}...")
            if context['metadata']:
                print(f"    Metadata: {context['metadata']}")
            print()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def interactive_mode(agent: Optional[ClouderaAgent] = None):
    """Interactive query mode

    Args:
        agent: Optional agent instance. If not provided, creates a new one.
    """
    try:
        if agent is None:
            agent = create_cloudera_agent()

        print("Interactive Query Mode")
        print("Enter queries (or 'quit' to exit):\n")

        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                result = agent.answer_with_context(query, top_k=3)
                print(f"\nFound {result['num_results']} results:\n")
                for i, context in enumerate(result['context'], 1):
                    print(f"[{i}] Similarity: {context['similarity']:.4f}")
                    print(f"    {context['text'][:150]}...")
                print()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Cloudera Inference With CursorAI - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Health check
  cloudera-agent health

  # Test a query
  cloudera-agent query "What is Python?"

  # Interactive mode
  cloudera-agent interactive

  # Add documents and query
  cloudera-agent add "Document text here" --query "search query"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Health check command
    health_parser = subparsers.add_parser('health', help='Perform health check on endpoint')

    # Query command
    query_parser = subparsers.add_parser('query', help='Test a query')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--top-k', type=int, default=3, help='Number of results (default: 3)')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive query mode')

    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add documents to knowledge base')
    add_parser.add_argument('documents', nargs='+', help='Document texts to add')
    add_parser.add_argument('--query', help='Query to run after adding documents')
    add_parser.add_argument('--top-k', type=int, default=3, help='Number of results (default: 3)')

    # Verbose flag
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        agent = create_cloudera_agent()
    except Exception as e:
        print(f"Failed to create agent: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that config.json exists and is properly formatted")
        print("  2. Verify environment variables are set correctly")
        print("  3. See README.md for configuration instructions")
        sys.exit(1)

    # Execute command
    if args.command == 'health':
        success = health_check(agent)
        sys.exit(0 if success else 1)

    elif args.command == 'query':
        test_query(agent, args.query, args.top_k)

    elif args.command == 'interactive':
        interactive_mode(agent)

    elif args.command == 'add':
        print(f"Adding {len(args.documents)} documents...")
        agent.add_knowledge(args.documents)
        print(f"✓ Added {len(args.documents)} documents\n")

        if args.query:
            test_query(agent, args.query, args.top_k)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

