#!/usr/bin/env python3
"""
Example usage of Cloudera Inference With CursorAI
"""

import logging
from agents import create_cloudera_agent

# Configure logging (optional - set level to DEBUG for detailed logs)
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("Creating Cloudera Agent...")

    # Create agent (will use environment variables or /tmp/jwt)
    agent = create_cloudera_agent()

    print("Agent created successfully!")
    print(f"Agent stats: {agent.get_stats()}\n")

    # Add knowledge base documents
    print("Adding knowledge base documents...")
    knowledge_base = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Embeddings are vector representations of text that capture semantic meaning.",
        "RAG (Retrieval Augmented Generation) combines retrieval of relevant documents with language generation.",
        "Cloudera provides enterprise data cloud solutions for machine learning and analytics.",
        "Vector stores are databases optimized for storing and searching high-dimensional vectors.",
        "The NVIDIA nv-embedqa-e5-v5 model is designed for question-answering and semantic search tasks.",
    ]

    agent.add_knowledge(knowledge_base)
    print(f"Added {len(knowledge_base)} documents to knowledge base\n")

    # Example queries
    queries = [
        "What is Python?",
        "How does machine learning work?",
        "What are embeddings used for?",
        "Tell me about Cloudera",
    ]

    print("Running queries...\n")
    for query in queries:
        print(f"Query: {query}")

        # Try to use LLM if available, otherwise use retrieval-only
        try:
            result = agent.answer_with_llm(query, top_k=2, use_context=True)
            print(f"Answer: {result['answer']}")
            if result['context']:
                print(f"Used {result['num_results']} context sources:")
                for i, context in enumerate(result['context'], 1):
                    print(f"  [{i}] Similarity: {context['similarity']:.4f}")
                    print(f"      Text: {context['text'][:80]}...")
        except ValueError as e:
            # LLM not configured, fall back to retrieval-only
            print("(LLM not configured, using retrieval-only mode)")
            result = agent.answer_with_context(query, top_k=2)
            print(f"Found {result['num_results']} relevant contexts:")
            for i, context in enumerate(result['context'], 1):
                print(f"  [{i}] Similarity: {context['similarity']:.4f}")
                print(f"      Text: {context['text'][:80]}...")
        print()

    print("Example completed!")


if __name__ == "__main__":
    main()

