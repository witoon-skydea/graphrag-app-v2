"""
Integration test for the EmbeddingManager with Ollama and mxbai-embed-large model.
"""

import os
import sys
import time
import numpy as np
import pytest

# Add the src directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.embedding.embedding_manager import EmbeddingManager

@pytest.mark.integration
def test_embedding_manager_with_mxbai_embed_large():
    """Test the EmbeddingManager with the mxbai-embed-large model."""
    # Initialize the EmbeddingManager with the recommended configuration
    embedding_manager = EmbeddingManager(
        embedding_source="ollama",
        model_name="mxbai-embed-large"
    )
    
    # Check that the dimensions are set correctly
    assert embedding_manager.dimensions == 1024
    
    # Test with a simple text
    text = "This is a test sentence for embedding."
    
    # Get the embedding
    embedding = embedding_manager.get_embedding(text)
    
    # Check that we got a valid embedding
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == 1024
    assert all(isinstance(x, float) for x in embedding)
    
    # Check that the vector is normalized (magnitude should be very close to 1)
    magnitude = np.linalg.norm(np.array(embedding))
    assert abs(magnitude - 1.0) < 1e-6

@pytest.mark.integration
def test_embedding_similarity_with_mxbai_embed_large():
    """Test similarity calculations with embeddings from mxbai-embed-large."""
    embedding_manager = EmbeddingManager(
        embedding_source="ollama",
        model_name="mxbai-embed-large"
    )
    
    # Similar sentences
    similar_text1 = "The quick brown fox jumps over the lazy dog."
    similar_text2 = "A fast auburn fox leaps above the sleepy canine."
    
    # Different sentence
    different_text = "Machine learning models process large amounts of data."
    
    # Get embeddings
    embedding1 = embedding_manager.get_embedding(similar_text1)
    embedding2 = embedding_manager.get_embedding(similar_text2)
    embedding3 = embedding_manager.get_embedding(different_text)
    
    # Calculate similarities
    similarity_similar = embedding_manager.calculate_similarity(embedding1, embedding2)
    similarity_different = embedding_manager.calculate_similarity(embedding1, embedding3)
    
    # Similar sentences should have higher similarity than different ones
    assert similarity_similar > similarity_different
    
    # The similarity ratio should be at least 2x
    assert similarity_similar / similarity_different > 2.0
    
    print(f"Similarity between similar sentences: {similarity_similar}")
    print(f"Similarity between different sentences: {similarity_different}")
    print(f"Similarity ratio: {similarity_similar / similarity_different:.2f}x")

@pytest.mark.integration
def test_embedding_performance_with_mxbai_embed_large():
    """Test the performance of the mxbai-embed-large model."""
    embedding_manager = EmbeddingManager(
        embedding_source="ollama",
        model_name="mxbai-embed-large"
    )
    
    # Short text
    short_text = "This is a short test sentence."
    
    # Medium text (a paragraph)
    medium_text = """
    This is a medium-length text that contains multiple sentences.
    It is designed to test the embedding performance with paragraphs.
    The embedding model should process this paragraph and generate
    a meaningful vector representation of its semantic content.
    """
    
    # Longer text
    long_text = medium_text * 3
    
    # Benchmark short text
    start_time = time.time()
    short_embedding = embedding_manager.get_embedding(short_text)
    short_time = time.time() - start_time
    
    # Benchmark medium text
    start_time = time.time()
    medium_embedding = embedding_manager.get_embedding(medium_text)
    medium_time = time.time() - start_time
    
    # Benchmark long text
    start_time = time.time()
    long_embedding = embedding_manager.get_embedding(long_text)
    long_time = time.time() - start_time
    
    # Print results
    print(f"Short text ({len(short_text)} chars) embedding time: {short_time:.4f} seconds")
    print(f"Medium text ({len(medium_text)} chars) embedding time: {medium_time:.4f} seconds")
    print(f"Long text ({len(long_text)} chars) embedding time: {long_time:.4f} seconds")
    
    # Check that all embeddings were generated successfully
    assert short_embedding is not None
    assert medium_embedding is not None
    assert long_embedding is not None
    
    # All embeddings should be the same length
    assert len(short_embedding) == len(medium_embedding) == len(long_embedding)
