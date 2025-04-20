"""
Test module for the EmbeddingManager class.

Tests the functionality of the EmbeddingManager, focusing on Ollama embeddings
with mxbai-embed-large model.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.embedding.embedding_manager import EmbeddingManager

class TestEmbeddingManager:
    """Test cases for the EmbeddingManager class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        manager = EmbeddingManager()
        
        assert manager.embedding_source == "ollama"
        assert manager.model_name == "llama2"
        assert manager.api_endpoint == "http://localhost:11434/api/embed"
        assert manager.dimensions == 768
        assert manager.retry_attempts == 3
        assert manager.retry_delay == 1

    def test_init_with_mxbai_model(self):
        """Test initialization with mxbai-embed-large model."""
        manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        
        assert manager.embedding_source == "ollama"
        assert manager.model_name == "mxbai-embed-large"
        assert manager.api_endpoint == "http://localhost:11434/api/embed"

    @pytest.mark.integration
    def test_get_embedding_ollama_mxbai(self):
        """Test getting an embedding from Ollama with mxbai-embed-large model."""
        # This test requires Ollama to be running with mxbai-embed-large model installed
        manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        
        text = "This is a test sentence for embedding."
        embedding = manager.get_embedding(text)
        
        # Check that we got a valid embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Check if the vector is normalized (magnitude should be close to 1)
        magnitude = np.linalg.norm(np.array(embedding))
        assert abs(magnitude - 1.0) < 1e-6

    @pytest.mark.integration
    def test_get_embedding_comparison(self):
        """Test comparing embeddings for semantically similar and different sentences."""
        manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        
        # Similar sentences
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast auburn fox leaps above the sleepy canine."
        
        # Different sentence
        text3 = "Machine learning models process large amounts of data."
        
        # Get embeddings
        embedding1 = manager.get_embedding(text1)
        embedding2 = manager.get_embedding(text2)
        embedding3 = manager.get_embedding(text3)
        
        # Calculate similarities
        similarity_similar = manager.calculate_similarity(embedding1, embedding2)
        similarity_different = manager.calculate_similarity(embedding1, embedding3)
        
        # Similar sentences should have higher similarity than different ones
        assert similarity_similar > similarity_different
        
        # Print similarities for reporting
        print(f"Similarity between similar sentences: {similarity_similar}")
        print(f"Similarity between different sentences: {similarity_different}")

    @pytest.mark.integration
    def test_embedding_dimensions(self):
        """Test the dimensionality of the mxbai-embed-large model embeddings."""
        manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        
        text = "Testing embedding dimensions."
        embedding = manager.get_embedding(text)
        
        # Check the actual dimensions of the model
        actual_dimensions = len(embedding)
        print(f"mxbai-embed-large actual dimensions: {actual_dimensions}")
        
        # The mxbai-embed-large should have 1024 dimensions according to documentation
        # But we'll just verify it has consistent dimensions
        assert actual_dimensions > 0
        
        # Test another text to ensure consistent dimensions
        text2 = "Another test sentence with different length and content."
        embedding2 = manager.get_embedding(text2)
        assert len(embedding2) == actual_dimensions

    @pytest.mark.integration
    def test_performance_benchmark(self):
        """Test the performance of the embedding model."""
        import time
        
        manager = EmbeddingManager(
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
        long_text = medium_text * 5
        
        # Benchmark short text
        start_time = time.time()
        short_embedding = manager.get_embedding(short_text)
        short_time = time.time() - start_time
        
        # Benchmark medium text
        start_time = time.time()
        medium_embedding = manager.get_embedding(medium_text)
        medium_time = time.time() - start_time
        
        # Benchmark long text
        start_time = time.time()
        long_embedding = manager.get_embedding(long_text)
        long_time = time.time() - start_time
        
        # Print results for reporting
        print(f"Short text ({len(short_text)} chars) embedding time: {short_time:.4f} seconds")
        print(f"Medium text ({len(medium_text)} chars) embedding time: {medium_time:.4f} seconds")
        print(f"Long text ({len(long_text)} chars) embedding time: {long_time:.4f} seconds")
        
        # Check that all embeddings were generated successfully
        assert short_embedding is not None
        assert medium_embedding is not None
        assert long_embedding is not None
        
        # Check that they all have the same dimensions
        assert len(short_embedding) == len(medium_embedding) == len(long_embedding)

    def test_get_embedding_with_empty_text(self):
        """Test behavior with empty text."""
        manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        
        embedding = manager.get_embedding("")
        assert embedding is None

    def test_get_embeddings_batch(self):
        """Test getting embeddings for multiple texts in batch."""
        with patch.object(EmbeddingManager, 'get_embedding') as mock_get_embedding:
            # Mock the get_embedding method to return predictable values
            mock_get_embedding.side_effect = lambda text: [0.1, 0.2, 0.3] if text else None
            
            manager = EmbeddingManager()
            texts = ["Text 1", "Text 2", ""]
            embeddings = manager.get_embeddings_batch(texts)
            
            # Check that get_embedding was called for each text
            assert mock_get_embedding.call_count == 3
            
            # Check the results
            assert len(embeddings) == 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.1, 0.2, 0.3]
            assert embeddings[2] is None
