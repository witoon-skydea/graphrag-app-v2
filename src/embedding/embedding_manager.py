"""
Embedding Manager Module

This module manages the creation of vector embeddings from text using
various embedding models, both local (Ollama) and from external APIs.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
import time
import numpy as np
import requests

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages the creation of vector embeddings from text.
    
    Supports multiple embedding sources:
    - Ollama (local)
    - OpenAI API
    - Anthropic API
    - Google Gemini API
    - OpenRouter API
    """
    
    def __init__(
        self,
        embedding_source: str = "ollama",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        dimensions: int = 768,
        retry_attempts: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the EmbeddingManager.
        
        Args:
            embedding_source: Source of embeddings ('ollama', 'openai', 'anthropic', 'gemini', 'openrouter')
            model_name: Name of the model to use (specific to the chosen source)
            api_key: API key for external APIs (required for OpenAI, Anthropic, Gemini, OpenRouter)
            api_endpoint: Custom API endpoint (optional, defaults to standard endpoints)
            dimensions: Dimensions of the embedding vectors (used for normalization)
            retry_attempts: Number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
        """
        self.embedding_source = embedding_source
        self.model_name = model_name
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.dimensions = dimensions
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Set default model names based on source
        if not self.model_name:
            if embedding_source == "ollama":
                self.model_name = "mxbai-embed-large"  # Updated default to mxbai-embed-large
            elif embedding_source == "openai":
                self.model_name = "text-embedding-3-small"
            elif embedding_source == "anthropic":
                self.model_name = "claude-3-haiku-20240307"
            elif embedding_source == "gemini":
                self.model_name = "embedding-001"
            elif embedding_source == "openrouter":
                self.model_name = "openai/text-embedding-3-small"
        
        # Set default API endpoints based on source
        if not self.api_endpoint:
            if embedding_source == "ollama":
                self.api_endpoint = "http://localhost:11434/api/embeddings"  # Fixed endpoint to use plural form
            elif embedding_source == "openai":
                self.api_endpoint = "https://api.openai.com/v1/embeddings"
            elif embedding_source == "anthropic":
                self.api_endpoint = "https://api.anthropic.com/v1/embeddings"
            elif embedding_source == "gemini":
                self.api_endpoint = "https://generativelanguage.googleapis.com/v1/models"
            elif embedding_source == "openrouter":
                self.api_endpoint = "https://openrouter.ai/api/v1/embeddings"
        
        # Set dimensions based on model
        if embedding_source == "ollama" and self.model_name == "mxbai-embed-large":
            self.dimensions = 1024  # mxbai-embed-large has 1024 dimensions
        
        logger.info(f"EmbeddingManager initialized with {embedding_source} ({self.model_name})")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding, or None if embedding fails
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            return None
        
        # Truncate text if it's too long
        # Different embedding sources have different max token limits
        max_length = 8000  # Default max length
        if self.embedding_source == "openai" and self.model_name == "text-embedding-3-small":
            max_length = 8191
        elif self.embedding_source == "openai" and self.model_name == "text-embedding-3-large":
            max_length = 8191
        
        if len(text) > max_length:
            logger.warning(f"Text exceeds maximum length ({len(text)} > {max_length}), truncating")
            text = text[:max_length]
        
        # Call the appropriate embedding function based on the source
        for attempt in range(self.retry_attempts):
            try:
                if self.embedding_source == "ollama":
                    return self._get_ollama_embedding(text)
                elif self.embedding_source == "openai":
                    return self._get_openai_embedding(text)
                elif self.embedding_source == "anthropic":
                    return self._get_anthropic_embedding(text)
                elif self.embedding_source == "gemini":
                    return self._get_gemini_embedding(text)
                elif self.embedding_source == "openrouter":
                    return self._get_openrouter_embedding(text)
                else:
                    logger.error(f"Unsupported embedding source: {self.embedding_source}")
                    return None
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to get embedding after {self.retry_attempts} attempts")
                    return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, one for each input text
        """
        embeddings = []
        
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        response = requests.post(self.api_endpoint, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle both 'embedding' and 'embeddings' keys in the response
            if "embedding" in result:
                return self._normalize_vector(result["embedding"])
            elif "embeddings" in result:
                # Extract the first embedding if it's an array
                embeddings = result["embeddings"]
                if len(embeddings) > 0:
                    if isinstance(embeddings[0], list):
                        return self._normalize_vector(embeddings[0])
                    else:
                        return self._normalize_vector(embeddings)
                else:
                    logger.error("Ollama API returned empty embeddings array")
                    raise ValueError("Ollama API returned empty embeddings array")
            else:
                logger.error(f"Unexpected Ollama API response: {result}")
                raise ValueError("Ollama API response did not contain embedding data")
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise ValueError(f"Ollama API error: {response.status_code}")
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API."""
        if not self.api_key:
            raise ValueError("API key is required for OpenAI embeddings")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": self.model_name,
            "encoding_format": "float"
        }
        
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                return self._normalize_vector(result["data"][0]["embedding"])
            else:
                logger.error(f"Unexpected OpenAI API response: {result}")
                raise ValueError("OpenAI API response did not contain an embedding")
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise ValueError(f"OpenAI API error: {response.status_code}")
    
    def _get_anthropic_embedding(self, text: str) -> List[float]:
        """Get embedding from Anthropic API."""
        if not self.api_key:
            raise ValueError("API key is required for Anthropic embeddings")
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "embedding" in result:
                return self._normalize_vector(result["embedding"])
            else:
                logger.error(f"Unexpected Anthropic API response: {result}")
                raise ValueError("Anthropic API response did not contain an embedding")
        else:
            logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
            raise ValueError(f"Anthropic API error: {response.status_code}")
    
    def _get_gemini_embedding(self, text: str) -> List[float]:
        """Get embedding from Google Gemini API."""
        if not self.api_key:
            raise ValueError("API key is required for Gemini embeddings")
        
        api_url = f"{self.api_endpoint}/{self.model_name}:embedContent?key={self.api_key}"
        
        payload = {
            "content": {
                "parts": [
                    {
                        "text": text
                    }
                ]
            },
            "taskType": "RETRIEVAL_DOCUMENT"
        }
        
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "embedding" in result and "values" in result["embedding"]:
                return self._normalize_vector(result["embedding"]["values"])
            else:
                logger.error(f"Unexpected Gemini API response: {result}")
                raise ValueError("Gemini API response did not contain an embedding")
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            raise ValueError(f"Gemini API error: {response.status_code}")
    
    def _get_openrouter_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenRouter API."""
        if not self.api_key:
            raise ValueError("API key is required for OpenRouter embeddings")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": self.model_name
        }
        
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                return self._normalize_vector(result["data"][0]["embedding"])
            else:
                logger.error(f"Unexpected OpenRouter API response: {result}")
                raise ValueError("OpenRouter API response did not contain an embedding")
        else:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            raise ValueError(f"OpenRouter API error: {response.status_code}")
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: The vector to normalize
            
        Returns:
            Normalized vector
        """
        # Convert to numpy array for efficient operations
        vec_array = np.array(vector, dtype=np.float32)
        
        # Calculate vector norm
        norm = np.linalg.norm(vec_array)
        
        # Avoid division by zero
        if norm > 0:
            normalized = vec_array / norm
        else:
            normalized = vec_array
        
        return normalized.tolist()
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1_array = np.array(vec1, dtype=np.float32)
        vec2_array = np.array(vec2, dtype=np.float32)
        
        # Calculate dot product
        dot_product = np.dot(vec1_array, vec2_array)
        
        # Calculate magnitudes
        mag1 = np.linalg.norm(vec1_array)
        mag2 = np.linalg.norm(vec2_array)
        
        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            cosine_similarity = dot_product / (mag1 * mag2)
            # Clip to range [0, 1] to handle floating point errors
            return max(0.0, min(1.0, cosine_similarity))
        else:
            return 0.0
