"""
Tests for EntityExtractor module.

This module tests the functionality of the EntityExtractor class.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from src.graphrag_engine.knowledge_graph.entity_extractor import EntityExtractor


class TestEntityExtractor:
    """Test suite for EntityExtractor class."""

    def test_init(self):
        """Test initialization of EntityExtractor."""
        # Test with default values
        extractor = EntityExtractor()
        assert extractor.extraction_method == "ollama"
        assert extractor.model_name == "llama2"
        assert extractor.api_key is None
        assert extractor.endpoint is not None
        assert len(extractor.entity_types) > 0
        
        # Test with custom values
        entity_types = ["PERSON", "ORGANIZATION"]
        extractor = EntityExtractor(
            extraction_method="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            endpoint="https://custom-endpoint.com",
            entity_types=entity_types
        )
        assert extractor.extraction_method == "openai"
        assert extractor.model_name == "gpt-3.5-turbo"
        assert extractor.api_key == "test-api-key"
        assert extractor.endpoint == "https://custom-endpoint.com"
        assert extractor.entity_types == entity_types

    def test_create_prompt(self):
        """Test prompt creation."""
        extractor = EntityExtractor(entity_types=["PERSON", "ORGANIZATION"])
        prompt = extractor._create_prompt("John works at Acme Corp.")
        
        # Check that prompt contains entity types
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        
        # Check that prompt contains input text
        assert "John works at Acme Corp." in prompt
        
        # Check that prompt asks for JSON response
        assert "JSON Response" in prompt

    @patch('requests.post')
    def test_extract_with_ollama(self, mock_post, mock_response):
        """Test extraction using Ollama."""
        # Configure mock
        mock_response.json.return_value = {
            "response": "[{\"text\": \"John Smith\", \"type\": \"PERSON\", \"start_pos\": 0, \"end_pos\": 10}]"
        }
        mock_post.return_value = mock_response
        
        # Create extractor and test
        extractor = EntityExtractor(extraction_method="ollama")
        entities = extractor._extract_with_ollama("John Smith works at Acme Corp.")
        
        # Verify results
        assert len(entities) == 1
        assert entities[0]["text"] == "John Smith"
        assert entities[0]["type"] == "PERSON"
        
        # Verify that the correct endpoint was used
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/generate"
        assert kwargs["json"]["model"] == "llama2"
        assert not kwargs["json"]["stream"]

    @patch('requests.post')
    def test_extract_with_openai(self, mock_post, mock_response):
        """Test extraction using OpenAI API."""
        # Configure mock
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "[{\"text\": \"John Smith\", \"type\": \"PERSON\", \"start_pos\": 0, \"end_pos\": 10}]"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Create extractor and test
        extractor = EntityExtractor(
            extraction_method="openai",
            api_key="test-api-key"
        )
        entities = extractor._extract_with_openai("John Smith works at Acme Corp.")
        
        # Verify results
        assert len(entities) == 1
        assert entities[0]["text"] == "John Smith"
        assert entities[0]["type"] == "PERSON"
        
        # Verify that the correct endpoint and headers were used
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.openai.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    def test_extract_with_openai_missing_api_key(self):
        """Test that extraction with OpenAI raises error when API key is missing."""
        extractor = EntityExtractor(extraction_method="openai")
        
        with pytest.raises(ValueError) as excinfo:
            extractor._extract_with_openai("John Smith works at Acme Corp.")
        
        assert "API key required" in str(excinfo.value)

    @patch('requests.post')
    def test_extract_entities_empty_text(self, mock_post):
        """Test that extract_entities handles empty text properly."""
        extractor = EntityExtractor()
        
        # Test with empty string
        entities = extractor.extract_entities("")
        assert entities == []
        
        # Test with None
        entities = extractor.extract_entities(None)
        assert entities == []
        
        # Test with whitespace
        entities = extractor.extract_entities("   ")
        assert entities == []
        
        # Verify that no API calls were made
        mock_post.assert_not_called()

    @patch('requests.post')
    def test_extract_entities_with_long_text(self, mock_post, mock_response):
        """Test that extract_entities truncates long text."""
        # Configure mock
        mock_response.json.return_value = {
            "response": "[{\"text\": \"John\", \"type\": \"PERSON\", \"start_pos\": 0, \"end_pos\": 4}]"
        }
        mock_post.return_value = mock_response
        
        # Create extractor
        extractor = EntityExtractor()
        
        # Create text longer than 10000 characters
        long_text = "a" * 15000
        
        # Call extract_entities
        extractor.extract_entities(long_text)
        
        # Verify that text was truncated
        args, kwargs = mock_post.call_args
        prompt = kwargs["json"]["prompt"]
        assert len(prompt) < 15000

    @patch('requests.post')
    def test_extract_entities_api_error(self, mock_post):
        """Test that extract_entities handles API errors gracefully."""
        # Configure mock to simulate error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # Create extractor
        extractor = EntityExtractor()
        
        # Call extract_entities
        entities = extractor.extract_entities("John Smith works at Acme Corp.")
        
        # Verify that empty list is returned on error
        assert entities == []

    def test_extract_with_rules(self):
        """Test rule-based entity extraction."""
        extractor = EntityExtractor(extraction_method="rule_based")
        
        text = "John Smith works at Acme Corp. He paid $1000 on January 15, 2023. Contact him at john@example.com."
        
        entities = extractor._extract_with_rules(text)
        
        # Verify that entities were extracted
        assert len(entities) > 0
        
        # Check for specific entity types
        entity_types = [entity["type"] for entity in entities]
        assert "PERSON" in entity_types
        assert "DATE" in entity_types or "TIME" in entity_types
        assert "MONEY" in entity_types or "EMAIL" in entity_types
