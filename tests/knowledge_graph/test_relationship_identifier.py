"""
Tests for RelationshipIdentifier module.

This module tests the functionality of the RelationshipIdentifier class.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graphrag_engine.knowledge_graph.relationship_identifier import RelationshipIdentifier


class TestRelationshipIdentifier:
    """Test suite for RelationshipIdentifier class."""

    def test_init(self):
        """Test initialization of RelationshipIdentifier."""
        # Test with default values
        identifier = RelationshipIdentifier()
        assert identifier.identification_method == "ollama"
        assert identifier.model_name == "llama2"
        assert identifier.api_key is None
        assert identifier.endpoint is not None
        assert len(identifier.relationship_types) > 0
        assert identifier.confidence_threshold == 0.5
        
        # Test with custom values
        relationship_types = ["WORKS_FOR", "LOCATED_IN"]
        identifier = RelationshipIdentifier(
            identification_method="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            endpoint="https://custom-endpoint.com",
            relationship_types=relationship_types,
            threshold=0.7
        )
        assert identifier.identification_method == "openai"
        assert identifier.model_name == "gpt-3.5-turbo"
        assert identifier.api_key == "test-api-key"
        assert identifier.endpoint == "https://custom-endpoint.com"
        assert identifier.relationship_types == relationship_types
        assert identifier.confidence_threshold == 0.7

    def test_create_entity_pairs(self):
        """Test creation of entity pairs."""
        identifier = RelationshipIdentifier()
        
        entities = [
            {"text": "John", "type": "PERSON"},
            {"text": "Acme", "type": "ORGANIZATION"},
            {"text": "New York", "type": "LOCATION"}
        ]
        
        # Test with default batch size
        pairs_batches = identifier._create_entity_pairs(entities)
        assert len(pairs_batches) == 1
        assert len(pairs_batches[0]) == 3  # 3 pairs: (0,1), (0,2), (1,2)
        
        # Test with custom batch size
        pairs_batches = identifier._create_entity_pairs(entities, batch_size=2)
        assert len(pairs_batches) == 2
        assert len(pairs_batches[0]) == 2
        assert len(pairs_batches[1]) == 1

    def test_create_prompt(self):
        """Test prompt creation."""
        identifier = RelationshipIdentifier(relationship_types=["WORKS_FOR", "LOCATED_IN"])
        
        entity_pairs = [
            (
                {"text": "John", "type": "PERSON"},
                {"text": "Acme", "type": "ORGANIZATION"}
            )
        ]
        
        prompt = identifier._create_prompt(entity_pairs)
        
        # Check that prompt contains relationship types
        assert "WORKS_FOR" in prompt
        assert "LOCATED_IN" in prompt
        
        # Check that prompt contains entity information
        assert "John" in prompt
        assert "PERSON" in prompt
        assert "Acme" in prompt
        assert "ORGANIZATION" in prompt
        
        # Check that prompt asks for JSON response
        assert "JSON Response" in prompt
        
        # Test with context text
        context_text = "John works at Acme Corp in New York."
        prompt_with_context = identifier._create_prompt(entity_pairs, context_text)
        
        # Check that context is included
        assert context_text in prompt_with_context

    @patch('requests.post')
    def test_identify_with_ollama(self, mock_post, sample_entities, mock_response):
        """Test relationship identification using Ollama."""
        # Configure mock
        mock_response.json.return_value = {
            "response": """[
                {
                    "pair_id": 1,
                    "source": "John Smith",
                    "source_type": "PERSON",
                    "target": "Acme Corporation",
                    "target_type": "ORGANIZATION",
                    "relationship": "WORKS_FOR",
                    "confidence": 0.9,
                    "bidirectional": false
                }
            ]"""
        }
        mock_post.return_value = mock_response
        
        # Create identifier and test
        identifier = RelationshipIdentifier(identification_method="ollama")
        text = "John Smith works for Acme Corporation."
        relationships = identifier._identify_with_ollama(sample_entities, text)
        
        # Verify results
        assert len(relationships) == 1
        assert relationships[0]["source"] == "John Smith"
        assert relationships[0]["target"] == "Acme Corporation"
        assert relationships[0]["relationship"] == "WORKS_FOR"
        assert relationships[0]["confidence"] == 0.9
        
        # Verify that the correct endpoint was used
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/generate"
        assert kwargs["json"]["model"] == "llama2"

    @patch('requests.post')
    def test_identify_with_openai(self, mock_post, sample_entities, mock_response):
        """Test relationship identification using OpenAI API."""
        # Configure mock
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """[
                            {
                                "pair_id": 1,
                                "source": "John Smith",
                                "source_type": "PERSON",
                                "target": "Acme Corporation",
                                "target_type": "ORGANIZATION",
                                "relationship": "WORKS_FOR",
                                "confidence": 0.9,
                                "bidirectional": false
                            }
                        ]"""
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Create identifier and test
        identifier = RelationshipIdentifier(
            identification_method="openai",
            api_key="test-api-key"
        )
        text = "John Smith works for Acme Corporation."
        relationships = identifier._identify_with_openai(sample_entities, text)
        
        # Verify results
        assert len(relationships) == 1
        assert relationships[0]["source"] == "John Smith"
        assert relationships[0]["target"] == "Acme Corporation"
        assert relationships[0]["relationship"] == "WORKS_FOR"
        assert relationships[0]["confidence"] == 0.9
        
        # Verify that the correct endpoint and headers were used
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.openai.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    def test_identify_with_openai_missing_api_key(self, sample_entities):
        """Test that identification with OpenAI raises error when API key is missing."""
        identifier = RelationshipIdentifier(identification_method="openai")
        
        with pytest.raises(ValueError) as excinfo:
            identifier._identify_with_openai(sample_entities, "John Smith works for Acme Corporation.")
        
        assert "API key required" in str(excinfo.value)

    def test_identify_relationships_not_enough_entities(self):
        """Test that identify_relationships handles case with too few entities."""
        identifier = RelationshipIdentifier()
        
        # Test with empty list
        relationships = identifier.identify_relationships([])
        assert relationships == []
        
        # Test with single entity
        relationships = identifier.identify_relationships([{"text": "John", "type": "PERSON"}])
        assert relationships == []

    def test_identify_with_rules(self, sample_entities):
        """Test rule-based relationship identification."""
        identifier = RelationshipIdentifier(identification_method="rule_based")
        
        text = "John Smith works for Acme Corporation. Acme Corporation is headquartered in New York City."
        
        relationships = identifier._identify_with_rules(sample_entities, text)
        
        # Verify that relationships were identified
        assert len(relationships) > 0
        
        # Check for specific relationship types
        found_work_relation = False
        found_location_relation = False
        
        for rel in relationships:
            if rel["relationship"] == "WORKS_FOR" and rel["source"] == "John Smith" and rel["target"] == "Acme Corporation":
                found_work_relation = True
            elif rel["relationship"] == "LOCATED_IN" and rel["source"] == "Acme Corporation" and rel["target"] == "New York City":
                found_location_relation = True
        
        assert found_work_relation or found_location_relation

    def test_identify_with_proximity(self, sample_entities):
        """Test proximity-based relationship identification."""
        identifier = RelationshipIdentifier(identification_method="proximity")
        
        text = "John Smith works for Acme Corporation. Acme Corporation is headquartered in New York City."
        
        relationships = identifier._identify_with_proximity(sample_entities, text)
        
        # Verify that relationships were identified
        assert len(relationships) > 0
        
        # Check that proximity-based confidence is calculated
        for rel in relationships:
            assert "confidence" in rel
            assert 0.0 <= rel["confidence"] <= 1.0
            
            # Check that metadata includes distance information
            assert "metadata" in rel
            assert "distance" in rel["metadata"]

    def test_infer_relationship_from_types(self):
        """Test inference of relationship types based on entity types."""
        identifier = RelationshipIdentifier()
        
        # Test known entity type pairs
        rel_type = identifier._infer_relationship_from_types("PERSON", "ORGANIZATION")
        assert rel_type == "WORKS_FOR"
        
        rel_type = identifier._infer_relationship_from_types("ORGANIZATION", "LOCATION")
        assert rel_type == "HEADQUARTERED_IN"
        
        # Test unknown entity type pair
        rel_type = identifier._infer_relationship_from_types("LANGUAGE", "CONCEPT")
        assert rel_type is None

    def test_identify_with_cooccurrence(self, sample_entities):
        """Test co-occurrence-based relationship identification."""
        identifier = RelationshipIdentifier(identification_method="cooccurrence")
        
        text = """
        John Smith is a senior engineer at Acme Corporation.
        Sarah Johnson is the CEO of Acme Corporation.
        Acme Corporation has its headquarters in New York City.
        John Smith studied at Stanford University.
        """
        
        relationships = identifier._identify_with_cooccurrence(sample_entities, text)
        
        # Verify that relationships were identified
        assert len(relationships) > 0
        
        # Check that relationships have co-occurrence metadata
        for rel in relationships:
            assert "metadata" in rel
            assert "common_segments" in rel["metadata"]
            assert "source_frequency" in rel["metadata"]
            assert "target_frequency" in rel["metadata"]
