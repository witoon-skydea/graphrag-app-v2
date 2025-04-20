"""
Conftest for GraphRAG tests

This file contains pytest fixtures for testing GraphRAG components.
"""

import os
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add root directory to sys.path to allow importing from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphrag_engine.knowledge_graph.entity_extractor import EntityExtractor
from src.graphrag_engine.knowledge_graph.relationship_identifier import RelationshipIdentifier
from src.graphrag_engine.knowledge_graph.node_registry import NodeRegistry
from src.graphrag_engine.knowledge_graph.edge_generator import EdgeGenerator
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.graphrag_engine.vector_db.weaviate_client import VectorDBClient


@pytest.fixture
def sample_text():
    """Sample text for entity extraction and relationship identification."""
    return """
    John Smith works for Acme Corporation as a senior engineer. 
    Acme Corporation is headquartered in New York City.
    John Smith graduated from Stanford University in 2010.
    Sarah Johnson is the CEO of Acme Corporation.
    """


@pytest.fixture
def sample_entities():
    """Sample extracted entities."""
    return [
        {
            "text": "John Smith",
            "type": "PERSON",
            "start_pos": 5,
            "end_pos": 15,
            "metadata": {}
        },
        {
            "text": "Acme Corporation",
            "type": "ORGANIZATION",
            "start_pos": 26,
            "end_pos": 42,
            "metadata": {}
        },
        {
            "text": "New York City",
            "type": "LOCATION",
            "start_pos": 83,
            "end_pos": 96,
            "metadata": {}
        },
        {
            "text": "Stanford University",
            "type": "ORGANIZATION",
            "start_pos": 127,
            "end_pos": 147,
            "metadata": {}
        },
        {
            "text": "2010",
            "type": "DATE",
            "start_pos": 151,
            "end_pos": 155,
            "metadata": {}
        },
        {
            "text": "Sarah Johnson",
            "type": "PERSON",
            "start_pos": 161,
            "end_pos": 174,
            "metadata": {}
        }
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships between entities."""
    return [
        {
            "source": "John Smith",
            "source_type": "PERSON",
            "target": "Acme Corporation",
            "target_type": "ORGANIZATION",
            "relationship": "WORKS_FOR",
            "confidence": 0.9,
            "bidirectional": False,
            "metadata": {}
        },
        {
            "source": "Acme Corporation",
            "source_type": "ORGANIZATION",
            "target": "New York City",
            "target_type": "LOCATION",
            "relationship": "HEADQUARTERED_IN",
            "confidence": 0.85,
            "bidirectional": False,
            "metadata": {}
        },
        {
            "source": "John Smith",
            "source_type": "PERSON",
            "target": "Stanford University",
            "target_type": "ORGANIZATION",
            "relationship": "GRADUATED_FROM",
            "confidence": 0.8,
            "bidirectional": False,
            "metadata": {}
        },
        {
            "source": "Sarah Johnson",
            "source_type": "PERSON",
            "target": "Acme Corporation",
            "target_type": "ORGANIZATION",
            "relationship": "WORKS_FOR",
            "confidence": 0.95,
            "bidirectional": False,
            "metadata": {}
        }
    ]


@pytest.fixture
def mock_entity_extractor(sample_entities):
    """Mock EntityExtractor for testing."""
    mock = Mock(spec=EntityExtractor)
    mock.extract_entities.return_value = sample_entities
    mock.entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]
    return mock


@pytest.fixture
def mock_relationship_identifier(sample_relationships):
    """Mock RelationshipIdentifier for testing."""
    mock = Mock(spec=RelationshipIdentifier)
    mock.identify_relationships.return_value = sample_relationships
    mock.relationship_types = ["WORKS_FOR", "HEADQUARTERED_IN", "GRADUATED_FROM"]
    mock.confidence_threshold = 0.5
    return mock


@pytest.fixture
def temp_persist_path():
    """Temporary directory for persisting data during tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def node_registry():
    """NodeRegistry instance for testing."""
    return NodeRegistry(similarity_threshold=0.85, case_sensitive=False)


@pytest.fixture
def edge_generator():
    """EdgeGenerator instance for testing."""
    return EdgeGenerator(confidence_threshold=0.5)


@pytest.fixture
def mock_weaviate_client():
    """Mock VectorDBClient for testing."""
    mock = Mock(spec=VectorDBClient)
    mock.check_connection.return_value = True
    mock.create_schema.return_value = True
    mock.add_document_chunk.return_value = "mock-uuid"
    mock.add_document_chunks_batch.return_value = 10
    mock.search_by_text.return_value = []
    mock.search_by_vector.return_value = []
    return mock


@pytest.fixture
def mock_response():
    """Mock response for API requests."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "[{\"text\": \"John Smith\", \"type\": \"PERSON\", \"start_pos\": 5, \"end_pos\": 15}]"
                }
            }
        ]
    }
    return mock
