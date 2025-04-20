"""
Tests for NodeRegistry module.

This module tests the functionality of the NodeRegistry class.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch

from src.graphrag_engine.knowledge_graph.node_registry import NodeRegistry


class TestNodeRegistry:
    """Test suite for NodeRegistry class."""

    def test_init(self):
        """Test initialization of NodeRegistry."""
        # Test with default values
        registry = NodeRegistry()
        assert registry.similarity_threshold == 0.85
        assert not registry.case_sensitive
        assert registry.nodes == {}
        
        # Test with custom values
        registry = NodeRegistry(similarity_threshold=0.75, case_sensitive=True)
        assert registry.similarity_threshold == 0.75
        assert registry.case_sensitive

    def test_register_entity(self, sample_entities):
        """Test registering a single entity."""
        registry = NodeRegistry()
        
        entity = sample_entities[0]  # John Smith
        document_id = "doc123"
        
        # Register entity
        node_id = registry.register_entity(entity, document_id)
        
        # Verify that entity was registered correctly
        assert node_id in registry.nodes
        node = registry.nodes[node_id]
        assert node["text"] == entity["text"]
        assert node["type"] == entity["type"]
        assert document_id in node["documents"]

    def test_register_entities(self, sample_entities):
        """Test registering multiple entities."""
        registry = NodeRegistry()
        document_id = "doc123"
        
        # Register entities
        node_ids = registry.register_entities(sample_entities, document_id)
        
        # Verify results
        assert len(node_ids) == len(sample_entities)
        
        # Check that all entities were registered
        for entity, node_id in zip(sample_entities, node_ids):
            node = registry.nodes[node_id]
            assert node["text"] == entity["text"]
            assert node["type"] == entity["type"]
            assert document_id in node["documents"]

    def test_get_node(self, sample_entities):
        """Test retrieving a node by ID."""
        registry = NodeRegistry()
        
        # Register an entity
        entity = sample_entities[0]
        node_id = registry.register_entity(entity, "doc123")
        
        # Get node by ID
        node = registry.get_node(node_id)
        
        # Verify that the correct node was retrieved
        assert node is not None
        assert node["text"] == entity["text"]
        assert node["type"] == entity["type"]
        
        # Test with non-existent ID
        node = registry.get_node("non-existent-id")
        assert node is None

    def test_get_node_by_entity(self, sample_entities):
        """Test retrieving a node by entity text."""
        registry = NodeRegistry()
        
        # Register entities
        for entity in sample_entities:
            registry.register_entity(entity, "doc123")
        
        # Get node by entity text
        node = registry.get_node_by_entity("John Smith")
        
        # Verify that the correct node was retrieved
        assert node is not None
        assert node["text"] == "John Smith"
        assert node["type"] == "PERSON"
        
        # Test with case-insensitive matching
        node = registry.get_node_by_entity("john smith")
        assert node is not None
        assert node["text"] == "John Smith"
        
        # Test with case-sensitive registry
        registry = NodeRegistry(case_sensitive=True)
        for entity in sample_entities:
            registry.register_entity(entity, "doc123")
        
        node = registry.get_node_by_entity("John Smith")
        assert node is not None
        
        node = registry.get_node_by_entity("john smith")
        assert node is None
        
        # Test with non-existent entity
        node = registry.get_node_by_entity("Non-existent Entity")
        assert node is None

    def test_find_similar_entity(self, sample_entities):
        """Test finding a similar entity."""
        registry = NodeRegistry(similarity_threshold=0.75)
        
        # Register entities
        for entity in sample_entities:
            registry.register_entity(entity, "doc123")
        
        # Test with exact match
        node_id = registry._find_similar_entity("John Smith", "PERSON")
        assert node_id is not None
        
        # Test with similar text
        node_id = registry._find_similar_entity("John Smith Jr.", "PERSON")
        assert node_id is not None
        assert registry.nodes[node_id]["text"] == "John Smith"
        
        # Test with different entity type
        node_id = registry._find_similar_entity("John Smith", "ORGANIZATION")
        assert node_id is None
        
        # Test with non-existent entity
        node_id = registry._find_similar_entity("Completely Different Name", "PERSON")
        assert node_id is None

    def test_merge_nodes(self, sample_entities):
        """Test merging two nodes."""
        registry = NodeRegistry()
        
        # Create two similar entities
        entity1 = {
            "text": "John Smith",
            "type": "PERSON",
            "start_pos": 0,
            "end_pos": 10,
            "metadata": {"title": "Mr."}
        }
        
        entity2 = {
            "text": "John Smith",
            "type": "PERSON",
            "start_pos": 50,
            "end_pos": 60,
            "metadata": {"role": "Engineer"}
        }
        
        # Register entities with different document IDs
        node1_id = registry.register_entity(entity1, "doc1")
        node2_id = registry.register_entity(entity2, "doc2")
        
        # Both nodes should exist initially
        assert node1_id in registry.nodes
        assert node2_id in registry.nodes
        
        # Merge nodes
        merged_id = registry.merge_nodes(node1_id, node2_id)
        
        # The second node should be removed
        assert merged_id == node1_id
        assert node1_id in registry.nodes
        assert node2_id not in registry.nodes
        
        # The merged node should contain information from both nodes
        merged_node = registry.nodes[merged_id]
        assert "doc1" in merged_node["documents"]
        assert "doc2" in merged_node["documents"]
        assert "title" in merged_node["metadata"]
        assert "role" in merged_node["metadata"]

    def test_export_import_json(self):
        """Test exporting and importing registry to/from JSON."""
        registry = NodeRegistry()
        
        # Add sample nodes
        registry.register_entity({"text": "John Smith", "type": "PERSON"}, "doc1")
        registry.register_entity({"text": "Acme Corp", "type": "ORGANIZATION"}, "doc2")
        
        # Export to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        registry.export_to_json(tmp_path)
        
        # Create a new registry and import from the file
        new_registry = NodeRegistry()
        new_registry.import_from_json(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Verify that the import was successful
        assert len(new_registry.nodes) == len(registry.nodes)
        
        # Check that all nodes were imported correctly
        for node_id, node in registry.nodes.items():
            assert node_id in new_registry.nodes
            imported_node = new_registry.nodes[node_id]
            assert imported_node["text"] == node["text"]
            assert imported_node["type"] == node["type"]
            assert imported_node["documents"] == node["documents"]

    def test_get_all_nodes(self):
        """Test retrieving all nodes."""
        registry = NodeRegistry()
        
        # Add sample nodes
        registry.register_entity({"text": "John Smith", "type": "PERSON"}, "doc1")
        registry.register_entity({"text": "Acme Corp", "type": "ORGANIZATION"}, "doc2")
        
        # Get all nodes
        nodes = registry.get_all_nodes()
        
        # Verify results
        assert len(nodes) == 2
        assert isinstance(nodes, list)
        
        # Check that nodes contain the expected information
        for node in nodes:
            assert "id" in node
            assert "text" in node
            assert "type" in node
            assert "documents" in node
            
            # Check that node texts match expected values
            assert node["text"] in ["John Smith", "Acme Corp"]

    def test_string_similarity(self):
        """Test string similarity calculation."""
        registry = NodeRegistry()
        
        # Test with identical strings
        similarity = registry._string_similarity("John Smith", "John Smith")
        assert similarity == 1.0
        
        # Test with similar strings
        similarity = registry._string_similarity("John Smith", "John Smithson")
        assert 0.7 < similarity < 1.0
        
        # Test with completely different strings
        similarity = registry._string_similarity("John Smith", "Alice Johnson")
        assert similarity < 0.5
        
        # Test with case differences
        similarity = registry._string_similarity("John Smith", "john smith")
        assert similarity == 1.0  # Default is case-insensitive
        
        # Test with case-sensitive registry
        registry = NodeRegistry(case_sensitive=True)
        similarity = registry._string_similarity("John Smith", "john smith")
        assert similarity < 1.0
