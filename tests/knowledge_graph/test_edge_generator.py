"""
Tests for EdgeGenerator module.

This module tests the functionality of the EdgeGenerator class.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from src.graphrag_engine.knowledge_graph.edge_generator import EdgeGenerator


class TestEdgeGenerator:
    """Test suite for EdgeGenerator class."""

    def test_init(self):
        """Test initialization of EdgeGenerator."""
        # Test with default values
        edge_gen = EdgeGenerator()
        assert edge_gen.confidence_threshold == 0.5
        assert edge_gen.edges == {}
        
        # Test with custom values
        edge_gen = EdgeGenerator(confidence_threshold=0.75)
        assert edge_gen.confidence_threshold == 0.75

    def test_create_edge_from_relationship(self, sample_relationships, node_registry):
        """Test creating an edge from a relationship."""
        edge_gen = EdgeGenerator()
        
        # Register entities in the node registry
        node_registry.register_entity({
            "text": "John Smith",
            "type": "PERSON"
        }, "doc1")
        
        node_registry.register_entity({
            "text": "Acme Corporation",
            "type": "ORGANIZATION"
        }, "doc1")
        
        # Create edge from relationship
        relationship = sample_relationships[0]  # John Smith WORKS_FOR Acme Corporation
        document_id = "doc1"
        
        edge_id = edge_gen.create_edge_from_relationship(relationship, node_registry, document_id)
        
        # Verify that edge was created correctly
        assert edge_id is not None
        assert edge_id in edge_gen.edges
        edge = edge_gen.edges[edge_id]
        
        # Check edge properties
        assert edge["type"] == relationship["relationship"]
        assert edge["confidence"] == relationship["confidence"]
        assert document_id in edge["documents"]
        
        # Check that source and target nodes are correct
        source_node = node_registry.get_node_by_entity(relationship["source"])
        target_node = node_registry.get_node_by_entity(relationship["target"])
        assert edge["source_id"] == source_node["id"]
        assert edge["target_id"] == target_node["id"]

    def test_create_edge_with_nonexistent_nodes(self, sample_relationships, node_registry):
        """Test creating an edge with non-existent nodes."""
        edge_gen = EdgeGenerator()
        
        # Create edge from relationship with non-existent nodes
        relationship = sample_relationships[0]  # John Smith WORKS_FOR Acme Corporation
        document_id = "doc1"
        
        # Node registry is empty, so nodes don't exist
        edge_id = edge_gen.create_edge_from_relationship(relationship, node_registry, document_id)
        
        # Edge should not be created
        assert edge_id is None
        assert len(edge_gen.edges) == 0

    def test_create_edges_from_relationships(self, sample_relationships, node_registry):
        """Test creating multiple edges from relationships."""
        edge_gen = EdgeGenerator()
        
        # Register entities in the node registry
        for rel in sample_relationships:
            node_registry.register_entity({
                "text": rel["source"],
                "type": rel["source_type"]
            }, "doc1")
            
            node_registry.register_entity({
                "text": rel["target"],
                "type": rel["target_type"]
            }, "doc1")
        
        # Create edges from relationships
        document_id = "doc1"
        edge_ids = edge_gen.create_edges_from_relationships(sample_relationships, node_registry, document_id)
        
        # Verify results
        assert len(edge_ids) == len(sample_relationships)
        assert all(edge_id is not None for edge_id in edge_ids)
        assert len(edge_gen.edges) == len(sample_relationships)

    def test_filter_by_confidence(self, sample_relationships, node_registry):
        """Test filtering relationships by confidence."""
        # Create edge generator with high confidence threshold
        edge_gen = EdgeGenerator(confidence_threshold=0.9)
        
        # Register entities in the node registry
        for rel in sample_relationships:
            node_registry.register_entity({
                "text": rel["source"],
                "type": rel["source_type"]
            }, "doc1")
            
            node_registry.register_entity({
                "text": rel["target"],
                "type": rel["target_type"]
            }, "doc1")
        
        # Create edges from relationships
        document_id = "doc1"
        edge_ids = edge_gen.create_edges_from_relationships(sample_relationships, node_registry, document_id)
        
        # Only relationships with confidence >= 0.9 should be created
        high_confidence_rels = [rel for rel in sample_relationships if rel["confidence"] >= 0.9]
        assert len(edge_ids) == len(high_confidence_rels)

    def test_get_edge(self):
        """Test retrieving an edge by ID."""
        edge_gen = EdgeGenerator()
        
        # Create a sample edge
        edge = {
            "source_id": "node1",
            "target_id": "node2",
            "type": "WORKS_FOR",
            "confidence": 0.9,
            "bidirectional": False,
            "documents": ["doc1"],
            "metadata": {}
        }
        edge_id = "edge1"
        edge_gen.edges[edge_id] = edge
        
        # Get edge by ID
        retrieved_edge = edge_gen.get_edge(edge_id)
        
        # Verify that the correct edge was retrieved
        assert retrieved_edge is not None
        assert retrieved_edge == edge
        
        # Test with non-existent ID
        retrieved_edge = edge_gen.get_edge("non-existent-id")
        assert retrieved_edge is None

    def test_get_edges_by_type(self):
        """Test retrieving edges by type."""
        edge_gen = EdgeGenerator()
        
        # Create sample edges of different types
        edge_gen.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            "edge2": {
                "source_id": "node3",
                "target_id": "node4",
                "type": "WORKS_FOR",
                "confidence": 0.8,
                "bidirectional": False,
                "documents": ["doc2"],
                "metadata": {}
            },
            "edge3": {
                "source_id": "node5",
                "target_id": "node6",
                "type": "LOCATED_IN",
                "confidence": 0.7,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            }
        }
        
        # Get edges by type
        works_for_edges = edge_gen.get_edges_by_type("WORKS_FOR")
        located_in_edges = edge_gen.get_edges_by_type("LOCATED_IN")
        non_existent_edges = edge_gen.get_edges_by_type("NON_EXISTENT")
        
        # Verify results
        assert len(works_for_edges) == 2
        assert len(located_in_edges) == 1
        assert len(non_existent_edges) == 0
        
        # Check that the correct edges were retrieved
        assert all(edge["type"] == "WORKS_FOR" for edge in works_for_edges)
        assert all(edge["type"] == "LOCATED_IN" for edge in located_in_edges)

    def test_get_edges_by_node(self):
        """Test retrieving edges connected to a node."""
        edge_gen = EdgeGenerator()
        
        # Create sample edges connected to different nodes
        edge_gen.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            "edge2": {
                "source_id": "node3",
                "target_id": "node1",
                "type": "KNOWS",
                "confidence": 0.8,
                "bidirectional": True,
                "documents": ["doc2"],
                "metadata": {}
            },
            "edge3": {
                "source_id": "node4",
                "target_id": "node5",
                "type": "LOCATED_IN",
                "confidence": 0.7,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            }
        }
        
        # Get edges by node
        node1_edges = edge_gen.get_edges_by_node("node1")
        node4_edges = edge_gen.get_edges_by_node("node4")
        non_existent_edges = edge_gen.get_edges_by_node("non-existent-node")
        
        # Verify results
        assert len(node1_edges) == 2  # node1 is connected to edge1 and edge2
        assert len(node4_edges) == 1  # node4 is connected to edge3
        assert len(non_existent_edges) == 0
        
        # Check that the correct edges were retrieved
        assert any(edge["source_id"] == "node1" for edge in node1_edges)
        assert any(edge["target_id"] == "node1" for edge in node1_edges)
        assert all(edge["source_id"] == "node4" for edge in node4_edges)

    def test_merge_parallel_edges(self):
        """Test merging parallel edges between the same nodes."""
        edge_gen = EdgeGenerator()
        
        # Create parallel edges between the same nodes
        edge_gen.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.8,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {"start_date": "2020-01-01"}
            },
            "edge2": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc2"],
                "metadata": {"position": "Engineer"}
            }
        }
        
        # Merge parallel edges
        merged_id = edge_gen.merge_parallel_edges("edge1", "edge2")
        
        # Verify that edges were merged correctly
        assert merged_id == "edge1"
        assert "edge2" not in edge_gen.edges
        
        # Check that the merged edge contains information from both edges
        merged_edge = edge_gen.edges[merged_id]
        assert merged_edge["confidence"] == 0.9  # Higher confidence should be kept
        assert "doc1" in merged_edge["documents"]
        assert "doc2" in merged_edge["documents"]
        assert "start_date" in merged_edge["metadata"]
        assert "position" in merged_edge["metadata"]

    def test_export_import_json(self):
        """Test exporting and importing edges to/from JSON."""
        edge_gen = EdgeGenerator()
        
        # Add sample edges
        edge_gen.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            "edge2": {
                "source_id": "node3",
                "target_id": "node4",
                "type": "LOCATED_IN",
                "confidence": 0.8,
                "bidirectional": False,
                "documents": ["doc2"],
                "metadata": {}
            }
        }
        
        # Export to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        edge_gen.export_to_json(tmp_path)
        
        # Create a new edge generator and import from the file
        new_edge_gen = EdgeGenerator()
        new_edge_gen.import_from_json(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Verify that the import was successful
        assert len(new_edge_gen.edges) == len(edge_gen.edges)
        
        # Check that all edges were imported correctly
        for edge_id, edge in edge_gen.edges.items():
            assert edge_id in new_edge_gen.edges
            imported_edge = new_edge_gen.edges[edge_id]
            assert imported_edge["source_id"] == edge["source_id"]
            assert imported_edge["target_id"] == edge["target_id"]
            assert imported_edge["type"] == edge["type"]
            assert imported_edge["confidence"] == edge["confidence"]
            assert imported_edge["bidirectional"] == edge["bidirectional"]
            assert imported_edge["documents"] == edge["documents"]

    def test_get_all_edges(self):
        """Test retrieving all edges."""
        edge_gen = EdgeGenerator()
        
        # Add sample edges
        edge_gen.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            "edge2": {
                "source_id": "node3",
                "target_id": "node4",
                "type": "LOCATED_IN",
                "confidence": 0.8,
                "bidirectional": False,
                "documents": ["doc2"],
                "metadata": {}
            }
        }
        
        # Get all edges
        edges = edge_gen.get_all_edges()
        
        # Verify results
        assert len(edges) == 2
        assert isinstance(edges, list)
        
        # Check that edges contain the expected information
        for edge in edges:
            assert "id" in edge
            assert "source_id" in edge
            assert "target_id" in edge
            assert "type" in edge
            assert "confidence" in edge
            
            # Check that edge types match expected values
            assert edge["type"] in ["WORKS_FOR", "LOCATED_IN"]