"""
Tests for KnowledgeGraphBuilder module.

This module tests the functionality of the KnowledgeGraphBuilder class.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder


class TestKnowledgeGraphBuilder:
    """Test suite for KnowledgeGraphBuilder class."""

    def test_init(self):
        """Test initialization of KnowledgeGraphBuilder."""
        # Test with default values
        builder = KnowledgeGraphBuilder()
        assert builder.extraction_method == "ollama"
        assert builder.identification_method == "ollama"
        assert builder.model_name == "llama2"
        assert builder.api_key is None
        assert builder.similarity_threshold == 0.85
        assert builder.confidence_threshold == 0.5
        assert builder.case_sensitive is False
        assert builder.persist_path is None
        
        # Test with custom values
        builder = KnowledgeGraphBuilder(
            extraction_method="openai",
            identification_method="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            entity_types=["PERSON", "ORGANIZATION"],
            relationship_types=["WORKS_FOR"],
            similarity_threshold=0.75,
            confidence_threshold=0.7,
            case_sensitive=True,
            persist_path="/tmp/test"
        )
        assert builder.extraction_method == "openai"
        assert builder.identification_method == "openai"
        assert builder.model_name == "gpt-3.5-turbo"
        assert builder.api_key == "test-api-key"
        assert builder.similarity_threshold == 0.75
        assert builder.confidence_threshold == 0.7
        assert builder.case_sensitive is True
        assert builder.persist_path == "/tmp/test"
        
        # Verify that components were initialized
        assert builder.entity_extractor is not None
        assert builder.relationship_identifier is not None
        assert builder.node_registry is not None
        assert builder.edge_generator is not None

    def test_process_document(self, sample_text, sample_entities, sample_relationships, mock_entity_extractor, mock_relationship_identifier):
        """Test processing a document."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.entity_extractor = mock_entity_extractor
        builder.relationship_identifier = mock_relationship_identifier
        
        # Process the document
        document_id = "doc123"
        result = builder.process_document(sample_text, document_id)
        
        # Verify that the entity extractor was called
        mock_entity_extractor.extract_entities.assert_called_once_with(sample_text)
        
        # Verify that the relationship identifier was called
        mock_relationship_identifier.identify_relationships.assert_called_once()
        
        # Check that the result contains expected information
        assert result["document_id"] == document_id
        assert result["entities_extracted"] == len(sample_entities)
        assert result["relationships_identified"] == len(sample_relationships)
        assert "nodes_created" in result
        assert "edges_created" in result

    def test_process_document_no_entities(self, mock_entity_extractor):
        """Test processing a document with no entities."""
        # Set up mock to return empty list
        mock_entity_extractor.extract_entities.return_value = []
        
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.entity_extractor = mock_entity_extractor
        
        # Process the document
        document_id = "doc123"
        result = builder.process_document("Empty document", document_id)
        
        # Verify that the relationship identifier was not called
        assert result["entities_extracted"] == 0
        assert result["relationships_identified"] == 0
        assert result["nodes_created"] == 0
        assert result["edges_created"] == 0

    def test_process_documents(self, sample_text, mock_entity_extractor, mock_relationship_identifier):
        """Test processing multiple documents."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.entity_extractor = mock_entity_extractor
        builder.relationship_identifier = mock_relationship_identifier
        
        # Create sample documents
        documents = {
            "doc1": "This is document 1.",
            "doc2": "This is document 2."
        }
        
        # Process the documents
        result = builder.process_documents(documents)
        
        # Check that the result contains expected information
        assert result["documents_processed"] == len(documents)
        assert "entities_extracted" in result
        assert "nodes_created" in result
        assert "relationships_identified" in result
        assert "edges_created" in result

    def test_add_entity(self, node_registry):
        """Test adding a single entity."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        
        # Add an entity
        entity_text = "John Smith"
        entity_type = "PERSON"
        document_id = "doc123"
        
        node_id = builder.add_entity(entity_text, entity_type, {}, document_id)
        
        # Verify that the entity was added
        assert node_id is not None
        node = node_registry.get_node(node_id)
        assert node["text"] == entity_text
        assert node["type"] == entity_type
        assert document_id in node["documents"]

    def test_add_relationship(self, node_registry, edge_generator):
        """Test adding a relationship between entities."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Add nodes for the relationship
        node_registry.register_entity({"text": "John Smith", "type": "PERSON"}, "doc1")
        node_registry.register_entity({"text": "Acme Corp", "type": "ORGANIZATION"}, "doc1")
        
        # Add a relationship
        edge_id = builder.add_relationship(
            "John Smith", "Acme Corp", "WORKS_FOR", 0.9, False, {}, "doc1"
        )
        
        # Verify that the relationship was added
        assert edge_id is not None
        edge = edge_generator.get_edge(edge_id)
        assert edge["type"] == "WORKS_FOR"
        assert edge["confidence"] == 0.9
        assert "doc1" in edge["documents"]

    def test_query_entities(self, node_registry, sample_entities):
        """Test querying entities."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        
        # Register some entities
        for entity in sample_entities:
            node_registry.register_entity(entity, "doc1")
        
        # Query entities
        entities = builder.query_entities()
        assert len(entities) == len(sample_entities)
        
        # Query by entity type
        person_entities = builder.query_entities(entity_type="PERSON")
        assert len(person_entities) > 0
        assert all(entity["type"] == "PERSON" for entity in person_entities)
        
        # Query by document ID
        doc_entities = builder.query_entities(document_id="doc1")
        assert len(doc_entities) == len(sample_entities)
        
        # Query by entity text
        specific_entity = builder.query_entities(entity_text="John Smith")
        assert len(specific_entity) > 0
        assert specific_entity[0]["text"] == "John Smith"

    def test_query_relationships(self, node_registry, edge_generator, sample_relationships):
        """Test querying relationships."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Register entities and relationships
        for rel in sample_relationships:
            source_id = node_registry.register_entity({
                "text": rel["source"],
                "type": rel["source_type"]
            }, "doc1")
            
            target_id = node_registry.register_entity({
                "text": rel["target"],
                "type": rel["target_type"]
            }, "doc1")
            
            edge_id = f"edge_{source_id}_{target_id}"
            edge_generator.edges[edge_id] = {
                "source_id": source_id,
                "target_id": target_id,
                "type": rel["relationship"],
                "confidence": rel["confidence"],
                "bidirectional": rel.get("bidirectional", False),
                "documents": ["doc1"],
                "metadata": rel.get("metadata", {})
            }
        
        # Query all relationships
        relationships = builder.query_relationships()
        assert len(relationships) == len(sample_relationships)
        
        # Query by relationship type
        works_for_rels = builder.query_relationships(relationship_type="WORKS_FOR")
        assert len(works_for_rels) > 0
        assert all(rel["relationship"] == "WORKS_FOR" for rel in works_for_rels)
        
        # Query by document ID
        doc_rels = builder.query_relationships(document_id="doc1")
        assert len(doc_rels) == len(sample_relationships)
        
        # Query by source/target
        specific_rels = builder.query_relationships(source_text="John Smith")
        assert len(specific_rels) > 0
        assert specific_rels[0]["source_text"] == "John Smith"

    def test_get_entity_relationships(self, node_registry, edge_generator):
        """Test getting relationships for a specific entity."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Create nodes
        node1 = node_registry.register_entity({"text": "John", "type": "PERSON"}, "doc1")
        node2 = node_registry.register_entity({"text": "Acme", "type": "ORGANIZATION"}, "doc1")
        node3 = node_registry.register_entity({"text": "Sarah", "type": "PERSON"}, "doc1")
        
        # Create edges
        edge_generator.edges = {
            "edge1": {
                "source_id": node1,
                "target_id": node2,
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            "edge2": {
                "source_id": node3,
                "target_id": node1,
                "type": "KNOWS",
                "confidence": 0.8,
                "bidirectional": True,
                "documents": ["doc1"],
                "metadata": {}
            }
        }
        
        # Get relationships for John
        relationships = builder.get_entity_relationships("John")
        
        # Verify results
        assert "outgoing" in relationships
        assert "incoming" in relationships
        assert len(relationships["outgoing"]) == 1  # John WORKS_FOR Acme
        assert len(relationships["incoming"]) == 1  # Sarah KNOWS John
        
        # Check that relationship information is correct
        outgoing = relationships["outgoing"][0]
        assert outgoing["type"] == "WORKS_FOR"
        assert outgoing["source_text"] == "John"
        assert outgoing["target_text"] == "Acme"
        
        incoming = relationships["incoming"][0]
        assert incoming["type"] == "KNOWS"
        assert incoming["source_text"] == "Sarah"
        assert incoming["target_text"] == "John"

    def test_persist_and_load_graph(self, temp_persist_path, sample_entities, sample_relationships):
        """Test persisting and loading the knowledge graph."""
        # Create a builder with a persistence path
        builder = KnowledgeGraphBuilder(persist_path=temp_persist_path)
        
        # Add some data
        document_id = "doc1"
        for entity in sample_entities:
            builder.node_registry.register_entity(entity, document_id)
        
        for rel in sample_relationships:
            source_id = builder.node_registry.get_node_by_entity(rel["source"])["id"]
            target_id = builder.node_registry.get_node_by_entity(rel["target"])["id"]
            
            edge_id = f"edge_{source_id}_{target_id}"
            builder.edge_generator.edges[edge_id] = {
                "source_id": source_id,
                "target_id": target_id,
                "type": rel["relationship"],
                "confidence": rel["confidence"],
                "bidirectional": rel.get("bidirectional", False),
                "documents": [document_id],
                "metadata": rel.get("metadata", {})
            }
        
        # Persist the graph
        builder.persist_graph()
        
        # Create a new builder and load the graph
        new_builder = KnowledgeGraphBuilder(persist_path=temp_persist_path)
        new_builder.load_graph()
        
        # Verify that the graph was loaded correctly
        assert len(new_builder.node_registry.nodes) == len(builder.node_registry.nodes)
        assert len(new_builder.edge_generator.edges) == len(builder.edge_generator.edges)
        
        # Check that we can retrieve entities and relationships
        entities = new_builder.query_entities()
        assert len(entities) == len(sample_entities)
        
        relationships = new_builder.query_relationships()
        assert len(relationships) == len(sample_relationships)

    def test_graph_statistics(self, node_registry, edge_generator):
        """Test getting statistics about the knowledge graph."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Add some data
        node_registry.register_entity({"text": "John", "type": "PERSON"}, "doc1")
        node_registry.register_entity({"text": "Acme", "type": "ORGANIZATION"}, "doc1")
        node_registry.register_entity({"text": "New York", "type": "LOCATION"}, "doc2")
        
        edge_id1 = "edge1"
        edge_id2 = "edge2"
        
        edge_generator.edges = {
            edge_id1: {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            },
            edge_id2: {
                "source_id": "node2",
                "target_id": "node3",
                "type": "LOCATED_IN",
                "confidence": 0.8,
                "bidirectional": False,
                "documents": ["doc1", "doc2"],
                "metadata": {}
            }
        }
        
        # Get statistics
        stats = builder.graph_statistics()
        
        # Verify results
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "total_documents" in stats
        assert "node_types" in stats
        assert "edge_types" in stats
        assert "average_degree" in stats
        
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2
        assert stats["total_documents"] >= 1
        assert "PERSON" in stats["node_types"]
        assert "ORGANIZATION" in stats["node_types"]
        assert "WORKS_FOR" in stats["edge_types"]
        assert "LOCATED_IN" in stats["edge_types"]

    def test_export_to_network_format(self, node_registry, edge_generator, temp_persist_path):
        """Test exporting the graph to a network format."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Add some data
        node1 = node_registry.register_entity({"text": "John", "type": "PERSON"}, "doc1")
        node2 = node_registry.register_entity({"text": "Acme", "type": "ORGANIZATION"}, "doc1")
        
        edge_id = "edge1"
        edge_generator.edges[edge_id] = {
            "source_id": node1,
            "target_id": node2,
            "type": "WORKS_FOR",
            "confidence": 0.9,
            "bidirectional": False,
            "documents": ["doc1"],
            "metadata": {}
        }
        
        # Export to a temporary file
        output_file = os.path.join(temp_persist_path, "network.json")
        builder.export_to_network_format(output_file)
        
        # Verify that the file was created
        assert os.path.exists(output_file)
        
        # Load and check the exported data
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        
        # Check node and edge properties
        node = data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "group" in node
        
        edge = data["edges"][0]
        assert "id" in edge
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge
        assert "type" in edge
        assert "confidence" in edge

    def test_clear_graph(self, node_registry, edge_generator):
        """Test clearing the knowledge graph."""
        # Create a builder with mocked components
        builder = KnowledgeGraphBuilder()
        builder.node_registry = node_registry
        builder.edge_generator = edge_generator
        
        # Add some data
        node_registry.register_entity({"text": "John", "type": "PERSON"}, "doc1")
        node_registry.register_entity({"text": "Acme", "type": "ORGANIZATION"}, "doc1")
        
        edge_generator.edges = {
            "edge1": {
                "source_id": "node1",
                "target_id": "node2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "bidirectional": False,
                "documents": ["doc1"],
                "metadata": {}
            }
        }
        
        # Clear the graph
        builder.clear_graph()
        
        # Verify that the graph was cleared
        assert len(node_registry.nodes) == 0
        assert len(edge_generator.edges) == 0