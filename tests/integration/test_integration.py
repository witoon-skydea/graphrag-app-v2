"""
Integration tests for GraphRAG system.

This module tests the integration between different components of the GraphRAG system.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graphrag_engine.knowledge_graph.entity_extractor import EntityExtractor
from src.graphrag_engine.knowledge_graph.relationship_identifier import RelationshipIdentifier
from src.graphrag_engine.knowledge_graph.node_registry import NodeRegistry
from src.graphrag_engine.knowledge_graph.edge_generator import EdgeGenerator
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.graphrag_engine.vector_db.weaviate_client import VectorDBClient
from src.graphrag_engine.graph_rag_engine import GraphRAGEngine


class TestIntegration:
    """Test suite for integration testing of GraphRAG components."""

    @patch.object(VectorDBClient, 'check_connection')
    @patch.object(VectorDBClient, 'create_schema')
    def test_graph_rag_engine_initialization(self, mock_create_schema, mock_check_connection):
        """Test initialization of GraphRAGEngine with all components."""
        # Mock VectorDBClient methods
        mock_check_connection.return_value = True
        mock_create_schema.return_value = True
        
        # Create GraphRAGEngine
        engine = GraphRAGEngine(
            extraction_method="ollama",
            identification_method="ollama",
            model_name="llama2",
            weaviate_url="http://localhost:8080",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize the engine
        result = engine.initialize()
        
        # Verify that initialization was successful
        assert result is True
        mock_check_connection.assert_called_once()
        mock_create_schema.assert_called_once()
        
        # Verify that components were initialized correctly
        assert engine.kg_builder is not None
        assert engine.vector_db is not None
        assert engine.kg_builder.entity_extractor is not None
        assert engine.kg_builder.relationship_identifier is not None
        assert engine.kg_builder.node_registry is not None
        assert engine.kg_builder.edge_generator is not None
        
        # Verify that configuration parameters were passed correctly
        assert engine.kg_builder.extraction_method == "ollama"
        assert engine.kg_builder.identification_method == "ollama"
        assert engine.kg_builder.model_name == "llama2"

    @patch.object(EntityExtractor, 'extract_entities')
    @patch.object(RelationshipIdentifier, 'identify_relationships')
    @patch.object(VectorDBClient, 'add_document_chunks_batch')
    def test_document_processing_workflow(self, mock_add_batch, mock_identify, mock_extract, sample_entities, sample_relationships):
        """Test the complete document processing workflow."""
        # Mock component methods
        mock_extract.return_value = sample_entities
        mock_identify.return_value = sample_relationships
        mock_add_batch.return_value = 4  # Number of chunks
        
        # Create GraphRAGEngine with mocked database client
        mock_vector_db = MagicMock(spec=VectorDBClient)
        mock_vector_db.add_document_chunks_batch.return_value = 4
        
        engine = GraphRAGEngine(
            vector_db_client=mock_vector_db,
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Process a document
        document_text = "This is a test document with multiple paragraphs.\n\nJohn Smith works at Acme Corporation. Acme Corporation is headquartered in New York City."
        document_id = "test-doc-001"
        title = "Test Document"
        document_type = "text"
        source = "test.txt"
        
        result = engine.process_document(
            document_text=document_text,
            document_id=document_id,
            title=title,
            document_type=document_type,
            source=source
        )
        
        # Verify that all components were called correctly
        mock_extract.assert_called_once()
        mock_identify.assert_called_once()
        mock_vector_db.add_document_chunks_batch.assert_called_once()
        
        # Verify the result contains expected keys
        assert "document_id" in result
        assert "chunks_created" in result
        assert "entities_extracted" in result
        assert "nodes_created" in result
        assert "relationships_identified" in result
        assert "edges_created" in result
        assert "chunks_indexed" in result
        
        # Verify that the result has expected values
        assert result["document_id"] == document_id
        assert result["entities_extracted"] == len(sample_entities)
        assert result["relationships_identified"] == len(sample_relationships)

    @patch('requests.post')
    def test_entity_extractor_to_relationship_identifier(self, mock_post, sample_text):
        """Test the workflow from entity extraction to relationship identification."""
        # Mock the API response for entity extraction
        mock_extract_response = MagicMock()
        mock_extract_response.status_code = 200
        mock_extract_response.json.return_value = {
            "response": """[
                {"text": "John Smith", "type": "PERSON", "start_pos": 0, "end_pos": 10},
                {"text": "Acme Corporation", "type": "ORGANIZATION", "start_pos": 25, "end_pos": 41},
                {"text": "New York City", "type": "LOCATION", "start_pos": 80, "end_pos": 93}
            ]"""
        }
        
        # Mock the API response for relationship identification
        mock_relation_response = MagicMock()
        mock_relation_response.status_code = 200
        mock_relation_response.json.return_value = {
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
                },
                {
                    "pair_id": 2,
                    "source": "Acme Corporation",
                    "source_type": "ORGANIZATION",
                    "target": "New York City",
                    "target_type": "LOCATION",
                    "relationship": "HEADQUARTERED_IN",
                    "confidence": 0.8,
                    "bidirectional": false
                }
            ]"""
        }
        
        # Set up the mock post to return different responses for different calls
        def side_effect(*args, **kwargs):
            if "generate" in args[0] and "Extract entities" in kwargs.get("json", {}).get("prompt", ""):
                return mock_extract_response
            elif "generate" in args[0] and "Identify relationships" in kwargs.get("json", {}).get("prompt", ""):
                return mock_relation_response
            return MagicMock()
        
        mock_post.side_effect = side_effect
        
        # Create the components
        extractor = EntityExtractor(extraction_method="ollama")
        identifier = RelationshipIdentifier(identification_method="ollama")
        
        # Extract entities
        entities = extractor.extract_entities(sample_text)
        
        # Verify that entities were extracted correctly
        assert len(entities) == 3
        assert entities[0]["text"] == "John Smith"
        assert entities[1]["text"] == "Acme Corporation"
        assert entities[2]["text"] == "New York City"
        
        # Identify relationships
        relationships = identifier.identify_relationships(entities, sample_text)
        
        # Verify that relationships were identified correctly
        assert len(relationships) == 2
        assert relationships[0]["source"] == "John Smith"
        assert relationships[0]["target"] == "Acme Corporation"
        assert relationships[0]["relationship"] == "WORKS_FOR"
        assert relationships[1]["source"] == "Acme Corporation"
        assert relationships[1]["target"] == "New York City"
        assert relationships[1]["relationship"] == "HEADQUARTERED_IN"

    def test_node_registry_to_edge_generator(self, sample_entities, sample_relationships):
        """Test the workflow from node registry to edge generator."""
        # Create the components
        registry = NodeRegistry()
        edge_gen = EdgeGenerator()
        
        # Register entities
        doc_id = "test-doc"
        node_ids = []
        for entity in sample_entities:
            node_id = registry.register_entity(entity, doc_id)
            node_ids.append(node_id)
        
        # Verify that entities were registered correctly
        assert len(registry.nodes) == len(sample_entities)
        
        # Create edges from relationships
        edge_ids = edge_gen.create_edges_from_relationships(sample_relationships, registry, doc_id)
        
        # Verify that edges were created correctly
        assert len(edge_ids) >= len(sample_relationships)
        
        # Verify that we can retrieve edges by node
        for node_id in node_ids:
            node_edges = edge_gen.get_edges_by_node(node_id)
            if node_edges:
                # If this node has edges, check they have the correct structure
                for edge in node_edges:
                    assert "source_id" in edge
                    assert "target_id" in edge
                    assert "type" in edge
                    assert "confidence" in edge
                    assert doc_id in edge["documents"]

    @patch.object(VectorDBClient, 'check_connection')
    @patch.object(VectorDBClient, 'search_by_text')
    @patch.object(KnowledgeGraphBuilder, 'get_entity_relationships')
    def test_search_workflow(self, mock_get_relationships, mock_search_by_text, mock_check_connection):
        """Test the search workflow combining vector search and knowledge graph."""
        # Mock database connection check
        mock_check_connection.return_value = True
        
        # Mock vector search results
        mock_search_by_text.return_value = [
            {
                "content": "John Smith works at Acme Corporation.",
                "document_id": "doc1",
                "chunk_index": 0,
                "title": "Employee Records",
                "score": 0.9,
                "id": "uuid1"
            }
        ]
        
        # Mock knowledge graph results
        mock_get_relationships.return_value = {
            "outgoing": [
                {
                    "source_text": "John Smith",
                    "source_type": "PERSON",
                    "target_text": "Acme Corporation",
                    "target_type": "ORGANIZATION",
                    "type": "WORKS_FOR",
                    "confidence": 0.9
                }
            ],
            "incoming": []
        }
        
        # Create GraphRAGEngine
        engine = GraphRAGEngine()
        
        # Perform search
        query = "John Smith Acme"
        search_results = engine.search(query)
        
        # Verify that both vector search and knowledge graph were used
        mock_search_by_text.assert_called_once()
        assert "query" in search_results
        assert "vector_results" in search_results
        assert "combined_results" in search_results
        
        # Verify that the combined results contain both vector and knowledge graph results
        assert len(search_results["combined_results"]) >= len(search_results["vector_results"])
        
        # Check that we have the correct result types
        result_types = [result["result_type"] for result in search_results["combined_results"]]
        assert "vector" in result_types