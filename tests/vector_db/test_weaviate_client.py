"""
Tests for VectorDBClient module.

This module tests the functionality of the VectorDBClient class.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import uuid

from src.graphrag_engine.vector_db.weaviate_client import VectorDBClient, get_valid_uuid


class TestVectorDBClient:
    """Test suite for VectorDBClient class."""

    def test_init(self):
        """Test initialization of VectorDBClient."""
        # Test with default values
        client = VectorDBClient()
        assert client.weaviate_url == "http://localhost:8080"
        assert client.openai_api_key is None
        assert client.batch_size == 100
        assert client.schema_class_name == "Document"
        assert client.default_vectorizer == "text2vec-transformers"
        
        # Test with custom values
        client = VectorDBClient(
            weaviate_url="http://custom:8080",
            openai_api_key="test-api-key",
            batch_size=50,
            timeout_config=(5, 30),
            schema_class_name="CustomDoc",
            default_vectorizer="text2vec-openai"
        )
        assert client.weaviate_url == "http://custom:8080"
        assert client.openai_api_key == "test-api-key"
        assert client.batch_size == 50
        assert client.timeout_config == (5, 30)
        assert client.schema_class_name == "CustomDoc"
        assert client.default_vectorizer == "text2vec-openai"

    @patch('weaviate.Client')
    def test_init_client(self, mock_weaviate_client):
        """Test initializing the Weaviate client."""
        # Test without OpenAI API key
        client = VectorDBClient()
        client._init_client()
        
        mock_weaviate_client.assert_called_once()
        args, kwargs = mock_weaviate_client.call_args
        assert kwargs["url"] == "http://localhost:8080"
        assert "additional_headers" in kwargs
        assert "timeout_config" in kwargs
        
        # Test with OpenAI API key and vectorizer
        mock_weaviate_client.reset_mock()
        client = VectorDBClient(
            openai_api_key="test-api-key",
            default_vectorizer="text2vec-openai"
        )
        client._init_client()
        
        mock_weaviate_client.assert_called_once()
        args, kwargs = mock_weaviate_client.call_args
        assert kwargs["url"] == "http://localhost:8080"
        assert "additional_headers" in kwargs
        assert "X-OpenAI-Api-Key" in kwargs["additional_headers"]
        assert kwargs["additional_headers"]["X-OpenAI-Api-Key"] == "test-api-key"

    @patch.object(VectorDBClient, '_init_client')
    def test_check_connection(self, mock_init_client):
        """Test checking connection to Weaviate."""
        # Set up mock client
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test successful connection
        assert client.check_connection() is True
        mock_client.is_ready.assert_called_once()
        
        # Test failed connection
        mock_client.is_ready.return_value = False
        assert client.check_connection() is False
        
        # Test exception handling
        mock_client.is_ready.side_effect = Exception("Connection error")
        assert client.check_connection() is False

    @patch.object(VectorDBClient, '_init_client')
    def test_create_schema(self, mock_init_client):
        """Test creating schema in Weaviate."""
        # Set up mock client
        mock_client = MagicMock()
        mock_schema = MagicMock()
        mock_client.schema = mock_schema
        
        # Mock schema.contains to control test flow
        mock_schema.contains.return_value = False
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test schema creation
        assert client.create_schema() is True
        mock_schema.contains.assert_called_once()
        mock_schema.create_class.assert_called()
        
        # Test when schema already exists
        mock_schema.reset_mock()
        mock_schema.contains.return_value = True
        
        assert client.create_schema() is True
        mock_schema.contains.assert_called_once()
        mock_schema.create_class.assert_not_called()
        
        # Test recreating schema
        mock_schema.reset_mock()
        mock_schema.contains.return_value = True
        
        assert client.create_schema(force_recreate=True) is True
        mock_schema.contains.assert_called_once()
        mock_schema.delete_class.assert_called_once()
        mock_schema.create_class.assert_called()
        
        # Test exception handling
        mock_schema.reset_mock()
        mock_schema.contains.side_effect = Exception("Schema error")
        
        assert client.create_schema() is False

    @patch.object(VectorDBClient, '_init_client')
    @patch.object(VectorDBClient, 'check_connection')
    def test_add_document_chunk(self, mock_check_connection, mock_init_client):
        """Test adding a document chunk to Weaviate."""
        # Set up mock client
        mock_client = MagicMock()
        mock_data_object = MagicMock()
        mock_client.data_object = mock_data_object
        mock_data_object.create.return_value = "test-uuid"
        
        # Mock check_connection
        mock_check_connection.return_value = True
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test adding a document chunk
        chunk_text = "This is a test chunk"
        document_id = "doc123"
        chunk_index = 0
        title = "Test Document"
        
        result = client.add_document_chunk(
            chunk_text=chunk_text,
            document_id=document_id,
            chunk_index=chunk_index,
            title=title
        )
        
        assert result == "test-uuid"
        mock_data_object.create.assert_called_once()
        
        # Check that correct data was passed
        args, kwargs = mock_data_object.create.call_args
        assert kwargs["class_name"] == client.schema_class_name
        assert kwargs["data_object"]["content"] == chunk_text
        assert kwargs["data_object"]["documentId"] == document_id
        assert kwargs["data_object"]["chunkIndex"] == chunk_index
        assert kwargs["data_object"]["title"] == title
        
        # Test with custom vector
        mock_data_object.reset_mock()
        
        custom_vector = [0.1, 0.2, 0.3]
        result = client.add_document_chunk(
            chunk_text=chunk_text,
            document_id=document_id,
            chunk_index=chunk_index,
            custom_vector=custom_vector
        )
        
        assert result == "test-uuid"
        mock_data_object.create.assert_called_once()
        
        # Check that vector was passed
        args, kwargs = mock_data_object.create.call_args
        assert "vector" in kwargs
        assert kwargs["vector"] == custom_vector
        
        # Test with invalid inputs
        mock_data_object.reset_mock()
        
        # Test with empty chunk text
        result = client.add_document_chunk(
            chunk_text="",
            document_id=document_id,
            chunk_index=chunk_index
        )
        
        assert result is None
        mock_data_object.create.assert_not_called()
        
        # Test with empty document ID
        result = client.add_document_chunk(
            chunk_text=chunk_text,
            document_id="",
            chunk_index=chunk_index
        )
        
        assert result is None
        mock_data_object.create.assert_not_called()
        
        # Test with connection failure
        mock_check_connection.return_value = False
        
        result = client.add_document_chunk(
            chunk_text=chunk_text,
            document_id=document_id,
            chunk_index=chunk_index
        )
        
        assert result is None
        mock_data_object.create.assert_not_called()
        
        # Test with exception
        mock_check_connection.return_value = True
        mock_data_object.create.side_effect = Exception("Creation error")
        
        result = client.add_document_chunk(
            chunk_text=chunk_text,
            document_id=document_id,
            chunk_index=chunk_index
        )
        
        assert result is None

    @patch.object(VectorDBClient, '_init_client')
    @patch.object(VectorDBClient, 'check_connection')
    def test_add_document_chunks_batch(self, mock_check_connection, mock_init_client):
        """Test adding document chunks in batch to Weaviate."""
        # Set up mock client
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch = mock_batch
        
        # Mock context manager
        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=None)
        
        # Mock check_connection
        mock_check_connection.return_value = True
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test adding document chunks in batch
        chunks = [
            {
                "chunk_text": "Chunk 1",
                "document_id": "doc123",
                "chunk_index": 0,
                "title": "Test Document"
            },
            {
                "chunk_text": "Chunk 2",
                "document_id": "doc123",
                "chunk_index": 1,
                "title": "Test Document"
            }
        ]
        
        result = client.add_document_chunks_batch(chunks)
        
        assert result == len(chunks)
        mock_batch.configure.assert_called_once()
        assert mock_batch.add_data_object.call_count == len(chunks)
        
        # Test with custom vectors
        mock_batch.reset_mock()
        mock_batch.add_data_object.reset_mock()
        
        custom_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = client.add_document_chunks_batch(chunks, custom_vectors)
        
        assert result == len(chunks)
        mock_batch.configure.assert_called_once()
        assert mock_batch.add_data_object.call_count == len(chunks)
        
        # Test exception handling
        mock_batch.configure.side_effect = Exception("Batch error")
        
        result = client.add_document_chunks_batch(chunks)
        
        assert result == 0

    @patch.object(VectorDBClient, '_init_client')
    def test_search_by_vector(self, mock_init_client):
        """Test searching documents by vector similarity."""
        # Set up mock client
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_client.query = mock_query
        
        # Mock query chain
        mock_get = MagicMock()
        mock_with_near_vector = MagicMock()
        mock_with_where = MagicMock()
        mock_with_limit = MagicMock()
        mock_with_offset = MagicMock()
        mock_with_additional = MagicMock()
        mock_do = MagicMock()
        
        mock_query.get.return_value = mock_get
        mock_get.with_near_vector.return_value = mock_with_near_vector
        mock_with_near_vector.with_where.return_value = mock_with_where
        mock_with_near_vector.with_limit.return_value = mock_with_limit
        mock_with_limit.with_offset.return_value = mock_with_offset
        mock_with_offset.with_additional.return_value = mock_with_additional
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "content": "Test content",
                            "documentId": "doc123",
                            "chunkIndex": 0,
                            "title": "Test Document",
                            "documentType": "text",
                            "source": "test.txt",
                            "_additional": {
                                "distance": 0.1,
                                "id": "test-uuid"
                            }
                        }
                    ]
                }
            }
        }
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test search by vector
        vector = [0.1, 0.2, 0.3]
        results = client.search_by_vector(vector)
        
        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["document_id"] == "doc123"
        assert results[0]["distance"] == 0.1
        assert results[0]["id"] == "test-uuid"
        
        # Test with filters
        filters = {"documentType": "text"}
        client.search_by_vector(vector, filters=filters)
        
        # Test with empty result
        mock_with_additional.do.return_value = {"data": {"Get": {}}}
        results = client.search_by_vector(vector)
        assert len(results) == 0
        
        # Test exception handling
        mock_with_additional.do.side_effect = Exception("Search error")
        results = client.search_by_vector(vector)
        assert len(results) == 0

    @patch.object(VectorDBClient, '_init_client')
    def test_search_by_text(self, mock_init_client):
        """Test searching documents by text."""
        # Set up mock client
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_client.query = mock_query
        
        # Mock query chain
        mock_get = MagicMock()
        mock_with_hybrid = MagicMock()
        mock_with_near_text = MagicMock()
        mock_with_where = MagicMock()
        mock_with_limit = MagicMock()
        mock_with_offset = MagicMock()
        mock_with_additional = MagicMock()
        mock_do = MagicMock()
        
        mock_query.get.return_value = mock_get
        mock_get.with_hybrid.return_value = mock_with_hybrid
        mock_get.with_near_text.return_value = mock_with_near_text
        mock_with_hybrid.with_where.return_value = mock_with_where
        mock_with_hybrid.with_limit.return_value = mock_with_limit
        mock_with_near_text.with_limit.return_value = mock_with_limit
        mock_with_limit.with_offset.return_value = mock_with_offset
        mock_with_offset.with_additional.return_value = mock_with_additional
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "content": "Test content",
                            "documentId": "doc123",
                            "chunkIndex": 0,
                            "title": "Test Document",
                            "documentType": "text",
                            "source": "test.txt",
                            "_additional": {
                                "score": 0.9,
                                "id": "test-uuid"
                            }
                        }
                    ]
                }
            }
        }
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test search by text with hybrid search
        text = "test search"
        results = client.search_by_text(text)
        
        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["document_id"] == "doc123"
        assert results[0]["score"] == 0.9
        assert results[0]["id"] == "test-uuid"
        
        # Test with vector search only
        mock_get.reset_mock()
        results = client.search_by_text(text, hybrid_search=False)
        
        mock_get.with_near_text.assert_called_once()
        assert len(results) == 1
        
        # Test with filters
        filters = {"documentType": "text"}
        client.search_by_text(text, filters=filters)
        
        # Test with empty result
        mock_with_additional.do.return_value = {"data": {"Get": {}}}
        results = client.search_by_text(text)
        assert len(results) == 0
        
        # Test exception handling
        mock_with_additional.do.side_effect = Exception("Search error")
        results = client.search_by_text(text)
        assert len(results) == 0

    @patch.object(VectorDBClient, '_init_client')
    def test_get_document_by_id(self, mock_init_client):
        """Test getting a document by ID."""
        # Set up mock client
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_client.query = mock_query
        
        # Mock query chain
        mock_get = MagicMock()
        mock_with_where = MagicMock()
        mock_with_additional = MagicMock()
        mock_do = MagicMock()
        
        mock_query.get.return_value = mock_get
        mock_get.with_where.return_value = mock_with_where
        mock_with_where.with_additional.return_value = mock_with_additional
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "content": "Chunk 1",
                            "documentId": "doc123",
                            "chunkIndex": 0,
                            "title": "Test Document",
                            "documentType": "text",
                            "source": "test.txt",
                            "metadata": "{}",
                            "_additional": {
                                "id": "uuid1"
                            }
                        },
                        {
                            "content": "Chunk 2",
                            "documentId": "doc123",
                            "chunkIndex": 1,
                            "title": "Test Document",
                            "documentType": "text",
                            "source": "test.txt",
                            "metadata": "{}",
                            "_additional": {
                                "id": "uuid2"
                            }
                        }
                    ]
                }
            }
        }
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test get document by ID with include_chunks=True
        document_id = "doc123"
        document = client.get_document_by_id(document_id)
        
        assert document is not None
        assert document["document_id"] == document_id
        assert document["title"] == "Test Document"
        assert document["document_type"] == "text"
        assert document["source"] == "test.txt"
        assert "chunks" in document
        assert len(document["chunks"]) == 2
        
        # Test get document by ID with include_chunks=False
        document = client.get_document_by_id(document_id, include_chunks=False)
        
        assert document is not None
        assert document["document_id"] == document_id
        assert "chunks" not in document
        
        # Test with metadata parsing
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "content": "Chunk 1",
                            "documentId": "doc123",
                            "chunkIndex": 0,
                            "title": "Test Document",
                            "documentType": "text",
                            "source": "test.txt",
                            "metadata": '{"author": "Test Author", "year": 2023}',
                            "_additional": {
                                "id": "uuid1"
                            }
                        }
                    ]
                }
            }
        }
        
        document = client.get_document_by_id(document_id)
        
        assert document is not None
        assert "metadata" in document
        assert document["metadata"]["author"] == "Test Author"
        assert document["metadata"]["year"] == 2023
        
        # Test with empty result
        mock_with_additional.do.return_value = {"data": {"Get": {"Document": []}}}
        document = client.get_document_by_id(document_id)
        assert document is None
        
        # Test with missing properties
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "documentId": "doc123",
                            "_additional": {
                                "id": "uuid1"
                            }
                        }
                    ]
                }
            }
        }
        
        document = client.get_document_by_id(document_id)
        
        assert document is not None
        assert document["document_id"] == document_id
        assert document["title"] == ""  # Default value for missing property
        
        # Test exception handling
        mock_with_additional.do.side_effect = Exception("Query error")
        document = client.get_document_by_id(document_id)
        assert document is None

    @patch.object(VectorDBClient, '_init_client')
    def test_delete_document(self, mock_init_client):
        """Test deleting a document and its chunks."""
        # Set up mock client
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_data_object = MagicMock()
        mock_client.query = mock_query
        mock_client.data_object = mock_data_object
        
        # Mock query chain
        mock_get = MagicMock()
        mock_with_where = MagicMock()
        mock_with_additional = MagicMock()
        mock_do = MagicMock()
        
        mock_query.get.return_value = mock_get
        mock_get.with_where.return_value = mock_with_where
        mock_with_where.with_additional.return_value = mock_with_additional
        mock_with_additional.do.return_value = {
            "data": {
                "Get": {
                    "Document": [
                        {
                            "documentId": "doc123",
                            "chunkIndex": 0,
                            "_additional": {
                                "id": "uuid1"
                            }
                        },
                        {
                            "documentId": "doc123",
                            "chunkIndex": 1,
                            "_additional": {
                                "id": "uuid2"
                            }
                        }
                    ]
                }
            }
        }
        
        # Create client with mocked _init_client
        client = VectorDBClient()
        client.client = mock_client
        
        # Test delete document
        document_id = "doc123"
        result = client.delete_document(document_id)
        
        assert result is True
        assert mock_data_object.delete.call_count == 2
        
        # Test with no chunks found
        mock_data_object.delete.reset_mock()
        mock_with_additional.do.return_value = {"data": {"Get": {"Document": []}}}
        
        result = client.delete_document(document_id)
        
        assert result is True
        assert mock_data_object.delete.call_count == 0
        
        # Test exception handling
        mock_with_additional.do.side_effect = Exception("Query error")
        
        result = client.delete_document(document_id)
        
        assert result is False

    def test_build_where_filter(self):
        """Test building where filter for Weaviate queries."""
        client = VectorDBClient()
        
        # Test simple equality filter
        filters = {"documentType": "text"}
        where_filter = client._build_where_filter(filters)
        
        assert where_filter["path"] == ["documentType"]
        assert where_filter["operator"] == "Equal"
        assert where_filter["valueType"] == "string"
        assert where_filter["value"] == "text"
        
        # Test complex filter with operator
        filters = {"chunkIndex": {"GreaterThan": 5}}
        where_filter = client._build_where_filter(filters)
        
        assert where_filter["path"] == ["chunkIndex"]
        assert where_filter["operator"] == "GreaterThan"
        assert where_filter["valueType"] == "int"
        assert where_filter["value"] == 5

    def test_get_value_type(self):
        """Test determining the Weaviate value type for Python values."""
        client = VectorDBClient()
        
        assert client._get_value_type(True) == "boolean"
        assert client._get_value_type(10) == "int"
        assert client._get_value_type(3.14) == "number"
        assert client._get_value_type("test") == "string"
        assert client._get_value_type([1, 2, 3]) == "string"  # Default for other types

    def test_fallback_generate_valid_uuid(self):
        """Test the fallback function for generating valid UUID."""
        # Test the fallback function directly
        fallback_uuid = generate_valid_uuid("test_seed")
        
        # Verify that the generated UUID is valid
        try:
            uuid_obj = uuid.UUID(fallback_uuid)
            assert uuid_obj.version == 5  # UUID v5
            is_valid = True
        except ValueError:
            is_valid = False
        
        assert is_valid
        
        # Test consistency (same seed should produce same UUID)
        fallback_uuid2 = generate_valid_uuid("test_seed")
        assert fallback_uuid == fallback_uuid2
        
        # Test different seeds produce different UUIDs
        fallback_uuid3 = generate_valid_uuid("different_seed")
        assert fallback_uuid != fallback_uuid3