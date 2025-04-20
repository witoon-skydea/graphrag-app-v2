"""
Weaviate Client Module for GraphRAG Engine

This module provides an interface for interacting with Weaviate Vector Database,
handling schema management, document indexing, and vector search operations.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
import time
from pathlib import Path

import weaviate
from weaviate.client import Client as WeaviateClient
from weaviate.exceptions import WeaviateBaseError

logger = logging.getLogger(__name__)

# Define a fallback function for get_valid_uuid in case import fails
def generate_valid_uuid(seed: str) -> str:
    """
    Generate a valid UUID v5 from a seed string.
    This is a fallback function if weaviate.util.get_valid_uuid is not available.
    
    Args:
        seed: Seed string for UUID generation
        
    Returns:
        UUID string
    """
    # Use UUID5 with a namespace to ensure consistency for the same seed
    namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
    return str(uuid.uuid5(namespace, seed))

# Try to import get_valid_uuid from weaviate, use fallback if it fails
try:
    from weaviate.util import get_valid_uuid
except ImportError:
    logger.warning("Could not import get_valid_uuid from weaviate.util, using fallback function")
    get_valid_uuid = generate_valid_uuid

class VectorDBClient:
    """
    Interface for interacting with Weaviate Vector Database.
    
    This class provides methods for:
    - Connecting to Weaviate
    - Managing schema
    - Indexing documents and chunks
    - Performing vector and hybrid searches
    - Integrating with Knowledge Graph
    """
    
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        openai_api_key: Optional[str] = None,
        batch_size: int = 100,
        timeout_config: Optional[Tuple[int, int]] = (10, 60),  # (timeout_connect, timeout_read)
        schema_class_name: str = "Document",
        default_vectorizer: str = "text2vec-transformers"
    ):
        """
        Initialize the Weaviate client.
        
        Args:
            weaviate_url: URL of the Weaviate instance
            openai_api_key: OpenAI API key for text2vec-openai vectorizer (if used)
            batch_size: Batch size for indexing operations
            timeout_config: Connection and read timeout configuration
            schema_class_name: Default class name for the schema
            default_vectorizer: Default vectorizer to use (text2vec-transformers or text2vec-openai)
        """
        self.weaviate_url = weaviate_url
        self.openai_api_key = openai_api_key
        self.batch_size = batch_size
        self.timeout_config = timeout_config
        self.schema_class_name = schema_class_name
        self.default_vectorizer = default_vectorizer
        
        # Initialize the client
        self.client = self._init_client()
        
        logger.info(f"VectorDBClient initialized with Weaviate at {weaviate_url}")
    
    def _init_client(self) -> WeaviateClient:
        """Initialize and return the Weaviate client."""
        headers = {}
        
        # Add OpenAI API key if provided
        if self.openai_api_key and self.default_vectorizer == "text2vec-openai":
            headers["X-OpenAI-Api-Key"] = self.openai_api_key
        
        # Create client
        client = weaviate.Client(
            url=self.weaviate_url,
            additional_headers=headers,
            timeout_config=self.timeout_config
        )
        
        return client
    
    def check_connection(self) -> bool:
        """
        Check if the connection to Weaviate is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            is_ready = self.client.is_ready()
            logger.info(f"Weaviate connection status: {'Ready' if is_ready else 'Not Ready'}")
            return is_ready
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            return False
    
    def create_schema(self, force_recreate: bool = False) -> bool:
        """
        Create the schema in Weaviate.
        
        Args:
            force_recreate: If True, delete the existing schema class before creating a new one
            
        Returns:
            True if schema creation is successful, False otherwise
        """
        try:
            # Check if schema already exists
            schema_exists = self.client.schema.contains({"class": self.schema_class_name})
            
            if schema_exists:
                if force_recreate:
                    logger.info(f"Deleting existing schema class: {self.schema_class_name}")
                    self.client.schema.delete_class(self.schema_class_name)
                else:
                    logger.info(f"Schema class {self.schema_class_name} already exists")
                    return True
            
            # Define schema class
            class_obj = {
                "class": self.schema_class_name,
                "vectorizer": self.default_vectorizer,
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean"
                    } if self.default_vectorizer == "text2vec-transformers" else None,
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text"
                    } if self.default_vectorizer == "text2vec-openai" else None
                },
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The content of the document chunk",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "The title of the document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "documentId",
                        "dataType": ["string"],
                        "description": "The ID of the document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "chunkIndex",
                        "dataType": ["int"],
                        "description": "The index of the chunk within the document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "documentType",
                        "dataType": ["string"],
                        "description": "The type of document (pdf, docx, etc.)",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "The source of the document (file path, URL, etc.)",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata for the document in JSON format",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "hasEntities",
                        "dataType": ["Entity"],
                        "description": "Entities contained in this document chunk",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    }
                ]
            }
            
            # Create Entity class for knowledge graph
            entity_class = {
                "class": "Entity",
                "vectorizer": self.default_vectorizer,
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean"
                    } if self.default_vectorizer == "text2vec-transformers" else None,
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text"
                    } if self.default_vectorizer == "text2vec-openai" else None
                },
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "The text of the entity",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "type",
                        "dataType": ["string"],
                        "description": "The type of entity (PERSON, ORGANIZATION, etc.)",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "occurrences",
                        "dataType": ["int"],
                        "description": "Number of times this entity appears",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata for the entity in JSON format",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "appearsIn",
                        "dataType": ["Document"],
                        "description": "Documents in which this entity appears",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    },
                    {
                        "name": "relatesTo",
                        "dataType": ["Entity"],
                        "description": "Other entities related to this entity",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-transformers" else None,
                            "text2vec-openai": {
                                "skip": True
                            } if self.default_vectorizer == "text2vec-openai" else None
                        }
                    }
                ]
            }
            
            # Create Relationship class for knowledge graph
            relationship_class = {
                "class": "Relationship",
                "vectorizer": "none",
                "properties": [
                    {
                        "name": "type",
                        "dataType": ["string"],
                        "description": "The type of relationship"
                    },
                    {
                        "name": "confidence",
                        "dataType": ["number"],
                        "description": "Confidence score for the relationship"
                    },
                    {
                        "name": "source",
                        "dataType": ["Entity"],
                        "description": "Source entity of the relationship"
                    },
                    {
                        "name": "target",
                        "dataType": ["Entity"],
                        "description": "Target entity of the relationship"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional metadata for the relationship in JSON format"
                    }
                ]
            }
            
            # Create schema classes
            self.client.schema.create_class(entity_class)
            logger.info(f"Created schema class: Entity")
            
            self.client.schema.create_class(relationship_class)
            logger.info(f"Created schema class: Relationship")
            
            self.client.schema.create_class(class_obj)
            logger.info(f"Created schema class: {self.schema_class_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return False
    
    def add_document_chunk(
        self,
        chunk_text: str,
        document_id: str,
        chunk_index: int,
        title: str = "",
        document_type: str = "text",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        custom_vector: Optional[List[float]] = None
    ) -> Optional[str]:
        """
        Add a document chunk to Weaviate.
        
        Args:
            chunk_text: Text content of the chunk
            document_id: ID of the parent document
            chunk_index: Index of the chunk within the document
            title: Title of the document
            document_type: Type of document (pdf, docx, etc.)
            source: Source of the document (file path, URL, etc.)
            metadata: Additional metadata for the chunk
            custom_vector: Optional custom vector embedding
            
        Returns:
            UUID of the added chunk, or None if addition failed
        """
        try:
            # Check if client is connected and ready
            if not self.check_connection():
                logger.error("Cannot add document chunk: Weaviate client is not connected")
                return None
                
            # Check if required parameters are valid
            if not chunk_text:
                logger.warning("Empty chunk text provided, skipping chunk addition")
                return None
                
            if not document_id:
                logger.warning("Empty document ID provided, skipping chunk addition")
                return None
            
            # Convert any complex metadata to JSON string
            metadata_str = None
            if metadata:
                try:
                    metadata_str = json.dumps(metadata)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to convert metadata to JSON: {e}, using empty metadata")
                    metadata_str = "{}"
            
            # Create data object
            data_object = {
                "content": chunk_text,
                "documentId": document_id,
                "chunkIndex": chunk_index,
                "title": title or "",  # Ensure title is never None
                "documentType": document_type or "text",  # Ensure document_type is never None
                "source": source or ""  # Ensure source is never None
            }
            
            if metadata_str:
                data_object["metadata"] = metadata_str
            
            # Generate UUID based on document_id and chunk_index
            try:
                uuid_seed = f"{document_id}_{chunk_index}"
                chunk_uuid = get_valid_uuid(uuid_seed)
            except Exception as e:
                logger.error(f"Error generating UUID for chunk: {e}")
                # Fallback to a random UUID
                chunk_uuid = str(uuid.uuid4())
            
            # Add the chunk to Weaviate
            try:
                if custom_vector:
                    result = self.client.data_object.create(
                        self.schema_class_name,
                        data_object,
                        uuid=chunk_uuid,
                        vector=custom_vector
                    )
                else:
                    result = self.client.data_object.create(
                        self.schema_class_name,
                        data_object,
                        uuid=chunk_uuid
                    )
                
                logger.debug(f"Added chunk {chunk_index} for document {document_id}")
                return result
            except WeaviateBaseError as e:
                logger.error(f"Weaviate error adding document chunk: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error adding document chunk: {e}")
            return None
    
    def add_document_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        custom_vectors: Optional[List[List[float]]] = None
    ) -> int:
        """
        Add multiple document chunks to Weaviate in batch.
        
        Args:
            chunks: List of chunk dictionaries with fields:
                   {chunk_text, document_id, chunk_index, title, document_type, source, metadata}
            custom_vectors: Optional list of custom vector embeddings for each chunk
            
        Returns:
            Number of chunks successfully added
        """
        try:
            # Configure batch
            self.client.batch.configure(
                batch_size=self.batch_size,
                callback=self._batch_callback
            )
            
            # Start batch
            with self.client.batch as batch:
                for i, chunk in enumerate(chunks):
                    # Convert any complex metadata to JSON string
                    metadata_str = None
                    if "metadata" in chunk and chunk["metadata"]:
                        metadata_str = json.dumps(chunk["metadata"])
                    
                    # Create data object
                    data_object = {
                        "content": chunk["chunk_text"],
                        "documentId": chunk["document_id"],
                        "chunkIndex": chunk["chunk_index"],
                        "title": chunk.get("title", ""),
                        "documentType": chunk.get("document_type", "text"),
                        "source": chunk.get("source", "")
                    }
                    
                    if metadata_str:
                        data_object["metadata"] = metadata_str
                    
                    # Generate UUID based on document_id and chunk_index
                    uuid_seed = f"{chunk['document_id']}_{chunk['chunk_index']}"
                    chunk_uuid = get_valid_uuid(uuid_seed)
                    
                    # Add the chunk to the batch
                    if custom_vectors and i < len(custom_vectors):
                        batch.add_data_object(
                            data_object,
                            self.schema_class_name,
                            uuid=chunk_uuid,
                            vector=custom_vectors[i]
                        )
                    else:
                        batch.add_data_object(
                            data_object,
                            self.schema_class_name,
                            uuid=chunk_uuid
                        )
            
            logger.info(f"Added {len(chunks)} chunks in batch")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error adding document chunks in batch: {e}")
            return 0
    
    def _batch_callback(self, results: Dict[str, Any]) -> None:
        """Callback function for batch operations."""
        if results["errors"]:
            for error in results["errors"]:
                logger.error(f"Batch error: {error}")
    
    def search_by_vector(
        self,
        vector: List[float],
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        include_vector: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search documents by vector similarity.
        
        Args:
            vector: Vector embedding to search with
            limit: Maximum number of results to return
            offset: Offset for pagination
            filters: Additional filters for the search
            include_vector: Whether to include vector embeddings in the results
            
        Returns:
            List of matching documents
        """
        try:
            # Configure the search query
            query = self.client.query.get(self.schema_class_name, ["content", "documentId", "chunkIndex", "title", "documentType", "source", "metadata"])
            
            # Add vector search
            query = query.with_near_vector({
                "vector": vector
            })
            
            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                query = query.with_where(where_filter)
            
            # Execute search
            result = query.with_limit(limit).with_offset(offset).with_additional(["distance", "id"]).do()
            
            # Process results
            search_results = []
            if result and "data" in result and "Get" in result["data"] and self.schema_class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.schema_class_name]:
                    result_item = {
                        "content": item.get("content", ""),
                        "document_id": item.get("documentId", ""),
                        "chunk_index": item.get("chunkIndex", 0),
                        "title": item.get("title", ""),
                        "document_type": item.get("documentType", ""),
                        "source": item.get("source", ""),
                        "distance": item.get("_additional", {}).get("distance", 1.0),
                        "id": item.get("_additional", {}).get("id", "")
                    }
                    
                    # Parse metadata if available
                    if "metadata" in item and item["metadata"]:
                        try:
                            result_item["metadata"] = json.loads(item["metadata"])
                        except:
                            result_item["metadata"] = item["metadata"]
                    
                    search_results.append(result_item)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error searching by vector: {e}")
            return []
    
    def search_by_text(
        self,
        text: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        hybrid_search: bool = True,
        alpha: float = 0.5  # Balance between vector and keyword search for hybrid
    ) -> List[Dict[str, Any]]:
        """
        Search documents by text, using vector search or hybrid search.
        
        Args:
            text: Text to search for
            limit: Maximum number of results to return
            offset: Offset for pagination
            filters: Additional filters for the search
            hybrid_search: Whether to use hybrid search (vector + keyword)
            alpha: Balance between vector and keyword search (0.0 = keyword only, 1.0 = vector only)
            
        Returns:
            List of matching documents
        """
        try:
            # Configure the search query
            query = self.client.query.get(self.schema_class_name, ["content", "documentId", "chunkIndex", "title", "documentType", "source", "metadata"])
            
            # Add search by text
            if hybrid_search:
                # Hybrid search (vector + keyword)
                query = query.with_hybrid(
                    query=text,
                    alpha=alpha,
                    properties=["content", "title"]
                )
            else:
                # Vector search only
                query = query.with_near_text({
                    "concepts": [text]
                })
            
            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                query = query.with_where(where_filter)
            
            # Execute search
            result = query.with_limit(limit).with_offset(offset).with_additional(["score", "id"]).do()
            
            # Process results
            search_results = []
            if result and "data" in result and "Get" in result["data"] and self.schema_class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.schema_class_name]:
                    result_item = {
                        "content": item.get("content", ""),
                        "document_id": item.get("documentId", ""),
                        "chunk_index": item.get("chunkIndex", 0),
                        "title": item.get("title", ""),
                        "document_type": item.get("documentType", ""),
                        "source": item.get("source", ""),
                        "score": item.get("_additional", {}).get("score", 0.0),
                        "id": item.get("_additional", {}).get("id", "")
                    }
                    
                    # Parse metadata if available
                    if "metadata" in item and item["metadata"]:
                        try:
                            result_item["metadata"] = json.loads(item["metadata"])
                        except:
                            result_item["metadata"] = item["metadata"]
                    
                    search_results.append(result_item)
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build a where filter for Weaviate queries from a filters dictionary."""
        where_filter = {}
        
        for field, filter_value in filters.items():
            if isinstance(filter_value, dict):
                # Complex filter with operator
                operator, value = next(iter(filter_value.items()))
                where_filter["path"] = [field]
                where_filter["operator"] = operator
                where_filter["valueType"] = self._get_value_type(value)
                where_filter["value"] = value
            else:
                # Simple equality filter
                where_filter["path"] = [field]
                where_filter["operator"] = "Equal"
                where_filter["valueType"] = self._get_value_type(filter_value)
                where_filter["value"] = filter_value
        
        return where_filter
    
    def _get_value_type(self, value: Any) -> str:
        """Determine the Weaviate value type for a Python value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        else:
            return "string"
    
    def get_document_by_id(
        self,
        document_id: str,
        include_chunks: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            document_id: ID of the document to retrieve
            include_chunks: Whether to include document chunks
            
        Returns:
            Document information or None if not found
        """
        try:
            # Search for chunks with the given document ID
            query = self.client.query.get(self.schema_class_name, ["content", "documentId", "chunkIndex", "title", "documentType", "source", "metadata"])
            query = query.with_where({
                "path": ["documentId"],
                "operator": "Equal",
                "valueType": "string",
                "value": document_id
            })
            
            # Execute query
            result = query.with_additional(["id"]).do()
            
            # Process results
            if result and "data" in result and "Get" in result["data"] and self.schema_class_name in result["data"]["Get"]:
                chunks = result["data"]["Get"][self.schema_class_name]
                
                if not chunks:
                    return None
                
                # Extract document metadata from the first chunk
                first_chunk = chunks[0]
                document = {
                    "document_id": document_id,
                    "title": first_chunk.get("title", ""),
                    "document_type": first_chunk.get("documentType", ""),
                    "source": first_chunk.get("source", ""),
                    "chunk_count": len(chunks)
                }
                
                # Parse metadata if available
                if "metadata" in first_chunk and first_chunk["metadata"]:
                    try:
                        document["metadata"] = json.loads(first_chunk["metadata"])
                    except:
                        document["metadata"] = first_chunk["metadata"]
                
                # Include chunks if requested
                if include_chunks:
                    # Sort chunks by chunk index
                    sorted_chunks = sorted(chunks, key=lambda x: x.get("chunkIndex", 0))
                    
                    document["chunks"] = []
                    for chunk in sorted_chunks:
                        chunk_data = {
                            "content": chunk.get("content", ""),
                            "chunk_index": chunk.get("chunkIndex", 0),
                            "id": chunk.get("_additional", {}).get("id", "")
                        }
                        document["chunks"].append(chunk_data)
                
                return document
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from Weaviate.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion is successful, False otherwise
        """
        try:
            # Find all chunks with the given document ID
            query = self.client.query.get(self.schema_class_name, ["documentId", "chunkIndex"])
            query = query.with_where({
                "path": ["documentId"],
                "operator": "Equal",
                "valueType": "string",
                "value": document_id
            })
            
            # Execute query
            result = query.with_additional(["id"]).do()
            
            # Delete chunks
            deleted_count = 0
            if result and "data" in result and "Get" in result["data"] and self.schema_class_name in result["data"]["Get"]:
                chunks = result["data"]["Get"][self.schema_class_name]
                
                for chunk in chunks:
                    chunk_id = chunk.get("_additional", {}).get("id", "")
                    if chunk_id:
                        self.client.data_object.delete(
                            uuid=chunk_id,
                            class_name=self.schema_class_name
                        )
                        deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def add_entity(
        self,
        entity_text: str,
        entity_type: str,
        occurrences: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        custom_vector: Optional[List[float]] = None
    ) -> Optional[str]:
        """
        Add an entity to Weaviate.
        
        Args:
            entity_text: Text of the entity
            entity_type: Type of the entity
            occurrences: Number of occurrences of the entity
            metadata: Additional metadata for the entity
            custom_vector: Optional custom vector embedding
            
        Returns:
            UUID of the added entity, or None if addition failed
        """
        try:
            # Convert any complex metadata to JSON string
            metadata_str = None
            if metadata:
                metadata_str = json.dumps(metadata)
            
            # Create data object
            data_object = {
                "text": entity_text,
                "type": entity_type,
                "occurrences": occurrences
            }
            
            if metadata_str:
                data_object["metadata"] = metadata_str
            
            # Generate UUID based on entity text and type
            uuid_seed = f"{entity_text}_{entity_type}"
            entity_uuid = get_valid_uuid(uuid_seed)
            
            # Add the entity to Weaviate
            if custom_vector:
                result = self.client.data_object.create(
                    "Entity",
                    data_object,
                    uuid=entity_uuid,
                    vector=custom_vector
                )
            else:
                result = self.client.data_object.create(
                    "Entity",
                    data_object,
                    uuid=entity_uuid
                )
            
            logger.debug(f"Added entity {entity_text} of type {entity_type}")
            return result
        
        except Exception as e:
            logger.error(f"Error adding entity: {e}")
            return None
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a relationship between entities to Weaviate.
        
        Args:
            source_id: UUID of the source entity
            target_id: UUID of the target entity
            relationship_type: Type of the relationship
            confidence: Confidence score for the relationship
            metadata: Additional metadata for the relationship
            
        Returns:
            UUID of the added relationship, or None if addition failed
        """
        try:
            # Convert any complex metadata to JSON string
            metadata_str = None
            if metadata:
                metadata_str = json.dumps(metadata)
            
            # Create data object
            data_object = {
                "type": relationship_type,
                "confidence": confidence
            }
            
            if metadata_str:
                data_object["metadata"] = metadata_str
            
            # Generate UUID for the relationship
            uuid_seed = f"{source_id}_{target_id}_{relationship_type}"
            relationship_uuid = get_valid_uuid(uuid_seed)
            
            # Add the relationship to Weaviate
            result = self.client.data_object.create(
                "Relationship",
                data_object,
                uuid=relationship_uuid
            )
            
            # Create cross-references
            self.client.data_object.reference.add(
                from_class_name="Relationship",
                from_uuid=relationship_uuid,
                from_property_name="source",
                to_class_name="Entity",
                to_uuid=source_id
            )
            
            self.client.data_object.reference.add(
                from_class_name="Relationship",
                from_uuid=relationship_uuid,
                from_property_name="target",
                to_class_name="Entity",
                to_uuid=target_id
            )
            
            # Add cross-reference from source entity to target entity
            self.client.data_object.reference.add(
                from_class_name="Entity",
                from_uuid=source_id,
                from_property_name="relatesTo",
                to_class_name="Entity",
                to_uuid=target_id
            )
            
            logger.debug(f"Added relationship {relationship_type} between entities {source_id} and {target_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
            return None
    
    def link_entity_to_document(
        self,
        entity_id: str,
        document_id: str,
        chunk_index: int
    ) -> bool:
        """
        Link an entity to a document chunk.
        
        Args:
            entity_id: UUID of the entity
            document_id: ID of the document
            chunk_index: Index of the chunk within the document
            
        Returns:
            True if linking is successful, False otherwise
        """
        try:
            # Generate UUID for the document chunk
            uuid_seed = f"{document_id}_{chunk_index}"
            chunk_uuid = get_valid_uuid(uuid_seed)
            
            # Add cross-reference from entity to document
            self.client.data_object.reference.add(
                from_class_name="Entity",
                from_uuid=entity_id,
                from_property_name="appearsIn",
                to_class_name=self.schema_class_name,
                to_uuid=chunk_uuid
            )
            
            # Add cross-reference from document to entity
            self.client.data_object.reference.add(
                from_class_name=self.schema_class_name,
                from_uuid=chunk_uuid,
                from_property_name="hasEntities",
                to_class_name="Entity",
                to_uuid=entity_id
            )
            
            logger.debug(f"Linked entity {entity_id} to document {document_id} chunk {chunk_index}")
            return True
        
        except Exception as e:
            logger.error(f"Error linking entity to document: {e}")
            return False
    
    def get_document_entities(
        self,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all entities linked to a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of entities linked to the document
        """
        try:
            # First, get all chunks for the document
            query = self.client.query.get(self.schema_class_name, ["documentId", "chunkIndex"])
            query = query.with_where({
                "path": ["documentId"],
                "operator": "Equal",
                "valueType": "string",
                "value": document_id
            })
            
            # Add hasEntities reference
            query = query.with_additional(["id"]).with_fields("hasEntities { ... on Entity { text type occurrences metadata _additional { id } } }")
            
            # Execute query
            result = query.do()
            
            # Process results
            entities = []
            seen_entity_ids = set()
            
            if result and "data" in result and "Get" in result["data"] and self.schema_class_name in result["data"]["Get"]:
                chunks = result["data"]["Get"][self.schema_class_name]
                
                for chunk in chunks:
                    if "hasEntities" in chunk:
                        for entity in chunk["hasEntities"]:
                            entity_id = entity.get("_additional", {}).get("id", "")
                            
                            # Skip if we've already seen this entity
                            if entity_id in seen_entity_ids:
                                continue
                            
                            seen_entity_ids.add(entity_id)
                            
                            entity_data = {
                                "text": entity.get("text", ""),
                                "type": entity.get("type", ""),
                                "occurrences": entity.get("occurrences", 0),
                                "id": entity_id
                            }
                            
                            # Parse metadata if available
                            if "metadata" in entity and entity["metadata"]:
                                try:
                                    entity_data["metadata"] = json.loads(entity["metadata"])
                                except:
                                    entity_data["metadata"] = entity["metadata"]
                            
                            entities.append(entity_data)
            
            return entities
        
        except Exception as e:
            logger.error(f"Error getting document entities: {e}")
            return []
    
    def get_entity_documents(
        self,
        entity_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all documents linked to an entity.
        
        Args:
            entity_id: UUID of the entity
            
        Returns:
            List of documents linked to the entity
        """
        try:
            # Query the entity
            query = self.client.query.get("Entity", ["text", "type"])
            query = query.with_additional(["id"]).with_fields("appearsIn { ... on " + self.schema_class_name + " { documentId chunkIndex title source _additional { id } } }")
            
            # Execute query
            result = query.by_id(entity_id).do()
            
            # Process results
            documents = []
            seen_doc_ids = set()
            
            if result and "data" in result and "Get" in result["data"] and "Entity" in result["data"]["Get"]:
                entity = result["data"]["Get"]["Entity"]
                
                if "appearsIn" in entity:
                    for doc_chunk in entity["appearsIn"]:
                        document_id = doc_chunk.get("documentId", "")
                        
                        # Skip if we've already seen this document
                        if document_id in seen_doc_ids:
                            continue
                        
                        seen_doc_ids.add(document_id)
                        
                        document_data = {
                            "document_id": document_id,
                            "title": doc_chunk.get("title", ""),
                            "source": doc_chunk.get("source", "")
                        }
                        
                        documents.append(document_data)
            
            return documents
        
        except Exception as e:
            logger.error(f"Error getting entity documents: {e}")
            return []
    
    def search_entities(
        self,
        text: str = "",
        entity_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by text and/or type.
        
        Args:
            text: Text to search for
            entity_type: Type of entities to search for
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of matching entities
        """
        try:
            # Configure the search query
            query = self.client.query.get("Entity", ["text", "type", "occurrences", "metadata"])
            
            # Add filters
            where_filter = None
            
            if entity_type:
                where_filter = {
                    "path": ["type"],
                    "operator": "Equal",
                    "valueType": "string",
                    "value": entity_type
                }
            
            if where_filter:
                query = query.with_where(where_filter)
            
            # Add search by text if provided
            if text:
                query = query.with_near_text({
                    "concepts": [text]
                })
            
            # Execute search
            result = query.with_limit(limit).with_offset(offset).with_additional(["id", "score"]).do()
            
            # Process results
            entities = []
            if result and "data" in result and "Get" in result["data"] and "Entity" in result["data"]["Get"]:
                for entity in result["data"]["Get"]["Entity"]:
                    entity_data = {
                        "text": entity.get("text", ""),
                        "type": entity.get("type", ""),
                        "occurrences": entity.get("occurrences", 0),
                        "id": entity.get("_additional", {}).get("id", ""),
                        "score": entity.get("_additional", {}).get("score", 0.0)
                    }
                    
                    # Parse metadata if available
                    if "metadata" in entity and entity["metadata"]:
                        try:
                            entity_data["metadata"] = json.loads(entity["metadata"])
                        except:
                            entity_data["metadata"] = entity["metadata"]
                    
                    entities.append(entity_data)
            
            return entities
        
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
    
    def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_id: UUID of the entity
            relationship_type: Type of relationships to filter for
            min_confidence: Minimum confidence score for relationships
            
        Returns:
            Dictionary with "outgoing" and "incoming" relationships
        """
        try:
            result = {
                "outgoing": [],
                "incoming": []
            }
            
            # Query outgoing relationships
            out_query = self.client.query.get("Relationship", ["type", "confidence", "metadata"])
            out_query = out_query.with_fields("source { ... on Entity { text type _additional { id } } }")
            out_query = out_query.with_fields("target { ... on Entity { text type _additional { id } } }")
            
            # Add filter for source entity
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["source", "Entity", "_additional", "id"],
                        "operator": "Equal",
                        "valueType": "string",
                        "value": entity_id
                    }
                ]
            }
            
            # Add relationship type filter if provided
            if relationship_type:
                where_filter["operands"].append({
                    "path": ["type"],
                    "operator": "Equal",
                    "valueType": "string",
                    "value": relationship_type
                })
            
            # Add confidence filter
            if min_confidence > 0:
                where_filter["operands"].append({
                    "path": ["confidence"],
                    "operator": "GreaterThanEqual",
                    "valueType": "number",
                    "value": min_confidence
                })
            
            out_query = out_query.with_where(where_filter)
            
            # Execute query
            out_result = out_query.with_additional(["id"]).do()
            
            # Process outgoing relationships
            if out_result and "data" in out_result and "Get" in out_result["data"] and "Relationship" in out_result["data"]["Get"]:
                for rel in out_result["data"]["Get"]["Relationship"]:
                    if "source" in rel and "target" in rel:
                        rel_data = {
                            "type": rel.get("type", ""),
                            "confidence": rel.get("confidence", 0.0),
                            "id": rel.get("_additional", {}).get("id", ""),
                            "source": {
                                "text": rel["source"].get("text", ""),
                                "type": rel["source"].get("type", ""),
                                "id": rel["source"].get("_additional", {}).get("id", "")
                            },
                            "target": {
                                "text": rel["target"].get("text", ""),
                                "type": rel["target"].get("type", ""),
                                "id": rel["target"].get("_additional", {}).get("id", "")
                            }
                        }
                        
                        # Parse metadata if available
                        if "metadata" in rel and rel["metadata"]:
                            try:
                                rel_data["metadata"] = json.loads(rel["metadata"])
                            except:
                                rel_data["metadata"] = rel["metadata"]
                        
                        result["outgoing"].append(rel_data)
            
            # Query incoming relationships
            in_query = self.client.query.get("Relationship", ["type", "confidence", "metadata"])
            in_query = in_query.with_fields("source { ... on Entity { text type _additional { id } } }")
            in_query = in_query.with_fields("target { ... on Entity { text type _additional { id } } }")
            
            # Add filter for target entity
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["target", "Entity", "_additional", "id"],
                        "operator": "Equal",
                        "valueType": "string",
                        "value": entity_id
                    }
                ]
            }
            
            # Add relationship type filter if provided
            if relationship_type:
                where_filter["operands"].append({
                    "path": ["type"],
                    "operator": "Equal",
                    "valueType": "string",
                    "value": relationship_type
                })
            
            # Add confidence filter
            if min_confidence > 0:
                where_filter["operands"].append({
                    "path": ["confidence"],
                    "operator": "GreaterThanEqual",
                    "valueType": "number",
                    "value": min_confidence
                })
            
            in_query = in_query.with_where(where_filter)
            
            # Execute query
            in_result = in_query.with_additional(["id"]).do()
            
            # Process incoming relationships
            if in_result and "data" in in_result and "Get" in in_result["data"] and "Relationship" in in_result["data"]["Get"]:
                for rel in in_result["data"]["Get"]["Relationship"]:
                    if "source" in rel and "target" in rel:
                        rel_data = {
                            "type": rel.get("type", ""),
                            "confidence": rel.get("confidence", 0.0),
                            "id": rel.get("_additional", {}).get("id", ""),
                            "source": {
                                "text": rel["source"].get("text", ""),
                                "type": rel["source"].get("type", ""),
                                "id": rel["source"].get("_additional", {}).get("id", "")
                            },
                            "target": {
                                "text": rel["target"].get("text", ""),
                                "type": rel["target"].get("type", ""),
                                "id": rel["target"].get("_additional", {}).get("id", "")
                            }
                        }
                        
                        # Parse metadata if available
                        if "metadata" in rel and rel["metadata"]:
                            try:
                                rel_data["metadata"] = json.loads(rel["metadata"])
                            except:
                                rel_data["metadata"] = rel["metadata"]
                        
                        result["incoming"].append(rel_data)
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return {"outgoing": [], "incoming": []}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "document_count": 0,
                "chunk_count": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "document_types": {},
                "entity_types": {}
            }
            
            # Count document chunks
            doc_query = self.client.query.aggregate(self.schema_class_name).with_meta_count().do()
            if doc_query and "data" in doc_query and "Aggregate" in doc_query["data"] and self.schema_class_name in doc_query["data"]["Aggregate"]:
                stats["chunk_count"] = doc_query["data"]["Aggregate"][self.schema_class_name][0]["meta"]["count"]
            
            # Count distinct documents
            doc_query = self.client.query.get(self.schema_class_name, ["documentId"]).with_additional(["id"]).do()
            if doc_query and "data" in doc_query and "Get" in doc_query["data"] and self.schema_class_name in doc_query["data"]["Get"]:
                doc_ids = set()
                for chunk in doc_query["data"]["Get"][self.schema_class_name]:
                    if "documentId" in chunk:
                        doc_ids.add(chunk["documentId"])
                stats["document_count"] = len(doc_ids)
            
            # Count document types
            type_query = self.client.query.get(self.schema_class_name, ["documentType"]).with_additional(["id"]).do()
            if type_query and "data" in type_query and "Get" in type_query["data"] and self.schema_class_name in type_query["data"]["Get"]:
                doc_types = {}
                for chunk in type_query["data"]["Get"][self.schema_class_name]:
                    if "documentType" in chunk:
                        doc_type = chunk["documentType"]
                        if doc_type not in doc_types:
                            doc_types[doc_type] = 0
                        doc_types[doc_type] += 1
                stats["document_types"] = doc_types
            
            # Count entities
            entity_query = self.client.query.aggregate("Entity").with_meta_count().do()
            if entity_query and "data" in entity_query and "Aggregate" in entity_query["data"] and "Entity" in entity_query["data"]["Aggregate"]:
                stats["entity_count"] = entity_query["data"]["Aggregate"]["Entity"][0]["meta"]["count"]
            
            # Count entity types
            entity_type_query = self.client.query.get("Entity", ["type"]).with_additional(["id"]).do()
            if entity_type_query and "data" in entity_type_query and "Get" in entity_type_query["data"] and "Entity" in entity_type_query["data"]["Get"]:
                entity_types = {}
                for entity in entity_type_query["data"]["Get"]["Entity"]:
                    if "type" in entity:
                        entity_type = entity["type"]
                        if entity_type not in entity_types:
                            entity_types[entity_type] = 0
                        entity_types[entity_type] += 1
                stats["entity_types"] = entity_types
            
            # Count relationships
            rel_query = self.client.query.aggregate("Relationship").with_meta_count().do()
            if rel_query and "data" in rel_query and "Aggregate" in rel_query["data"] and "Relationship" in rel_query["data"]["Aggregate"]:
                stats["relationship_count"] = rel_query["data"]["Aggregate"]["Relationship"][0]["meta"]["count"]
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "document_count": 0,
                "chunk_count": 0,
                "entity_count": 0,
                "relationship_count": 0,
                "document_types": {},
                "entity_types": {},
                "error": str(e)
            }
