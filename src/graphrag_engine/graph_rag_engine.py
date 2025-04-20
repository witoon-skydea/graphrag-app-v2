"""
GraphRAG Engine Module

This module integrates the Knowledge Graph Builder and Vector Database,
providing a unified interface for the GraphRAG system.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Set, TYPE_CHECKING
from pathlib import Path
import hashlib
from tqdm import tqdm
import time

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from .vector_db.weaviate_client import VectorDBClient

logger = logging.getLogger(__name__)

class GraphRAGEngine:
    """
    Main class for the GraphRAG Engine.
    
    The GraphRAGEngine integrates the Knowledge Graph Builder and Vector Database,
    providing methods for document processing, searching, and retrieving information.
    """
    
    def __init__(
        self,
        knowledge_graph_builder = None,  # Type hint removed to avoid circular import
        vector_db_client = None,  # Type hint removed to avoid circular import
        persist_path: Optional[str] = None,
        extraction_method: str = "ollama",
        identification_method: str = "ollama",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        weaviate_url: str = "http://localhost:8080",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize a GraphRAGEngine.
        
        Args:
            knowledge_graph_builder: Existing KnowledgeGraphBuilder instance
            vector_db_client: Existing VectorDBClient instance
            persist_path: Directory path for persisting data
            extraction_method: Method for entity extraction
            identification_method: Method for relationship identification
            model_name: Name of the model to use
            api_key: API key for external APIs
            weaviate_url: URL of the Weaviate instance
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.persist_path = persist_path
        self.extraction_method = extraction_method
        self.identification_method = identification_method
        self.model_name = model_name
        self.api_key = api_key
        self.weaviate_url = weaviate_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Import inside method to avoid circular imports
        if knowledge_graph_builder is None:
            from .knowledge_graph.graph_builder import KnowledgeGraphBuilder
            knowledge_graph_builder = KnowledgeGraphBuilder(
                extraction_method=extraction_method,
                identification_method=identification_method,
                model_name=model_name,
                api_key=api_key,
                persist_path=persist_path + "/knowledge_graph" if persist_path else None
            )
        
        if vector_db_client is None:
            from .vector_db.weaviate_client import VectorDBClient
            vector_db_client = VectorDBClient(
                weaviate_url=weaviate_url,
                openai_api_key=api_key if extraction_method == "openai" else None
            )
        
        # Initialize components
        self.kg_builder = knowledge_graph_builder
        self.vector_db = vector_db_client
        
        # Create persistence directory if specified
        if self.persist_path:
            os.makedirs(self.persist_path, exist_ok=True)
            
        logger.info(f"GraphRAGEngine initialized with {extraction_method} extraction and {weaviate_url} Weaviate")
    
    def initialize(self, setup_vector_db: bool = True) -> bool:
        """
        Initialize the GraphRAG Engine components.
        
        Args:
            setup_vector_db: Whether to set up the vector database schema
            
        Returns:
            True if initialization is successful, False otherwise
        """
        result = True
        
        # Check vector database connection
        vector_db_ready = self.vector_db.check_connection()
        if not vector_db_ready:
            logger.error("Vector database connection failed")
            result = False
        
        # Set up vector database schema if requested
        if setup_vector_db and vector_db_ready:
            schema_created = self.vector_db.create_schema(force_recreate=False)
            if not schema_created:
                logger.error("Vector database schema creation failed")
                result = False
        
        return result
    
    def process_document(
        self,
        document_text: str,
        document_id: str,
        title: str = "",
        document_type: str = "text",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        extract_entities: bool = True,
        identify_relationships: bool = True,
        custom_vectors: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Process a document for indexing and knowledge graph construction.
        
        Args:
            document_text: Text content of the document
            document_id: Unique identifier for the document
            title: Title of the document
            document_type: Type of document (pdf, docx, etc.)
            source: Source of the document (file path, URL, etc.)
            metadata: Additional metadata for the document
            extract_entities: Whether to extract entities
            identify_relationships: Whether to identify relationships
            custom_vectors: Optional custom vector embeddings for chunks
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "document_id": document_id,
            "chunks_created": 0,
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_identified": 0,
            "edges_created": 0
        }
        
        # Split document into chunks
        chunks = self._chunk_document(document_text, self.chunk_size, self.chunk_overlap)
        result["chunks_created"] = len(chunks)
        
        # Process chunks with Knowledge Graph Builder
        if extract_entities:
            logger.info(f"Processing document {document_id} with Knowledge Graph Builder")
            kg_result = self.kg_builder.process_document(
                document_text=document_text,
                document_id=document_id,
                extract_entities=extract_entities,
                identify_relationships=identify_relationships
            )
            
            result["entities_extracted"] = kg_result["entities_extracted"]
            result["nodes_created"] = kg_result["nodes_created"]
            result["relationships_identified"] = kg_result["relationships_identified"]
            result["edges_created"] = kg_result["edges_created"]
        
        # Index chunks in Vector Database
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                "chunk_text": chunk_text,
                "document_id": document_id,
                "chunk_index": i,
                "title": title,
                "document_type": document_type,
                "source": source,
                "metadata": metadata
            }
            chunk_objects.append(chunk)
        
        if custom_vectors:
            batch_result = self.vector_db.add_document_chunks_batch(chunk_objects, custom_vectors)
        else:
            batch_result = self.vector_db.add_document_chunks_batch(chunk_objects)
        
        result["chunks_indexed"] = batch_result
        
        # Link entities to document chunks in Vector Database
        if extract_entities:
            # Get extracted entities
            entities = self.kg_builder.query_entities(document_id=document_id)
            
            # Add entities to Vector Database
            for entity in entities:
                entity_id = self.vector_db.add_entity(
                    entity_text=entity["text"],
                    entity_type=entity["type"],
                    occurrences=entity.get("occurrences", 1),
                    metadata=entity.get("metadata")
                )
                
                if entity_id:
                    # Link entity to document chunks
                    for context in entity.get("contexts", []):
                        # Find the chunk that contains this entity
                        for i, chunk in enumerate(chunks):
                            # Simple overlap check
                            if context["start_pos"] < (i + 1) * self.chunk_size - self.chunk_overlap:
                                self.vector_db.link_entity_to_document(
                                    entity_id=entity_id,
                                    document_id=document_id,
                                    chunk_index=i
                                )
                                break
        
        return result
    
    def _chunk_document(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split a document into overlapping chunks.
        
        Args:
            text: Document text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))
            
            # If this is not the last chunk, try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                next_para = text.find("\n\n", end - chunk_overlap, end + chunk_overlap)
                if next_para != -1:
                    end = next_para + 2  # Include the paragraph break
                else:
                    # Look for sentence break (period followed by space)
                    next_sentence = text.find(". ", end - chunk_overlap, end + 50)
                    if next_sentence != -1:
                        end = next_sentence + 2  # Include the period and space
                    else:
                        # Look for space
                        next_space = text.find(" ", end - 20, end + 20)
                        if next_space != -1:
                            end = next_space + 1  # Include the space
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position
            start = max(end - chunk_overlap, start + 1)
        
        return chunks
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        hybrid_search: bool = True,
        use_knowledge_graph: bool = True,
        alpha: float = 0.5,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search for relevant information using GraphRAG.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            filters: Additional filters for the search
            hybrid_search: Whether to use hybrid search (vector + keyword)
            use_knowledge_graph: Whether to enhance results with knowledge graph
            alpha: Balance between vector and keyword search
            min_confidence: Minimum confidence for knowledge graph relationships
            
        Returns:
            Dictionary with search results
        """
        results = {
            "query": query,
            "vector_results": [],
            "knowledge_graph_results": [],
            "combined_results": []
        }
        
        # Perform vector search
        vector_results = self.vector_db.search_by_text(
            text=query,
            limit=limit,
            filters=filters,
            hybrid_search=hybrid_search,
            alpha=alpha
        )
        
        results["vector_results"] = vector_results
        
        # Extract relevant document IDs
        doc_ids = set()
        for result in vector_results:
            doc_ids.add(result["document_id"])
        
        # Enhance results with knowledge graph if requested
        if use_knowledge_graph and vector_results:
            # Extract entities from the query
            query_entities = self.kg_builder.entity_extractor.extract_entities(query)
            
            kg_results = []
            
            # Find entities in the knowledge graph similar to query entities
            for query_entity in query_entities:
                kg_entities = self.kg_builder.query_entities(
                    entity_text=query_entity["text"],
                    entity_type=query_entity["type"],
                    limit=5
                )
                
                for entity in kg_entities:
                    # Get related entities
                    relationships = self.kg_builder.get_entity_relationships(
                        entity_text=entity["text"],
                        include_incoming=True,
                        include_outgoing=True,
                        min_confidence=min_confidence
                    )
                    
                    # Add related entities to results
                    for direction in ["incoming", "outgoing"]:
                        for rel in relationships[direction]:
                            related_entity = rel["source"] if direction == "incoming" else rel["target"]
                            
                            # Find documents containing the related entity
                            entity_node = self.kg_builder.node_registry.get_node_by_entity(related_entity["text"])
                            if entity_node and "documents" in entity_node:
                                for doc_id in entity_node["documents"]:
                                    # Skip if this document is already in vector results
                                    if doc_id in doc_ids:
                                        continue
                                    
                                    # Get document from vector database
                                    doc = self.vector_db.get_document_by_id(doc_id, include_chunks=True)
                                    if doc:
                                        kg_results.append({
                                            "document_id": doc_id,
                                            "title": doc.get("title", ""),
                                            "source": doc.get("source", ""),
                                            "content": doc.get("chunks", [{}])[0].get("content", "") if doc.get("chunks") else "",
                                            "relationship": {
                                                "type": rel["type"],
                                                "confidence": rel["confidence"],
                                                "source_entity": related_entity["text"] if direction == "incoming" else entity["text"],
                                                "target_entity": entity["text"] if direction == "incoming" else related_entity["text"]
                                            }
                                        })
                                        
                                        # Add to set of seen document IDs
                                        doc_ids.add(doc_id)
            
            results["knowledge_graph_results"] = kg_results
        
        # Combine results
        combined_results = []
        
        # Add vector results first
        for result in vector_results:
            combined_results.append({
                "document_id": result["document_id"],
                "chunk_index": result["chunk_index"],
                "content": result["content"],
                "title": result["title"],
                "source": result["source"],
                "score": result["score"],
                "id": result["id"],
                "result_type": "vector"
            })
        
        # Add knowledge graph results
        if use_knowledge_graph:
            for result in results["knowledge_graph_results"]:
                combined_results.append({
                    "document_id": result["document_id"],
                    "content": result["content"],
                    "title": result["title"],
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "result_type": "knowledge_graph"
                })
        
        # Limit combined results
        combined_results = combined_results[:limit * 2]
        results["combined_results"] = combined_results
        
        return results
    
    def get_document(
        self,
        document_id: str,
        include_chunks: bool = True,
        include_entities: bool = True,
        include_relationships: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID, including vector and knowledge graph information.
        
        Args:
            document_id: ID of the document to retrieve
            include_chunks: Whether to include document chunks
            include_entities: Whether to include entities linked to the document
            include_relationships: Whether to include relationships between entities
            
        Returns:
            Document information or None if not found
        """
        # Get document from vector database
        document = self.vector_db.get_document_by_id(document_id, include_chunks)
        
        if not document:
            return None
        
        # Add entities if requested
        if include_entities:
            entities = self.vector_db.get_document_entities(document_id)
            document["entities"] = entities
            
            # Add relationships if requested
            if include_relationships and entities:
                for entity in document["entities"]:
                    entity_relationships = self.kg_builder.get_entity_relationships(
                        entity_text=entity["text"],
                        include_incoming=True,
                        include_outgoing=True
                    )
                    entity["relationships"] = entity_relationships
        
        return document
    
    def get_entity(
        self,
        entity_text: str,
        include_documents: bool = True,
        include_relationships: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entity by its text, including linked documents and relationships.
        
        Args:
            entity_text: Text of the entity to retrieve
            include_documents: Whether to include documents linked to the entity
            include_relationships: Whether to include relationships with other entities
            
        Returns:
            Entity information or None if not found
        """
        # Get entity from knowledge graph
        entity_node = self.kg_builder.node_registry.get_node_by_entity(entity_text)
        
        if not entity_node:
            return None
        
        # Create entity object
        entity = {
            "text": entity_node["text"],
            "type": entity_node["type"],
            "occurrences": entity_node.get("occurrences", 0),
            "id": entity_node["id"]
        }
        
        # Add metadata if available
        if "metadata" in entity_node:
            entity["metadata"] = entity_node["metadata"]
        
        # Add documents if requested
        if include_documents:
            documents = []
            doc_ids = set(entity_node.get("documents", []))
            
            for doc_id in doc_ids:
                # Get basic document info (without chunks)
                doc = self.vector_db.get_document_by_id(doc_id, include_chunks=False)
                if doc:
                    documents.append({
                        "document_id": doc_id,
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "document_type": doc.get("document_type", "")
                    })
            
            entity["documents"] = documents
        
        # Add relationships if requested
        if include_relationships:
            relationships = self.kg_builder.get_entity_relationships(
                entity_text=entity_text,
                include_incoming=True,
                include_outgoing=True
            )
            entity["relationships"] = relationships
        
        return entity
    
    def get_entity_network(
        self,
        entity_text: str,
        max_depth: int = 2,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get a network of entities connected to the specified entity.
        
        Args:
            entity_text: Text of the central entity
            max_depth: Maximum number of hops from the central entity
            min_confidence: Minimum confidence for relationships
            limit: Maximum number of entities to include
            
        Returns:
            Network of entities and relationships
        """
        network = {
            "central_entity": entity_text,
            "nodes": [],
            "edges": []
        }
        
        # Get central entity
        central_entity = self.kg_builder.node_registry.get_node_by_entity(entity_text)
        if not central_entity:
            return network
        
        # Add central entity to nodes
        network["nodes"].append({
            "id": central_entity["id"],
            "text": central_entity["text"],
            "type": central_entity["type"],
            "occurrences": central_entity.get("occurrences", 0),
            "group": 0  # Central entity group
        })
        
        # Keep track of processed nodes and edges
        processed_nodes = {central_entity["id"]}
        processed_edges = set()
        
        # BFS to explore the network
        queue = [(central_entity["id"], 0)]  # (node_id, depth)
        
        while queue and len(network["nodes"]) < limit:
            node_id, depth = queue.pop(0)
            
            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue
            
            # Get relationships for this node
            entity_node = self.kg_builder.node_registry.get_node(node_id)
            if not entity_node:
                continue
                
            entity_text = entity_node["text"]
            
            relationships = self.kg_builder.get_entity_relationships(
                entity_text=entity_text,
                include_incoming=True,
                include_outgoing=True,
                min_confidence=min_confidence
            )
            
            # Process outgoing relationships
            for rel in relationships["outgoing"]:
                target_id = rel["target"]["id"]
                
                # Add target node if not already processed
                if target_id not in processed_nodes:
                    processed_nodes.add(target_id)
                    
                    target_entity = self.kg_builder.node_registry.get_node(target_id)
                    if target_entity:
                        network["nodes"].append({
                            "id": target_id,
                            "text": target_entity["text"],
                            "type": target_entity["type"],
                            "occurrences": target_entity.get("occurrences", 0),
                            "group": depth + 1  # Group by depth
                        })
                        
                        # Add to queue for further exploration
                        queue.append((target_id, depth + 1))
                
                # Add edge if not already processed
                edge_id = f"{node_id}_{target_id}_{rel['type']}"
                if edge_id not in processed_edges:
                    processed_edges.add(edge_id)
                    
                    network["edges"].append({
                        "id": edge_id,
                        "source": node_id,
                        "target": target_id,
                        "type": rel["type"],
                        "confidence": rel["confidence"]
                    })
            
            # Process incoming relationships
            for rel in relationships["incoming"]:
                source_id = rel["source"]["id"]
                
                # Add source node if not already processed
                if source_id not in processed_nodes:
                    processed_nodes.add(source_id)
                    
                    source_entity = self.kg_builder.node_registry.get_node(source_id)
                    if source_entity:
                        network["nodes"].append({
                            "id": source_id,
                            "text": source_entity["text"],
                            "type": source_entity["type"],
                            "occurrences": source_entity.get("occurrences", 0),
                            "group": depth + 1  # Group by depth
                        })
                        
                        # Add to queue for further exploration
                        queue.append((source_id, depth + 1))
                
                # Add edge if not already processed
                edge_id = f"{source_id}_{node_id}_{rel['type']}"
                if edge_id not in processed_edges:
                    processed_edges.add(edge_id)
                    
                    network["edges"].append({
                        "id": edge_id,
                        "source": source_id,
                        "target": node_id,
                        "type": rel["type"],
                        "confidence": rel["confidence"]
                    })
        
        return network
    
    def find_paths_between_entities(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 3,
        min_confidence: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities in the knowledge graph.
        
        Args:
            source_entity: Text of the source entity
            target_entity: Text of the target entity
            max_depth: Maximum path length
            min_confidence: Minimum confidence for relationships
            
        Returns:
            List of paths, where each path is a list of relationships
        """
        return self.kg_builder.get_paths_between_entities(
            source_text=source_entity,
            target_text=target_entity,
            max_depth=max_depth,
            min_confidence=min_confidence
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the GraphRAG system.
        
        Returns:
            Dictionary with statistics
        """
        # Get knowledge graph statistics
        kg_stats = self.kg_builder.graph_statistics()
        
        # Get vector database statistics
        vdb_stats = self.vector_db.get_statistics()
        
        # Combine statistics
        stats = {
            "knowledge_graph": kg_stats,
            "vector_database": vdb_stats,
            "total_documents": vdb_stats.get("document_count", 0),
            "total_entities": kg_stats.get("total_nodes", 0),
            "total_relationships": kg_stats.get("total_edges", 0)
        }
        
        return stats
    
    def export_knowledge_graph(self, output_file: str) -> None:
        """
        Export the knowledge graph to a file for visualization.
        
        Args:
            output_file: Path to the output file
        """
        self.kg_builder.export_to_network_format(output_file)
        logger.info(f"Knowledge graph exported to {output_file}")
    
    def persist(self) -> None:
        """
        Persist the GraphRAG system state.
        """
        if not self.persist_path:
            logger.warning("Cannot persist GraphRAG: No persistence path specified")
            return
        
        # Persist knowledge graph
        self.kg_builder.persist_graph()
        
        # Persist configuration
        config = {
            "extraction_method": self.extraction_method,
            "identification_method": self.identification_method,
            "model_name": self.model_name,
            "weaviate_url": self.weaviate_url,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        config_file = os.path.join(self.persist_path, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GraphRAG system persisted to {self.persist_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the GraphRAG system state.
        
        Args:
            path: Path to the directory containing the persisted state
        """
        load_path = path or self.persist_path
        
        if not load_path:
            logger.warning("Cannot load GraphRAG: No path specified")
            return
        
        if not os.path.isdir(load_path):
            logger.warning(f"Cannot load GraphRAG: Path {load_path} is not a directory")
            return
        
        # Load configuration
        config_file = os.path.join(load_path, "config.json")
        if os.path.isfile(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update configuration
            self.extraction_method = config.get("extraction_method", self.extraction_method)
            self.identification_method = config.get("identification_method", self.identification_method)
            self.model_name = config.get("model_name", self.model_name)
            self.weaviate_url = config.get("weaviate_url", self.weaviate_url)
            self.chunk_size = config.get("chunk_size", self.chunk_size)
            self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        
        # Load knowledge graph
        kg_path = os.path.join(load_path, "knowledge_graph")
        if os.path.isdir(kg_path):
            self.kg_builder.load_graph(kg_path)
        
        # Update persist path if different
        if path and path != self.persist_path:
            self.persist_path = path
        
        logger.info(f"GraphRAG system loaded from {load_path}")
