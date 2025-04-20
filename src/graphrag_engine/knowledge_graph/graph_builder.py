"""
Knowledge Graph Builder Module for GraphRAG Engine

This module integrates all components of the knowledge graph builder:
- Entity Extraction
- Relationship Identification
- Node Registry
- Edge Generation

It provides a unified interface for building and managing the knowledge graph.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pathlib import Path
import hashlib
from tqdm import tqdm

from .entity_extractor import EntityExtractor
from .relationship_identifier import RelationshipIdentifier
from .node_registry import NodeRegistry
from .edge_generator import EdgeGenerator

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Main class for building and managing knowledge graphs.
    
    The KnowledgeGraphBuilder integrates all components of the knowledge graph
    building process and provides methods for creating, querying, and managing
    the knowledge graph.
    """
    
    def __init__(
        self,
        extraction_method: str = "ollama",
        identification_method: str = "ollama",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        similarity_threshold: float = 0.85,
        confidence_threshold: float = 0.5,
        case_sensitive: bool = False,
        persist_path: Optional[str] = None
    ):
        """
        Initialize a KnowledgeGraphBuilder.
        
        Args:
            extraction_method: Method for entity extraction ('ollama', 'openai', 'anthropic', 'gemini', 'rule_based')
            identification_method: Method for relationship identification ('ollama', 'openai', 'anthropic', 'gemini', 'rule_based', 'proximity', 'cooccurrence')
            model_name: Name of the model to use (default: 'llama2' for Ollama)
            api_key: API key for external APIs (required for OpenAI, Anthropic, Gemini)
            entity_types: List of entity types to extract
            relationship_types: List of relationship types to identify
            similarity_threshold: Threshold for entity similarity in node registry
            confidence_threshold: Threshold for relationship confidence in edge generator
            case_sensitive: Whether entity matching should be case-sensitive
            persist_path: Directory path for persisting the knowledge graph
        """
        self.extraction_method = extraction_method
        self.identification_method = identification_method
        self.model_name = model_name
        self.api_key = api_key
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.case_sensitive = case_sensitive
        self.persist_path = persist_path
        
        # Initialize components
        self.entity_extractor = EntityExtractor(
            extraction_method=extraction_method,
            model_name=model_name,
            api_key=api_key,
            entity_types=entity_types
        )
        
        self.relationship_identifier = RelationshipIdentifier(
            identification_method=identification_method,
            model_name=model_name,
            api_key=api_key,
            relationship_types=relationship_types,
            threshold=confidence_threshold
        )
        
        self.node_registry = NodeRegistry(
            similarity_threshold=similarity_threshold,
            case_sensitive=case_sensitive
        )
        
        self.edge_generator = EdgeGenerator(
            confidence_threshold=confidence_threshold
        )
        
        # Create persistence directory if specified
        if self.persist_path:
            os.makedirs(self.persist_path, exist_ok=True)
            
        logger.info(f"KnowledgeGraphBuilder initialized with {extraction_method} extraction and {identification_method} identification")
    
    def process_document(
        self,
        document_text: str,
        document_id: str,
        batch_size: int = 10,
        extract_entities: bool = True,
        identify_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Process a document to extract entities and relationships.
        
        Args:
            document_text: Text content of the document
            document_id: Unique identifier for the document
            batch_size: Batch size for processing entity pairs
            extract_entities: Whether to extract entities
            identify_relationships: Whether to identify relationships
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "document_id": document_id,
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_identified": 0,
            "edges_created": 0
        }
        
        # Extract entities
        if extract_entities:
            logger.info(f"Extracting entities from document {document_id}")
            entities = self.entity_extractor.extract_entities(document_text)
            result["entities_extracted"] = len(entities)
            
            # Register entities as nodes
            logger.info(f"Registering {len(entities)} entities as nodes")
            node_ids = self.node_registry.register_entities(entities, document_id)
            result["nodes_created"] = len(node_ids)
            
            # Identify relationships if there are multiple entities
            if identify_relationships and len(entities) > 1:
                logger.info(f"Identifying relationships between entities")
                relationships = self.relationship_identifier.identify_relationships(
                    entities, document_text, batch_size
                )
                result["relationships_identified"] = len(relationships)
                
                # Create edges from relationships
                logger.info(f"Creating edges from {len(relationships)} relationships")
                edge_ids = self.edge_generator.create_edges_from_relationships(
                    relationships, self.node_registry, document_id
                )
                result["edges_created"] = len(edge_ids)
        
        # Persist the knowledge graph if path is specified
        if self.persist_path:
            self.persist_graph()
        
        return result
    
    def process_documents(
        self,
        documents: Dict[str, str],
        batch_size: int = 10,
        extract_entities: bool = True,
        identify_relationships: bool = True,
        cross_document_relationships: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple documents to extract entities and relationships.
        
        Args:
            documents: Dictionary mapping document IDs to text content
            batch_size: Batch size for processing entity pairs
            extract_entities: Whether to extract entities
            identify_relationships: Whether to identify relationships
            cross_document_relationships: Whether to identify relationships across documents
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "documents_processed": len(documents),
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_identified": 0,
            "edges_created": 0,
            "cross_document_relationships": 0,
            "cross_document_edges": 0
        }
        
        # Process each document individually
        all_entities = {}  # Maps document_id to list of entities
        
        for doc_id, doc_text in tqdm(documents.items(), desc="Processing documents"):
            doc_result = self.process_document(
                doc_text, doc_id, batch_size, 
                extract_entities, identify_relationships
            )
            
            result["entities_extracted"] += doc_result["entities_extracted"]
            result["nodes_created"] += doc_result["nodes_created"]
            result["relationships_identified"] += doc_result["relationships_identified"]
            result["edges_created"] += doc_result["edges_created"]
            
            # Store extracted entities for cross-document processing
            if cross_document_relationships and extract_entities:
                all_entities[doc_id] = self.entity_extractor.extract_entities(doc_text)
        
        # Process cross-document relationships if requested
        if cross_document_relationships and len(documents) > 1 and extract_entities:
            logger.info("Processing cross-document relationships")
            
            # Collect all entities across documents
            all_entities_list = []
            for doc_id, entities in all_entities.items():
                for entity in entities:
                    entity["document_id"] = doc_id
                all_entities_list.extend(entities)
            
            # Identify relationships across all entities
            cross_relationships = self.relationship_identifier.identify_relationships(
                all_entities_list, None, batch_size
            )
            
            # Filter to keep only cross-document relationships
            cross_doc_relationships = []
            for rel in cross_relationships:
                source_doc = next((e["document_id"] for e in all_entities_list if e["text"] == rel["source"]), None)
                target_doc = next((e["document_id"] for e in all_entities_list if e["text"] == rel["target"]), None)
                
                if source_doc and target_doc and source_doc != target_doc:
                    rel["source_document"] = source_doc
                    rel["target_document"] = target_doc
                    cross_doc_relationships.append(rel)
            
            result["cross_document_relationships"] = len(cross_doc_relationships)
            
            # Create edges from cross-document relationships
            if cross_doc_relationships:
                logger.info(f"Creating edges from {len(cross_doc_relationships)} cross-document relationships")
                cross_edge_ids = self.edge_generator.create_edges_from_relationships(
                    cross_doc_relationships, self.node_registry, None
                )
                result["cross_document_edges"] = len(cross_edge_ids)
        
        # Persist the knowledge graph if path is specified
        if self.persist_path:
            self.persist_graph()
        
        return result
    
    def add_entity(
        self,
        entity_text: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add a single entity to the knowledge graph.
        
        Args:
            entity_text: Text of the entity
            entity_type: Type of the entity
            metadata: Additional metadata for the entity
            document_id: ID of the source document
            
        Returns:
            Node ID of the added entity
        """
        entity = {
            "text": entity_text,
            "type": entity_type,
            "metadata": metadata or {}
        }
        
        node_id = self.node_registry.register_entity(entity, document_id)
        
        # Persist the knowledge graph if path is specified
        if self.persist_path:
            self.persist_graph()
        
        return node_id
    
    def add_relationship(
        self,
        source_text: str,
        target_text: str,
        relationship_type: str,
        confidence: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Add a relationship between two entities to the knowledge graph.
        
        Args:
            source_text: Text of the source entity
            target_text: Text of the target entity
            relationship_type: Type of the relationship
            confidence: Confidence score for the relationship (0.0-1.0)
            bidirectional: Whether the relationship is bidirectional
            metadata: Additional metadata for the relationship
            document_id: ID of the source document
            
        Returns:
            Edge ID of the added relationship, or None if the relationship could not be added
        """
        relationship = {
            "source": source_text,
            "target": target_text,
            "relationship": relationship_type,
            "confidence": confidence,
            "bidirectional": bidirectional,
            "metadata": metadata or {}
        }
        
        edge_id = self.edge_generator.create_edge_from_relationship(
            relationship, self.node_registry, document_id
        )
        
        # Persist the knowledge graph if path is specified
        if self.persist_path:
            self.persist_graph()
        
        return edge_id
    
    def query_entities(
        self,
        entity_text: Optional[str] = None,
        entity_type: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query entities in the knowledge graph.
        
        Args:
            entity_text: Text pattern to match (case-insensitive substring match)
            entity_type: Type of entities to retrieve
            document_id: ID of the source document
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities
        """
        nodes = self.node_registry.get_all_nodes()
        
        # Apply filters
        if entity_type:
            nodes = [node for node in nodes if node["type"] == entity_type]
        
        if document_id:
            nodes = [node for node in nodes if document_id in node.get("documents", [])]
        
        if entity_text:
            if self.case_sensitive:
                nodes = [node for node in nodes if entity_text in node["text"]]
            else:
                entity_text_lower = entity_text.lower()
                nodes = [node for node in nodes if entity_text_lower in node["text"].lower()]
        
        # Apply limit
        if limit > 0:
            nodes = nodes[:limit]
        
        return nodes
    
    def query_relationships(
        self,
        source_text: Optional[str] = None,
        target_text: Optional[str] = None,
        relationship_type: Optional[str] = None,
        document_id: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query relationships in the knowledge graph.
        
        Args:
            source_text: Text pattern to match for source entity
            target_text: Text pattern to match for target entity
            relationship_type: Type of relationships to retrieve
            document_id: ID of the source document
            min_confidence: Minimum confidence score for relationships
            limit: Maximum number of results to return
            
        Returns:
            List of matching relationships
        """
        edges = self.edge_generator.get_all_edges()
        
        # Apply filters
        if relationship_type:
            edges = [edge for edge in edges if edge["type"] == relationship_type]
        
        if document_id:
            edges = [edge for edge in edges if document_id in edge.get("documents", [])]
        
        if min_confidence > 0:
            edges = [edge for edge in edges if edge.get("confidence", 0.0) >= min_confidence]
        
        # Filter by source and target text if provided
        if source_text or target_text:
            filtered_edges = []
            for edge in edges:
                source_node = self.node_registry.get_node(edge["source_id"])
                target_node = self.node_registry.get_node(edge["target_id"])
                
                if not source_node or not target_node:
                    continue
                
                source_match = True
                target_match = True
                
                if source_text:
                    if self.case_sensitive:
                        source_match = source_text in source_node["text"]
                    else:
                        source_match = source_text.lower() in source_node["text"].lower()
                
                if target_text:
                    if self.case_sensitive:
                        target_match = target_text in target_node["text"]
                    else:
                        target_match = target_text.lower() in target_node["text"].lower()
                
                if source_match and target_match:
                    # Add human-readable source and target text to the edge for convenience
                    edge_with_text = edge.copy()
                    edge_with_text["source_text"] = source_node["text"]
                    edge_with_text["source_type"] = source_node["type"]
                    edge_with_text["target_text"] = target_node["text"]
                    edge_with_text["target_type"] = target_node["type"]
                    filtered_edges.append(edge_with_text)
            
            edges = filtered_edges
        
        # Apply limit
        if limit > 0:
            edges = edges[:limit]
        
        return edges
    
    def get_entity_relationships(
        self,
        entity_text: str,
        include_incoming: bool = True,
        include_outgoing: bool = True,
        relationship_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships for a specific entity.
        
        Args:
            entity_text: Text of the entity
            include_incoming: Whether to include incoming relationships
            include_outgoing: Whether to include outgoing relationships
            relationship_types: Types of relationships to include
            min_confidence: Minimum confidence score for relationships
            
        Returns:
            Dictionary with "incoming" and "outgoing" relationships
        """
        result = {
            "incoming": [],
            "outgoing": []
        }
        
        # Find the node for the entity
        entity_node = self.node_registry.get_node_by_entity(entity_text)
        if not entity_node:
            return result
        
        node_id = entity_node["id"]
        
        # Get all edges connected to this node
        connected_edges = self.edge_generator.get_edges_by_node(node_id)
        
        # Filter and categorize edges
        for edge in connected_edges:
            # Skip if confidence is too low
            if edge.get("confidence", 0.0) < min_confidence:
                continue
            
            # Skip if relationship type doesn't match
            if relationship_types and edge["type"] not in relationship_types:
                continue
            
            # Get source and target nodes
            source_node = self.node_registry.get_node(edge["source_id"])
            target_node = self.node_registry.get_node(edge["target_id"])
            
            if not source_node or not target_node:
                continue
            
            # Create an enhanced edge with node text
            enhanced_edge = edge.copy()
            enhanced_edge["source_text"] = source_node["text"]
            enhanced_edge["source_type"] = source_node["type"]
            enhanced_edge["target_text"] = target_node["text"]
            enhanced_edge["target_type"] = target_node["type"]
            
            # Categorize as incoming or outgoing
            if edge["source_id"] == node_id and include_outgoing:
                result["outgoing"].append(enhanced_edge)
            elif edge["target_id"] == node_id and include_incoming:
                result["incoming"].append(enhanced_edge)
        
        return result
    
    def get_paths_between_entities(
        self,
        source_text: str,
        target_text: str,
        max_depth: int = 3,
        relationship_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities in the knowledge graph.
        
        Args:
            source_text: Text of the source entity
            target_text: Text of the target entity
            max_depth: Maximum path length to consider
            relationship_types: Types of relationships to include
            min_confidence: Minimum confidence score for relationships
            
        Returns:
            List of paths, where each path is a list of edges
        """
        # Find source and target nodes
        source_node = self.node_registry.get_node_by_entity(source_text)
        target_node = self.node_registry.get_node_by_entity(target_text)
        
        if not source_node or not target_node:
            return []
        
        source_id = source_node["id"]
        target_id = target_node["id"]
        
        # Use breadth-first search to find paths
        paths = self._find_paths_bfs(
            source_id, target_id, max_depth, relationship_types, min_confidence
        )
        
        # Enhance paths with node text
        enhanced_paths = []
        for path in paths:
            enhanced_path = []
            for edge in path:
                source_node = self.node_registry.get_node(edge["source_id"])
                target_node = self.node_registry.get_node(edge["target_id"])
                
                enhanced_edge = edge.copy()
                enhanced_edge["source_text"] = source_node["text"]
                enhanced_edge["source_type"] = source_node["type"]
                enhanced_edge["target_text"] = target_node["text"]
                enhanced_edge["target_type"] = target_node["type"]
                
                enhanced_path.append(enhanced_edge)
            
            enhanced_paths.append(enhanced_path)
        
        return enhanced_paths
    
    def _find_paths_bfs(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
        relationship_types: Optional[List[str]],
        min_confidence: float
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two nodes using breadth-first search.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            max_depth: Maximum path length
            relationship_types: Types of relationships to include
            min_confidence: Minimum confidence score for relationships
            
        Returns:
            List of paths, where each path is a list of edges
        """
        if source_id == target_id:
            return []
        
        # Initialize BFS queue with source node
        queue = [(source_id, [])]  # (node_id, path_so_far)
        visited = set([source_id])
        paths = []
        
        while queue:
            current_id, path = queue.pop(0)
            
            # Check if we've reached the target
            if current_id == target_id:
                paths.append(path)
                continue
            
            # Check if we've reached the maximum depth
            if len(path) >= max_depth:
                continue
            
            # Get all edges connected to the current node
            connected_edges = self.edge_generator.get_edges_by_node(current_id)
            
            for edge in connected_edges:
                # Skip if confidence is too low
                if edge.get("confidence", 0.0) < min_confidence:
                    continue
                
                # Skip if relationship type doesn't match
                if relationship_types and edge["type"] not in relationship_types:
                    continue
                
                # Determine the next node
                if edge["source_id"] == current_id:
                    next_id = edge["target_id"]
                else:
                    next_id = edge["source_id"]
                
                # Skip if we've already visited this node
                if next_id in visited:
                    continue
                
                # Add the next node to the queue
                visited.add(next_id)
                queue.append((next_id, path + [edge]))
        
        return paths
    
    def graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        nodes = self.node_registry.get_all_nodes()
        edges = self.edge_generator.get_all_edges()
        
        # Count nodes by type
        node_types = {}
        for node in nodes:
            node_type = node["type"]
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        
        # Count edges by type
        edge_types = {}
        for edge in edges:
            edge_type = edge["type"]
            if edge_type not in edge_types:
                edge_types[edge_type] = 0
            edge_types[edge_type] += 1
        
        # Count documents
        documents = set()
        for node in nodes:
            for doc in node.get("documents", []):
                if doc:
                    documents.add(doc)
        
        # Compute average degree
        if nodes:
            total_degree = sum(len(self.edge_generator.get_edges_by_node(node["id"])) for node in nodes)
            avg_degree = total_degree / len(nodes)
        else:
            avg_degree = 0
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_documents": len(documents),
            "node_types": node_types,
            "edge_types": edge_types,
            "average_degree": avg_degree
        }
    
    def persist_graph(self) -> None:
        """
        Persist the knowledge graph to disk.
        """
        if not self.persist_path:
            logger.warning("Cannot persist graph: No persistence path specified")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)
        
        # Save node registry
        nodes_file = os.path.join(self.persist_path, "nodes.json")
        self.node_registry.export_to_json(nodes_file)
        
        # Save edge generator
        edges_file = os.path.join(self.persist_path, "edges.json")
        self.edge_generator.export_to_json(edges_file)
        
        # Save configuration
        config = {
            "extraction_method": self.extraction_method,
            "identification_method": self.identification_method,
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "confidence_threshold": self.confidence_threshold,
            "case_sensitive": self.case_sensitive
        }
        
        config_file = os.path.join(self.persist_path, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Knowledge graph persisted to {self.persist_path}")
    
    def load_graph(self, path: Optional[str] = None) -> None:
        """
        Load the knowledge graph from disk.
        
        Args:
            path: Path to the directory containing the persisted graph
        """
        load_path = path or self.persist_path
        
        if not load_path:
            logger.warning("Cannot load graph: No path specified")
            return
        
        if not os.path.isdir(load_path):
            logger.warning(f"Cannot load graph: Path {load_path} is not a directory")
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
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
            self.confidence_threshold = config.get("confidence_threshold", self.confidence_threshold)
            self.case_sensitive = config.get("case_sensitive", self.case_sensitive)
        
        # Load node registry
        nodes_file = os.path.join(load_path, "nodes.json")
        if os.path.isfile(nodes_file):
            self.node_registry.import_from_json(nodes_file)
        
        # Load edge generator
        edges_file = os.path.join(load_path, "edges.json")
        if os.path.isfile(edges_file):
            self.edge_generator.import_from_json(edges_file)
        
        # Update persist path if different
        if path and path != self.persist_path:
            self.persist_path = path
        
        logger.info(f"Knowledge graph loaded from {load_path}")
    
    def export_to_network_format(self, output_file: str) -> None:
        """
        Export the knowledge graph to a network format for visualization.
        
        Args:
            output_file: Path to the output JSON file
        """
        nodes = self.node_registry.get_all_nodes()
        edges = self.edge_generator.get_all_edges()
        
        # Create export-friendly nodes
        export_nodes = []
        for node in nodes:
            export_node = {
                "id": node["id"],
                "label": node["text"],
                "type": node["type"],
                "group": node["type"],  # For visualization grouping
                "occurrences": node.get("occurrences", 1),
                "documents": node.get("documents", [])
            }
            export_nodes.append(export_node)
        
        # Create export-friendly edges
        export_edges = []
        for edge in edges:
            source_node = self.node_registry.get_node(edge["source_id"])
            target_node = self.node_registry.get_node(edge["target_id"])
            
            if not source_node or not target_node:
                continue
            
            export_edge = {
                "id": edge["id"],
                "source": edge["source_id"],
                "target": edge["target_id"],
                "label": edge["type"],
                "type": edge["type"],
                "confidence": edge.get("confidence", 1.0),
                "sourceNode": source_node["text"],
                "targetNode": target_node["text"],
                "bidirectional": edge.get("bidirectional", False)
            }
            export_edges.append(export_edge)
        
        # Create export object
        export_data = {
            "nodes": export_nodes,
            "edges": export_edges
        }
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Knowledge graph exported to {output_file}")
    
    def clear_graph(self) -> None:
        """
        Clear the knowledge graph.
        """
        self.node_registry = NodeRegistry(
            similarity_threshold=self.similarity_threshold,
            case_sensitive=self.case_sensitive
        )
        
        self.edge_generator = EdgeGenerator(
            confidence_threshold=self.confidence_threshold
        )
        
        logger.info("Knowledge graph cleared")
