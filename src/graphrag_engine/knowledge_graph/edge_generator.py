"""
Edge Generator Module for Knowledge Graph Builder

This module is responsible for creating and managing edges (relationships)
between nodes in the knowledge graph.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EdgeGenerator:
    """
    Creates and manages edges between nodes in the knowledge graph.
    
    The EdgeGenerator is responsible for:
    - Creating edges from identified relationships
    - Validating and filtering edges
    - Assigning weights and properties to edges
    - Managing edge metadata
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        allow_bidirectional: bool = True,
        allow_self_loops: bool = False,
        invert_relationship_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize an EdgeGenerator.
        
        Args:
            confidence_threshold: Minimum confidence score for edge creation (0.0-1.0)
            allow_bidirectional: Whether to allow bidirectional relationships
            allow_self_loops: Whether to allow relationships from a node to itself
            invert_relationship_map: Mapping from relationship types to their inverses
        """
        self.edges = {}  # Maps edge_id to edge data
        self.node_edges = {}  # Maps node_id to set of connected edge_ids
        self.confidence_threshold = confidence_threshold
        self.allow_bidirectional = allow_bidirectional
        self.allow_self_loops = allow_self_loops
        
        # Default invert relationship map if not provided
        self.invert_relationship_map = invert_relationship_map or {
            "WORKS_FOR": "EMPLOYS",
            "EMPLOYS": "WORKS_FOR",
            "LOCATED_IN": "CONTAINS",
            "CONTAINS": "LOCATED_IN",
            "PART_OF": "HAS_PART",
            "HAS_PART": "PART_OF",
            "CREATOR_OF": "CREATED_BY",
            "CREATED_BY": "CREATOR_OF",
            "OWNER_OF": "OWNED_BY",
            "OWNED_BY": "OWNER_OF",
            "PARENT_OF": "CHILD_OF",
            "CHILD_OF": "PARENT_OF"
        }
        
        logger.info(f"EdgeGenerator initialized with confidence threshold {confidence_threshold}")
    
    def create_edges_from_relationships(
        self,
        relationships: List[Dict[str, Any]],
        node_registry: Any,
        source_document: Optional[str] = None,
        batch_processing: bool = True
    ) -> List[str]:
        """
        Create edges from identified relationships.
        
        Args:
            relationships: List of relationships between entities
            node_registry: NodeRegistry object containing the nodes
            source_document: Source document from which relationships were extracted
            batch_processing: Whether to use batch processing for efficiency
            
        Returns:
            List of edge IDs for the created edges
        """
        edge_ids = []
        
        if batch_processing:
            for rel in tqdm(relationships, desc="Creating edges"):
                edge_id = self.create_edge_from_relationship(rel, node_registry, source_document)
                if edge_id:
                    edge_ids.append(edge_id)
        else:
            for rel in relationships:
                edge_id = self.create_edge_from_relationship(rel, node_registry, source_document)
                if edge_id:
                    edge_ids.append(edge_id)
        
        return edge_ids
    
    def create_edge_from_relationship(
        self,
        relationship: Dict[str, Any],
        node_registry: Any,
        source_document: Optional[str] = None
    ) -> Optional[str]:
        """
        Create an edge from a single relationship.
        
        Args:
            relationship: Relationship information
            node_registry: NodeRegistry object containing the nodes
            source_document: Source document from which relationship was extracted
            
        Returns:
            Edge ID of the created edge, or None if edge could not be created
        """
        # Extract relationship information
        source_text = relationship.get("source")
        target_text = relationship.get("target")
        rel_type = relationship.get("relationship")
        confidence = relationship.get("confidence", 1.0)
        bidirectional = relationship.get("bidirectional", False)
        
        # Validate relationship information
        if not source_text or not target_text or not rel_type:
            logger.warning("Missing required fields in relationship")
            return None
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Relationship confidence {confidence} below threshold {self.confidence_threshold}")
            return None
        
        # Get node IDs from entity texts
        source_node = node_registry.get_node_by_entity(source_text)
        target_node = node_registry.get_node_by_entity(target_text)
        
        if not source_node or not target_node:
            logger.warning(f"Source or target node not found for relationship {source_text} -> {target_text}")
            return None
        
        source_id = source_node.get("id")
        target_id = target_node.get("id")
        
        # Check for self-loops
        if source_id == target_id and not self.allow_self_loops:
            logger.debug(f"Self-loop not allowed: {source_text} -> {target_text}")
            return None
        
        # Create the edge
        edge_id = self._generate_edge_id(source_id, target_id, rel_type)
        
        # Check if this edge already exists
        if edge_id in self.edges:
            # Update existing edge with additional information
            self._update_existing_edge(edge_id, relationship, source_document)
        else:
            # Create a new edge
            self._create_new_edge(edge_id, source_id, target_id, rel_type, relationship, source_document)
        
        # Create inverse edge if bidirectional
        if bidirectional and self.allow_bidirectional:
            inverse_rel_type = self._get_inverse_relationship(rel_type)
            if inverse_rel_type:
                inverse_edge_id = self._generate_edge_id(target_id, source_id, inverse_rel_type)
                
                # Clone relationship with inverse parameters
                inverse_relationship = relationship.copy()
                inverse_relationship["source"] = target_text
                inverse_relationship["target"] = source_text
                inverse_relationship["relationship"] = inverse_rel_type
                
                if inverse_edge_id in self.edges:
                    self._update_existing_edge(inverse_edge_id, inverse_relationship, source_document)
                else:
                    self._create_new_edge(inverse_edge_id, target_id, source_id, inverse_rel_type, 
                                         inverse_relationship, source_document)
        
        return edge_id
    
    def _generate_edge_id(self, source_id: str, target_id: str, rel_type: str) -> str:
        """Generate a unique edge ID based on source, target, and relationship type."""
        # Create a string that combines source, target, and relationship type
        edge_string = f"{source_id}_{target_id}_{rel_type}"
        
        # Generate a hash as the edge ID
        hash_obj = hashlib.md5(edge_string.encode())
        edge_id = hash_obj.hexdigest()
        
        return edge_id
    
    def _update_existing_edge(
        self,
        edge_id: str,
        relationship: Dict[str, Any],
        source_document: Optional[str] = None
    ) -> None:
        """Update an existing edge with additional information."""
        edge = self.edges[edge_id]
        
        # Increment occurrence count
        edge["occurrences"] += 1
        
        # Update confidence (take the maximum)
        confidence = relationship.get("confidence", 0.0)
        if confidence > edge.get("confidence", 0.0):
            edge["confidence"] = confidence
        
        # Add source document if not already present
        if source_document and source_document not in edge.get("documents", []):
            if "documents" not in edge:
                edge["documents"] = []
            edge["documents"].append(source_document)
        
        # Update metadata if provided
        if "metadata" in relationship:
            if "metadata" not in edge:
                edge["metadata"] = {}
            for key, value in relationship.get("metadata", {}).items():
                if key not in edge["metadata"]:
                    edge["metadata"][key] = value
                elif isinstance(edge["metadata"][key], list):
                    if isinstance(value, list):
                        for v in value:
                            if v not in edge["metadata"][key]:
                                edge["metadata"][key].append(v)
                    elif value not in edge["metadata"][key]:
                        edge["metadata"][key].append(value)
                else:
                    if isinstance(value, list):
                        edge["metadata"][key] = [edge["metadata"][key]] + value
                    else:
                        edge["metadata"][key] = [edge["metadata"][key], value]
    
    def _create_new_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        rel_type: str,
        relationship: Dict[str, Any],
        source_document: Optional[str] = None
    ) -> None:
        """Create a new edge from a relationship."""
        # Create the edge object
        edge = {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "type": rel_type,
            "confidence": relationship.get("confidence", 1.0),
            "occurrences": 1,
            "documents": [source_document] if source_document else [],
            "bidirectional": relationship.get("bidirectional", False)
        }
        
        # Add metadata if available
        if "metadata" in relationship:
            edge["metadata"] = relationship["metadata"]
        
        # Store the edge in the registry
        self.edges[edge_id] = edge
        
        # Update node_edges mapping
        if source_id not in self.node_edges:
            self.node_edges[source_id] = set()
        self.node_edges[source_id].add(edge_id)
        
        if target_id not in self.node_edges:
            self.node_edges[target_id] = set()
        self.node_edges[target_id].add(edge_id)
    
    def _get_inverse_relationship(self, rel_type: str) -> Optional[str]:
        """Get the inverse relationship type for a given relationship type."""
        return self.invert_relationship_map.get(rel_type)
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by its ID."""
        return self.edges.get(edge_id)
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges in the registry."""
        return list(self.edges.values())
    
    def get_edges_by_type(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get all edges of a specific relationship type."""
        return [edge for edge in self.edges.values() if edge["type"] == rel_type]
    
    def get_edges_by_document(self, document: str) -> List[Dict[str, Any]]:
        """Get all edges that appear in a specific document."""
        return [edge for edge in self.edges.values() if document in edge.get("documents", [])]
    
    def get_edges_by_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all edges connected to a specific node."""
        edge_ids = self.node_edges.get(node_id, set())
        return [self.edges[edge_id] for edge_id in edge_ids if edge_id in self.edges]
    
    def get_edges_between_nodes(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Get all edges between two specific nodes."""
        return [
            edge for edge in self.edges.values() 
            if (edge["source_id"] == source_id and edge["target_id"] == target_id) or
               (edge["source_id"] == target_id and edge["target_id"] == source_id)
        ]
    
    def remove_edge(self, edge_id: str) -> bool:
        """
        Remove an edge from the registry.
        
        Args:
            edge_id: ID of the edge to remove
            
        Returns:
            True if edge was successfully removed, False otherwise
        """
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        
        # Remove edge from node_edges mappings
        if source_id in self.node_edges:
            self.node_edges[source_id].discard(edge_id)
        
        if target_id in self.node_edges:
            self.node_edges[target_id].discard(edge_id)
        
        # Remove the edge
        del self.edges[edge_id]
        
        return True
    
    def filter_edges_by_confidence(self, min_confidence: float) -> int:
        """
        Filter out edges with confidence below the specified threshold.
        
        Args:
            min_confidence: Minimum confidence score for edges to keep
            
        Returns:
            Number of edges removed
        """
        removed_count = 0
        edge_ids_to_remove = []
        
        for edge_id, edge in self.edges.items():
            if edge.get("confidence", 0.0) < min_confidence:
                edge_ids_to_remove.append(edge_id)
        
        for edge_id in edge_ids_to_remove:
            if self.remove_edge(edge_id):
                removed_count += 1
        
        return removed_count
    
    def merge_parallel_edges(self) -> int:
        """
        Merge parallel edges (multiple edges of the same type between the same nodes).
        
        Returns:
            Number of edges merged
        """
        merged_count = 0
        edges_by_endpoints = {}  # Maps (source_id, target_id, type) to list of edge_ids
        
        # Group edges by endpoints and type
        for edge_id, edge in self.edges.items():
            key = (edge["source_id"], edge["target_id"], edge["type"])
            if key not in edges_by_endpoints:
                edges_by_endpoints[key] = []
            edges_by_endpoints[key].append(edge_id)
        
        # Merge parallel edges
        for key, edge_ids in edges_by_endpoints.items():
            if len(edge_ids) <= 1:
                continue
            
            # Keep the edge with the highest confidence
            primary_edge_id = max(edge_ids, key=lambda eid: self.edges[eid].get("confidence", 0.0))
            secondary_edge_ids = [eid for eid in edge_ids if eid != primary_edge_id]
            
            # Merge information from secondary edges into primary edge
            primary_edge = self.edges[primary_edge_id]
            
            for sec_edge_id in secondary_edge_ids:
                sec_edge = self.edges[sec_edge_id]
                
                # Update occurrences
                primary_edge["occurrences"] += sec_edge.get("occurrences", 1)
                
                # Merge documents
                if "documents" in sec_edge:
                    if "documents" not in primary_edge:
                        primary_edge["documents"] = []
                    for doc in sec_edge["documents"]:
                        if doc not in primary_edge["documents"]:
                            primary_edge["documents"].append(doc)
                
                # Merge metadata
                if "metadata" in sec_edge:
                    if "metadata" not in primary_edge:
                        primary_edge["metadata"] = {}
                    for key, value in sec_edge.get("metadata", {}).items():
                        if key not in primary_edge["metadata"]:
                            primary_edge["metadata"][key] = value
                        elif isinstance(primary_edge["metadata"][key], list):
                            if isinstance(value, list):
                                for v in value:
                                    if v not in primary_edge["metadata"][key]:
                                        primary_edge["metadata"][key].append(v)
                            elif value not in primary_edge["metadata"][key]:
                                primary_edge["metadata"][key].append(value)
                        else:
                            if isinstance(value, list):
                                primary_edge["metadata"][key] = [primary_edge["metadata"][key]] + value
                            else:
                                primary_edge["metadata"][key] = [primary_edge["metadata"][key], value]
                
                # Remove the secondary edge
                self.remove_edge(sec_edge_id)
                merged_count += 1
        
        return merged_count
    
    def export_to_json(self, output_file: str) -> None:
        """
        Export the edge registry to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        # Convert sets to lists for JSON serialization
        node_edges_json = {node_id: list(edge_ids) for node_id, edge_ids in self.node_edges.items()}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "edges": self.edges,
                "node_edges": node_edges_json
            }, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, input_file: str) -> None:
        """
        Import the edge registry from a JSON file.
        
        Args:
            input_file: Path to the input JSON file
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.edges = data.get("edges", {})
        
        # Convert lists back to sets
        node_edges_data = data.get("node_edges", {})
        self.node_edges = {node_id: set(edge_ids) for node_id, edge_ids in node_edges_data.items()}
        
        logger.info(f"Imported {len(self.edges)} edges from {input_file}")
