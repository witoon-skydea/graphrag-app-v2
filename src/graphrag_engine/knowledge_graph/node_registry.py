"""
Node Registry Module for Knowledge Graph Builder

This module is responsible for managing and registering entities as nodes
in the knowledge graph, handling entity resolution and deduplication.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import hashlib
from tqdm import tqdm

import numpy as np

logger = logging.getLogger(__name__)

class NodeRegistry:
    """
    Manages and registers entities as nodes in the knowledge graph.
    
    The NodeRegistry is responsible for:
    - Maintaining a registry of unique nodes/entities
    - Entity resolution and deduplication
    - Assigning unique identifiers to nodes
    - Tracking node metadata and properties
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        case_sensitive: bool = False,
        entity_embeddings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a NodeRegistry.
        
        Args:
            similarity_threshold: Threshold for considering two entities as the same (0.0-1.0)
            case_sensitive: Whether entity matching should be case-sensitive
            entity_embeddings: Optional dictionary of entity embeddings for similarity matching
        """
        self.nodes = {}  # Maps node_id to node data
        self.entity_to_node_id = {}  # Maps entity text to node_id
        self.similarity_threshold = similarity_threshold
        self.case_sensitive = case_sensitive
        self.entity_embeddings = entity_embeddings or {}
        
        logger.info(f"NodeRegistry initialized with similarity threshold {similarity_threshold}")
    
    def register_entities(
        self, 
        entities: List[Dict[str, Any]], 
        source_document: Optional[str] = None,
        batch_processing: bool = True
    ) -> List[str]:
        """
        Register multiple entities in the node registry.
        
        Args:
            entities: List of entities to register
            source_document: Source document from which entities were extracted
            batch_processing: Whether to use batch processing for efficiency
            
        Returns:
            List of node IDs for the registered entities
        """
        node_ids = []
        
        if batch_processing:
            for entity in tqdm(entities, desc="Registering entities"):
                node_id = self.register_entity(entity, source_document)
                node_ids.append(node_id)
        else:
            for entity in entities:
                node_id = self.register_entity(entity, source_document)
                node_ids.append(node_id)
        
        return node_ids
    
    def register_entity(
        self, 
        entity: Dict[str, Any], 
        source_document: Optional[str] = None
    ) -> str:
        """
        Register a single entity in the node registry.
        
        Args:
            entity: Entity information to register
            source_document: Source document from which entity was extracted
            
        Returns:
            Node ID of the registered entity
        """
        entity_text = entity["text"]
        entity_type = entity["type"]
        
        # Normalize entity text if case-insensitive
        lookup_text = entity_text if self.case_sensitive else entity_text.lower()
        
        # Check if this entity already exists
        existing_node_id = self._find_matching_entity(lookup_text, entity_type)
        
        if existing_node_id:
            # Update existing node with additional information
            self._update_existing_node(existing_node_id, entity, source_document)
            return existing_node_id
        else:
            # Create a new node
            return self._create_new_node(entity, source_document)
    
    def _find_matching_entity(self, entity_text: str, entity_type: str) -> Optional[str]:
        """Find an existing entity that matches the given entity text and type."""
        # Check for exact matches
        if entity_text in self.entity_to_node_id:
            node_id = self.entity_to_node_id[entity_text]
            if self.nodes[node_id]["type"] == entity_type:
                return node_id
        
        # Check for semantic similarity matches if embeddings are available
        if self.entity_embeddings and entity_text in self.entity_embeddings:
            entity_embedding = self.entity_embeddings[entity_text]
            
            for other_text, other_node_id in self.entity_to_node_id.items():
                if other_text in self.entity_embeddings and self.nodes[other_node_id]["type"] == entity_type:
                    other_embedding = self.entity_embeddings[other_text]
                    similarity = self._calculate_similarity(entity_embedding, other_embedding)
                    
                    if similarity >= self.similarity_threshold:
                        return other_node_id
        
        # Check for string similarity
        for other_text, other_node_id in self.entity_to_node_id.items():
            if self.nodes[other_node_id]["type"] == entity_type:
                normalized_other = other_text if self.case_sensitive else other_text.lower()
                
                # Check if one is a substring of the other
                if normalized_other in entity_text or entity_text in normalized_other:
                    string_similarity = self._calculate_string_similarity(entity_text, normalized_other)
                    if string_similarity >= self.similarity_threshold:
                        return other_node_id
        
        return None
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_string_similarity(self, text1: str, text2: str) -> float:
        """Calculate string similarity using Jaccard similarity on character n-grams."""
        # Using character 3-grams for similarity
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        if len(text1) < 3 or len(text2) < 3:
            # For very short strings, use character overlap
            set1 = set(text1)
            set2 = set(text2)
        else:
            set1 = get_ngrams(text1)
            set2 = get_ngrams(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_node_id(self, entity: Dict[str, Any]) -> str:
        """Generate a unique node ID for an entity."""
        # Create a string that combines entity text and type
        entity_string = f"{entity['text']}_{entity['type']}"
        
        # Generate a hash as the node ID
        hash_obj = hashlib.md5(entity_string.encode())
        node_id = hash_obj.hexdigest()
        
        return node_id
    
    def _update_existing_node(
        self, 
        node_id: str, 
        entity: Dict[str, Any], 
        source_document: Optional[str] = None
    ) -> None:
        """Update an existing node with additional information."""
        node = self.nodes[node_id]
        
        # Increment occurrence count
        node["occurrences"] += 1
        
        # Add source document if not already present
        if source_document and source_document not in node.get("documents", []):
            if "documents" not in node:
                node["documents"] = []
            node["documents"].append(source_document)
        
        # Update metadata if provided
        if "metadata" in entity:
            if "metadata" not in node:
                node["metadata"] = {}
            for key, value in entity["metadata"].items():
                if key not in node["metadata"]:
                    node["metadata"][key] = value
                elif isinstance(node["metadata"][key], list):
                    if value not in node["metadata"][key]:
                        node["metadata"][key].append(value)
                else:
                    node["metadata"][key] = [node["metadata"][key], value]
        
        # Update contexts if available
        if "start_pos" in entity and "end_pos" in entity and source_document:
            context = {
                "document": source_document,
                "start_pos": entity["start_pos"],
                "end_pos": entity["end_pos"]
            }
            if "contexts" not in node:
                node["contexts"] = []
            node["contexts"].append(context)
    
    def _create_new_node(
        self, 
        entity: Dict[str, Any], 
        source_document: Optional[str] = None
    ) -> str:
        """Create a new node from an entity."""
        # Generate a unique node ID
        node_id = self._generate_node_id(entity)
        
        # Create the node object
        node = {
            "id": node_id,
            "text": entity["text"],
            "type": entity["type"],
            "occurrences": 1,
            "documents": [source_document] if source_document else [],
            "contexts": []
        }
        
        # Add metadata if available
        if "metadata" in entity:
            node["metadata"] = entity["metadata"]
        
        # Add context if position information is available
        if "start_pos" in entity and "end_pos" in entity and source_document:
            node["contexts"].append({
                "document": source_document,
                "start_pos": entity["start_pos"],
                "end_pos": entity["end_pos"]
            })
        
        # Store the node in the registry
        self.nodes[node_id] = node
        
        # Map the entity text to the node ID
        lookup_text = entity["text"] if self.case_sensitive else entity["text"].lower()
        self.entity_to_node_id[lookup_text] = node_id
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_node_by_entity(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """Get a node by entity text."""
        lookup_text = entity_text if self.case_sensitive else entity_text.lower()
        node_id = self.entity_to_node_id.get(lookup_text)
        
        if node_id:
            return self.nodes.get(node_id)
        
        return None
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the registry."""
        return list(self.nodes.values())
    
    def get_nodes_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific entity type."""
        return [node for node in self.nodes.values() if node["type"] == entity_type]
    
    def get_nodes_by_document(self, document: str) -> List[Dict[str, Any]]:
        """Get all nodes that appear in a specific document."""
        return [node for node in self.nodes.values() if document in node.get("documents", [])]
    
    def merge_nodes(self, node_id1: str, node_id2: str) -> str:
        """
        Merge two nodes into one.
        
        Args:
            node_id1: ID of the first node
            node_id2: ID of the second node
            
        Returns:
            Node ID of the merged node
        """
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            raise ValueError(f"Both node IDs must exist in the registry.")
        
        node1 = self.nodes[node_id1]
        node2 = self.nodes[node_id2]
        
        # Keep the node with more occurrences as the primary node
        if node1["occurrences"] >= node2["occurrences"]:
            primary_node_id, primary_node = node_id1, node1
            secondary_node_id, secondary_node = node_id2, node2
        else:
            primary_node_id, primary_node = node_id2, node2
            secondary_node_id, secondary_node = node_id1, node1
        
        # Update primary node with information from secondary node
        primary_node["occurrences"] += secondary_node["occurrences"]
        
        # Merge documents
        if "documents" in secondary_node:
            for doc in secondary_node["documents"]:
                if doc not in primary_node.get("documents", []):
                    if "documents" not in primary_node:
                        primary_node["documents"] = []
                    primary_node["documents"].append(doc)
        
        # Merge contexts
        if "contexts" in secondary_node:
            if "contexts" not in primary_node:
                primary_node["contexts"] = []
            primary_node["contexts"].extend(secondary_node["contexts"])
        
        # Merge metadata
        if "metadata" in secondary_node:
            if "metadata" not in primary_node:
                primary_node["metadata"] = {}
            for key, value in secondary_node["metadata"].items():
                if key not in primary_node["metadata"]:
                    primary_node["metadata"][key] = value
                elif isinstance(primary_node["metadata"][key], list):
                    if isinstance(value, list):
                        for v in value:
                            if v not in primary_node["metadata"][key]:
                                primary_node["metadata"][key].append(v)
                    elif value not in primary_node["metadata"][key]:
                        primary_node["metadata"][key].append(value)
                else:
                    if isinstance(value, list):
                        primary_node["metadata"][key] = [primary_node["metadata"][key]] + value
                    else:
                        primary_node["metadata"][key] = [primary_node["metadata"][key], value]
        
        # Update entity_to_node_id mapping
        for entity_text, node_id in list(self.entity_to_node_id.items()):
            if node_id == secondary_node_id:
                self.entity_to_node_id[entity_text] = primary_node_id
        
        # Remove the secondary node
        del self.nodes[secondary_node_id]
        
        return primary_node_id
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the registry.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if node was successfully removed, False otherwise
        """
        if node_id not in self.nodes:
            return False
        
        # Remove entity_to_node_id mappings
        for entity_text, nid in list(self.entity_to_node_id.items()):
            if nid == node_id:
                del self.entity_to_node_id[entity_text]
        
        # Remove the node
        del self.nodes[node_id]
        
        return True
    
    def export_to_json(self, output_file: str) -> None:
        """
        Export the node registry to a JSON file.
        
        Args:
            output_file: Path to the output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "nodes": self.nodes,
                "entity_to_node_id": self.entity_to_node_id
            }, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, input_file: str) -> None:
        """
        Import the node registry from a JSON file.
        
        Args:
            input_file: Path to the input JSON file
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.nodes = data.get("nodes", {})
        self.entity_to_node_id = data.get("entity_to_node_id", {})
        
        logger.info(f"Imported {len(self.nodes)} nodes from {input_file}")
