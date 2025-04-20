"""
Relationship Identifier Module for Knowledge Graph Builder

This module is responsible for identifying relationships between entities
extracted from document content.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RelationshipIdentifier:
    """
    Identifies relationships between entities extracted from documents.
    
    The RelationshipIdentifier can use either local models (Ollama) or
    external APIs to identify relationships between entities.
    """
    
    def __init__(
        self,
        identification_method: str = "ollama",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize a RelationshipIdentifier.
        
        Args:
            identification_method: Method for relationship identification ('ollama', 'openai', 'anthropic',
                                   'gemini', 'rule_based', 'proximity', 'cooccurrence')
            model_name: Name of the model to use (default: 'llama2' for Ollama)
            api_key: API key for external APIs (required for OpenAI, Anthropic, Gemini)
            endpoint: Custom endpoint URL (default endpoint is used if not provided)
            relationship_types: List of relationship types to identify (e.g., 'WORKS_FOR', 'LOCATED_IN')
                              If None, all types will be identified
            threshold: Confidence threshold for relationship identification (0.0-1.0)
                      This is the minimum confidence value required for a relationship to be considered valid.
        """
        self.identification_method = identification_method
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.confidence_threshold = threshold  # Renamed for consistency
        
        # Set default relationship types if not provided
        self.relationship_types = relationship_types or [
            "WORKS_FOR", "LOCATED_IN", "PART_OF", "CREATOR_OF", "OWNER_OF",
            "MEMBER_OF", "RELATED_TO", "HAS_ROLE", "OCCURRED_AT", "SYNONYM_OF",
            "PARENT_OF", "CHILD_OF", "CONTAINS", "CAUSES", "INSTANCE_OF"
        ]
        
        # Set up the endpoint based on identification method
        if not self.endpoint:
            if identification_method == "ollama":
                self.endpoint = "http://localhost:11434/api/generate"
            elif identification_method == "openai":
                self.endpoint = "https://api.openai.com/v1/chat/completions"
            elif identification_method == "anthropic":
                self.endpoint = "https://api.anthropic.com/v1/messages"
            elif identification_method == "gemini":
                self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        
        logger.info(f"RelationshipIdentifier initialized with {identification_method} using model {model_name}")

    def identify_relationships(
        self, 
        entities: List[Dict[str, Any]], 
        text: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify relationships between provided entities.
        
        Args:
            entities: List of entities extracted from the document
            text: Original text from which entities were extracted (required for some methods)
            batch_size: Number of entity pairs to process in each batch (for LLM methods)
            
        Returns:
            List of dictionaries containing relationship information:
            [
                {
                    "source": "Entity1",
                    "source_type": "PERSON",
                    "target": "Entity2",
                    "target_type": "ORGANIZATION",
                    "relationship": "WORKS_FOR",
                    "confidence": 0.95,
                    "metadata": {}  # Additional metadata if available
                },
                ...
            ]
        """
        if not entities or len(entities) < 2:
            logger.warning("Not enough entities to identify relationships")
            return []
            
        # Some methods require the original text
        if self.identification_method in ["rule_based", "proximity", "cooccurrence"] and not text:
            raise ValueError(f"Original text is required for {self.identification_method} method")
        
        if self.identification_method == "ollama":
            return self._identify_with_ollama(entities, text, batch_size)
        elif self.identification_method == "openai":
            return self._identify_with_openai(entities, text, batch_size)
        elif self.identification_method == "anthropic":
            return self._identify_with_anthropic(entities, text, batch_size)
        elif self.identification_method == "gemini":
            return self._identify_with_gemini(entities, text, batch_size)
        elif self.identification_method == "rule_based":
            return self._identify_with_rules(entities, text)
        elif self.identification_method == "proximity":
            return self._identify_with_proximity(entities, text)
        elif self.identification_method == "cooccurrence":
            return self._identify_with_cooccurrence(entities, text)
        else:
            raise ValueError(f"Unsupported identification method: {self.identification_method}")
    
    def _create_entity_pairs(self, entities: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        """Create batches of entity pairs for relationship identification."""
        pairs = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Skip self-relationships
                if i != j:
                    pairs.append((entity1, entity2))
        
        # Create batches
        return [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    
    def _create_prompt(self, entity_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], text: Optional[str] = None) -> str:
        """Create a standardized prompt for relationship identification."""
        relationship_types_str = ", ".join(self.relationship_types)
        
        context = f"\nContext Text:\n{text[:2000]}...\n" if text else ""
        
        pairs_text = ""
        for i, (entity1, entity2) in enumerate(entity_pairs):
            pairs_text += f"Pair {i+1}:\n"
            pairs_text += f"  Entity1: {entity1['text']} (Type: {entity1['type']})\n"
            pairs_text += f"  Entity2: {entity2['text']} (Type: {entity2['type']})\n\n"
        
        prompt = f"""Identify relationships between the following pairs of entities.{context}

Consider the following relationship types: {relationship_types_str}

{pairs_text}
Return a JSON array with objects having the structure:
{{
  "pair_id": pair number (starting from 1),
  "source": "entity1 text",
  "source_type": "entity1 type",
  "target": "entity2 text",
  "target_type": "entity2 type",
  "relationship": "identified relationship type from the list above",
  "confidence": confidence score between 0.0 and 1.0,
  "bidirectional": true/false (whether the relationship goes both ways)
}}

If no clear relationship exists between a pair, do not include that pair in the results.

JSON Response (ONLY include the JSON array, no other text):
"""
        return prompt
        
    def _identify_with_ollama(
        self, 
        entities: List[Dict[str, Any]], 
        text: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify relationships using local Ollama model."""
        all_relationships = []
        entity_pairs_batches = self._create_entity_pairs(entities, batch_size)
        
        for batch in tqdm(entity_pairs_batches, desc="Identifying relationships"):
            prompt = self._create_prompt(batch, text)
            
            try:
                response = requests.post(
                    self.endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.text}")
                    continue
                
                response_data = response.json()
                response_text = response_data.get("response", "")
                
                # Extract JSON from response
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        relationships = json.loads(json_str)
                        # Filter relationships by confidence threshold
                        relationships = [rel for rel in relationships if rel.get("confidence", 0) >= self.confidence_threshold]
                        all_relationships.extend(relationships)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Ollama response JSON: {e}")
                else:
                    logger.error("No JSON found in Ollama response")
                    
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
        
        return all_relationships
    
    def _identify_with_openai(
        self, 
        entities: List[Dict[str, Any]], 
        text: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify relationships using OpenAI API."""
        if not self.api_key:
            raise ValueError("API key required for OpenAI identification")
        
        all_relationships = []
        entity_pairs_batches = self._create_entity_pairs(entities, batch_size)
        
        for batch in tqdm(entity_pairs_batches, desc="Identifying relationships"):
            prompt = self._create_prompt(batch, text)
            
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name or "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "You are a relationship identification assistant. Identify relationships between entities and return them in JSON format only."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.text}")
                    continue
                    
                response_data = response.json()
                response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Extract JSON from response
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        relationships = json.loads(json_str)
                        # Filter relationships by confidence threshold
                        relationships = [rel for rel in relationships if rel.get("confidence", 0) >= self.confidence_threshold]
                        all_relationships.extend(relationships)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse OpenAI response JSON: {e}")
                else:
                    logger.error("No JSON found in OpenAI response")
                    
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
        
        return all_relationships
    
    def _identify_with_anthropic(
        self, 
        entities: List[Dict[str, Any]], 
        text: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify relationships using Anthropic API."""
        if not self.api_key:
            raise ValueError("API key required for Anthropic identification")
        
        all_relationships = []
        entity_pairs_batches = self._create_entity_pairs(entities, batch_size)
        
        for batch in tqdm(entity_pairs_batches, desc="Identifying relationships"):
            prompt = self._create_prompt(batch, text)
            
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name or "claude-2.0",
                        "max_tokens": 1000,
                        "system": "You are a relationship identification assistant. Identify relationships between entities and return them in JSON format only.",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.0
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Anthropic API error: {response.text}")
                    continue
                    
                response_data = response.json()
                response_text = response_data.get("content", [{}])[0].get("text", "")
                
                # Extract JSON from response
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        relationships = json.loads(json_str)
                        # Filter relationships by confidence threshold
                        relationships = [rel for rel in relationships if rel.get("confidence", 0) >= self.confidence_threshold]
                        all_relationships.extend(relationships)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Anthropic response JSON: {e}")
                else:
                    logger.error("No JSON found in Anthropic response")
                    
            except Exception as e:
                logger.error(f"Error calling Anthropic API: {e}")
        
        return all_relationships
    
    def _identify_with_gemini(
        self, 
        entities: List[Dict[str, Any]], 
        text: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify relationships using Google Gemini API."""
        if not self.api_key:
            raise ValueError("API key required for Gemini identification")
        
        all_relationships = []
        entity_pairs_batches = self._create_entity_pairs(entities, batch_size)
        
        model_id = self.model_name or "gemini-pro"
        endpoint = f"{self.endpoint}/{model_id}:generateContent?key={self.api_key}"
        
        for batch in tqdm(entity_pairs_batches, desc="Identifying relationships"):
            prompt = self._create_prompt(batch, text)
            
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.0
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Gemini API error: {response.text}")
                    continue
                    
                response_data = response.json()
                response_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Extract JSON from response
                json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        relationships = json.loads(json_str)
                        # Filter relationships by confidence threshold
                        relationships = [rel for rel in relationships if rel.get("confidence", 0) >= self.confidence_threshold]
                        all_relationships.extend(relationships)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Gemini response JSON: {e}")
                else:
                    logger.error("No JSON found in Gemini response")
                    
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
        
        return all_relationships
    
    def _identify_with_rules(
        self, 
        entities: List[Dict[str, Any]], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Identify relationships using rule-based patterns."""
        if not text:
            raise ValueError("Text is required for rule-based relationship identification")
        
        relationships = []
        
        # Define patterns for common relationship types
        patterns = {
            "WORKS_FOR": [
                r'(\w+)\s+(?:works|worked|is employed|is working)\s+(?:for|at|in|with)\s+(\w+)',
                r'(\w+)\s+is\s+(?:a|an)\s+employee\s+of\s+(\w+)',
                r'(\w+)\s+(?:joined|hired by)\s+(\w+)'
            ],
            "LOCATED_IN": [
                r'(\w+)\s+(?:is located|is situated|is based|is found)\s+in\s+(\w+)',
                r'(\w+)\s+(?:is in|is at|in the)\s+(\w+)'
            ],
            "PART_OF": [
                r'(\w+)\s+is\s+(?:part of|a division of|a subsidiary of|a branch of)\s+(\w+)',
                r'(\w+)\s+(?:belongs to|is owned by)\s+(\w+)'
            ],
            "CREATOR_OF": [
                r'(\w+)\s+(?:created|developed|built|invented|designed)\s+(\w+)',
                r'(\w+)\s+is\s+the\s+(?:creator|author|developer|inventor)\s+of\s+(\w+)'
            ],
            "OCCURRED_AT": [
                r'(\w+)\s+(?:occurred|happened|took place|was held)\s+(?:in|at|on)\s+(\w+)',
                r'(\w+)\s+was\s+(?:during|in|at)\s+(\w+)'
            ]
        }
        
        # Create a mapping of entity text to its full entity object
        entity_map = {entity["text"]: entity for entity in entities}
        entity_texts = set(entity_map.keys())
        
        # Apply patterns to identify relationships
        for rel_type, pattern_list in patterns.items():
            if rel_type in self.relationship_types:
                for pattern in pattern_list:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        source_text = match.group(1)
                        target_text = match.group(2)
                        
                        # Find closest matching entities
                        source_entity = self._find_closest_entity(source_text, entity_texts)
                        target_entity = self._find_closest_entity(target_text, entity_texts)
                        
                        if source_entity and target_entity and source_entity != target_entity:
                            source = entity_map[source_entity]
                            target = entity_map[target_entity]
                            
                            relationships.append({
                                "source": source["text"],
                                "source_type": source["type"],
                                "target": target["text"],
                                "target_type": target["type"],
                                "relationship": rel_type,
                                "confidence": 0.7,  # Default confidence for rule-based
                                "bidirectional": False,
                                "metadata": {
                                    "match_text": match.group(0),
                                    "start_pos": match.start(),
                                    "end_pos": match.end()
                                }
                            })
        
        return relationships
    
    def _find_closest_entity(self, text: str, entity_texts: Set[str]) -> Optional[str]:
        """Find the closest matching entity text for a given text."""
        # Exact match
        if text in entity_texts:
            return text
        
        # Substring match
        for entity_text in entity_texts:
            if text.lower() in entity_text.lower() or entity_text.lower() in text.lower():
                return entity_text
        
        return None
    
    def _identify_with_proximity(
        self, 
        entities: List[Dict[str, Any]], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Identify relationships based on entity proximity in text."""
        if not text:
            raise ValueError("Text is required for proximity-based relationship identification")
        
        relationships = []
        max_distance = 50  # Maximum character distance between entities to consider a relationship
        
        # Sort entities by their position in the text
        sorted_entities = sorted(entities, key=lambda e: e["start_pos"])
        
        # Consider entities that are close to each other as potentially related
        for i, entity1 in enumerate(sorted_entities):
            for j, entity2 in enumerate(sorted_entities[i+1:], i+1):
                # Skip if entities are too far apart
                distance = entity2["start_pos"] - entity1["end_pos"]
                if distance > max_distance:
                    continue
                
                # Determine relationship type based on entity types
                rel_type = self._infer_relationship_from_types(entity1["type"], entity2["type"])
                if rel_type:
                    # Proximity-based confidence is inversely proportional to distance
                    confidence = max(0.5, 1.0 - (distance / max_distance))
                    
                    relationships.append({
                        "source": entity1["text"],
                        "source_type": entity1["type"],
                        "target": entity2["text"],
                        "target_type": entity2["type"],
                        "relationship": rel_type,
                        "confidence": confidence,
                        "bidirectional": False,
                        "metadata": {
                            "distance": distance,
                            "context": text[max(0, entity1["start_pos"]):min(len(text), entity2["end_pos"] + 20)]
                        }
                    })
        
        return relationships
    
    def _infer_relationship_from_types(self, type1: str, type2: str) -> Optional[str]:
        """Infer relationship type based on entity types."""
        # Simple heuristics for common entity type combinations
        type_pairs = {
            ("PERSON", "ORGANIZATION"): "WORKS_FOR",
            ("ORGANIZATION", "PERSON"): "EMPLOYS",
            ("PERSON", "LOCATION"): "LOCATED_IN",
            ("ORGANIZATION", "LOCATION"): "HEADQUARTERED_IN",
            ("PRODUCT", "ORGANIZATION"): "CREATED_BY",
            ("ORGANIZATION", "PRODUCT"): "CREATOR_OF",
            ("EVENT", "DATE"): "OCCURRED_ON",
            ("EVENT", "LOCATION"): "OCCURRED_AT",
            ("FACILITY", "LOCATION"): "LOCATED_IN",
            ("PERSON", "WORK_OF_ART"): "CREATOR_OF",
            ("WORK_OF_ART", "PERSON"): "CREATED_BY"
        }
        
        return type_pairs.get((type1, type2))
    
    def _identify_with_cooccurrence(
        self, 
        entities: List[Dict[str, Any]], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Identify relationships based on entity co-occurrence in text segments."""
        if not text:
            raise ValueError("Text is required for co-occurrence-based relationship identification")
        
        relationships = []
        
        # Split text into segments (e.g., sentences or paragraphs)
        segments = re.split(r'[.!?]\s+', text)
        
        # Track which entities appear in which segments
        entity_segments = {entity["text"]: set() for entity in entities}
        for i, segment in enumerate(segments):
            for entity in entities:
                if entity["text"] in segment:
                    entity_segments[entity["text"]].add(i)
        
        # Create entity pairs based on co-occurrence
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Find common segments
                common_segments = entity_segments[entity1["text"]] & entity_segments[entity2["text"]]
                if not common_segments:
                    continue
                
                # Calculate co-occurrence frequency
                freq1 = len(entity_segments[entity1["text"]])
                freq2 = len(entity_segments[entity2["text"]])
                common_freq = len(common_segments)
                
                # Calculate co-occurrence strength (Jaccard similarity)
                strength = common_freq / (freq1 + freq2 - common_freq) if (freq1 + freq2 - common_freq) > 0 else 0
                
                if strength >= self.confidence_threshold:
                    # Determine relationship type based on entity types or default to RELATED_TO
                    rel_type = self._infer_relationship_from_types(entity1["type"], entity2["type"]) or "RELATED_TO"
                    
                    relationships.append({
                        "source": entity1["text"],
                        "source_type": entity1["type"],
                        "target": entity2["text"],
                        "target_type": entity2["type"],
                        "relationship": rel_type,
                        "confidence": strength,
                        "bidirectional": True,
                        "metadata": {
                            "common_segments": len(common_segments),
                            "total_segments": len(segments),
                            "source_frequency": freq1,
                            "target_frequency": freq2
                        }
                    })
        
        return relationships