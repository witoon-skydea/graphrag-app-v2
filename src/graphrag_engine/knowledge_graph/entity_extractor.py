"""
Entity Extractor Module for Knowledge Graph Builder

This module is responsible for extracting entities from document content.
It supports extraction using both local models (Ollama) and external APIs.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import requests

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts named entities from text content using LLM or rule-based methods.
    
    The EntityExtractor can use either a local Ollama model or an external API
    like OpenAI, Anthropic, or Gemini to extract entities.
    """
    
    def __init__(
        self, 
        extraction_method: str = "ollama",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        entity_types: Optional[List[str]] = None
    ):
        """
        Initialize an EntityExtractor.
        
        Args:
            extraction_method: Method for entity extraction ('ollama', 'openai', 'anthropic', 'gemini', 'rule_based')
            model_name: Name of the model to use (default: 'llama2' for Ollama)
            api_key: API key for external APIs (required for OpenAI, Anthropic, Gemini)
            endpoint: Custom endpoint URL (default endpoint is used if not provided)
            entity_types: List of entity types to extract (e.g., 'PERSON', 'ORGANIZATION', 'LOCATION')
                          If None, default types will be used. These entity_types override any settings
                          in KnowledgeGraphBuilder.
        """
        self.extraction_method = extraction_method
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = endpoint
        
        # Default entity types - used only if entity_types parameter is None
        # When passed through KnowledgeGraphBuilder, the builder's entity_types take precedence
        default_entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", 
            "MONEY", "PERCENT", "FACILITY", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LAW", "LANGUAGE", "CONCEPT"
        ]
        
        # Use provided entity_types if available, otherwise use defaults
        self.entity_types = entity_types or default_entity_types
        
        # Set up the endpoint based on extraction method
        if not self.endpoint:
            if extraction_method == "ollama":
                self.endpoint = "http://localhost:11434/api/generate"
            elif extraction_method == "openai":
                self.endpoint = "https://api.openai.com/v1/chat/completions"
            elif extraction_method == "anthropic":
                self.endpoint = "https://api.anthropic.com/v1/messages"
            elif extraction_method == "gemini":
                self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        
        logger.info(f"EntityExtractor initialized with {extraction_method} using model {model_name}")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the provided text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of dictionaries containing entity information:
            [
                {
                    "text": "Entity text",
                    "type": "Entity type",
                    "start_pos": 0,  # Character position in the original text
                    "end_pos": 10,
                    "metadata": {}  # Additional metadata if available
                },
                ...
            ]
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for entity extraction")
            return []
            
        # Truncate text if it's too long
        if len(text) > 10000:
            logger.warning(f"Text is too long ({len(text)} chars), truncating to 10000 chars")
            text = text[:10000]
        
        if self.extraction_method == "ollama":
            return self._extract_with_ollama(text)
        elif self.extraction_method == "openai":
            return self._extract_with_openai(text)
        elif self.extraction_method == "anthropic":
            return self._extract_with_anthropic(text)
        elif self.extraction_method == "gemini":
            return self._extract_with_gemini(text)
        elif self.extraction_method == "rule_based":
            return self._extract_with_rules(text)
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")
    
    def _create_prompt(self, text: str) -> str:
        """Create a standardized prompt for entity extraction."""
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""Extract entities from the following text. 
Return a JSON array with objects having the structure:
{{
  "text": "the entity text",
  "type": "entity type from the list: {entity_types_str}",
  "start_pos": starting character position in the text,
  "end_pos": ending character position in the text
}}

Text:
{text}

JSON Response (ONLY include the JSON array, no other text):
"""
        return prompt
        
    def _extract_with_ollama(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using local Ollama model."""
        prompt = self._create_prompt(text)
        
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
                return []
            
            response_data = response.json()
            response_text = response_data.get("response", "")
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    entities = json.loads(json_str)
                    return entities
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Ollama response JSON: {e}")
                    return []
            else:
                logger.error("No JSON found in Ollama response")
                return []
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return []
    
    def _extract_with_openai(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using OpenAI API."""
        if not self.api_key:
            logger.error("API key required for OpenAI extraction but none provided")
            raise ValueError("API key required for OpenAI extraction")
        
        if not text:
            logger.warning("Empty text provided to _extract_with_openai")
            return []
        
        prompt = self._create_prompt(text)
        
        try:
            # Set timeout to avoid hanging indefinitely
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name or "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an entity extraction assistant. Extract entities and return them in JSON format only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                },
                timeout=(10, 60)  # (connect timeout, read timeout)
            )
            
            if response is None:
                logger.error("OpenAI API returned None response")
                return []
                
            if response.status_code != 200:
                logger.error(f"OpenAI API error: Status code {response.status_code}, Response: {response.text}")
                return []
                
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI API response as JSON: {e}, Response: {response.text}")
                return []
                
            if not response_data:
                logger.error("Empty response data from OpenAI API")
                return []
                
            choices = response_data.get("choices", [])
            if not choices:
                logger.error("No choices in OpenAI API response")
                return []
                
            first_choice = choices[0]
            if not first_choice:
                logger.error("Empty first choice in OpenAI API response")
                return []
                
            message = first_choice.get("message", {})
            if not message:
                logger.error("No message in first choice of OpenAI API response")
                return []
                
            response_text = message.get("content", "")
            if not response_text:
                logger.error("Empty content in message of OpenAI API response")
                return []
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON array found in OpenAI response text")
                return []
                
            json_str = json_match.group(0)
            try:
                entities = json.loads(json_str)
                
                # Validate entities structure
                valid_entities = []
                for entity in entities:
                    if not isinstance(entity, dict):
                        logger.warning(f"Skipping non-dict entity: {entity}")
                        continue
                        
                    if "text" not in entity or "type" not in entity:
                        logger.warning(f"Skipping entity missing required fields: {entity}")
                        continue
                        
                    # Ensure all required fields are present
                    valid_entity = {
                        "text": str(entity.get("text", "")),
                        "type": str(entity.get("type", "UNKNOWN")),
                        "start_pos": int(entity.get("start_pos", 0)),
                        "end_pos": int(entity.get("end_pos", 0)),
                        "metadata": entity.get("metadata", {})
                    }
                    valid_entities.append(valid_entity)
                
                return valid_entities
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response JSON: {e}, JSON string: {json_str}")
                return []
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing entities from OpenAI response: {e}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Network error when calling OpenAI API: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            return []
    
    def _extract_with_anthropic(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Anthropic API."""
        if not self.api_key:
            raise ValueError("API key required for Anthropic extraction")
        
        prompt = self._create_prompt(text)
        
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
                    "system": "You are an entity extraction assistant. Extract entities and return them in JSON format only.",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Anthropic API error: {response.text}")
                return []
                
            response_data = response.json()
            response_text = response_data.get("content", [{}])[0].get("text", "")
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    entities = json.loads(json_str)
                    return entities
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Anthropic response JSON: {e}")
                    return []
            else:
                logger.error("No JSON found in Anthropic response")
                return []
                
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return []
    
    def _extract_with_gemini(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Google Gemini API."""
        if not self.api_key:
            raise ValueError("API key required for Gemini extraction")
        
        prompt = self._create_prompt(text)
        
        model_id = self.model_name or "gemini-pro"
        endpoint = f"{self.endpoint}/{model_id}:generateContent?key={self.api_key}"
        
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
                return []
                
            response_data = response.json()
            response_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Extract JSON from response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    entities = json.loads(json_str)
                    return entities
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini response JSON: {e}")
                    return []
            else:
                logger.error("No JSON found in Gemini response")
                return []
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return []
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based approach with regex patterns."""
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            "DATE": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            "TIME": r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
            "MONEY": r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s+(?:dollars|baht|USD|THB)\b',
            "PERSON": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "ORGANIZATION": r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+\s+(?:Inc|LLC|Corp|Company|Co|Ltd)\b',
            "EMAIL": r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            "PHONE": r'\b\+?(?:\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "URL": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        }
        
        # Only use patterns for requested entity types
        for entity_type in self.entity_types:
            if entity_type in patterns:
                for match in re.finditer(patterns[entity_type], text):
                    entities.append({
                        "text": match.group(0),
                        "type": entity_type,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "metadata": {}
                    })
        
        return entities
