#!/usr/bin/env python3
"""
Test script to create Weaviate schema with none vectorizer
"""

import weaviate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create client
client = weaviate.Client(
    url="http://localhost:8080"
)

# Check connection
try:
    ready = client.is_ready()
    logger.info(f"Weaviate connection: {'ready' if ready else 'not ready'}")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    exit(1)

# Function to create schema with none vectorizer
def create_schema():
    # Document class
    class_obj = {
        "class": "Document",
        "vectorizer": "none",  # Use none vectorizer
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The content of the document chunk"
            },
            {
                "name": "title",
                "dataType": ["text"],
                "description": "The title of the document"
            },
            {
                "name": "documentId",
                "dataType": ["string"],
                "description": "The ID of the document"
            },
            {
                "name": "chunkIndex",
                "dataType": ["int"],
                "description": "The index of the chunk within the document"
            },
            {
                "name": "documentType",
                "dataType": ["string"],
                "description": "The type of document (pdf, docx, etc.)"
            },
            {
                "name": "source",
                "dataType": ["string"],
                "description": "The source of the document (file path, URL, etc.)"
            },
            {
                "name": "metadata",
                "dataType": ["text"],
                "description": "Additional metadata for the document in JSON format"
            }
        ]
    }
    
    # Entity class
    entity_class = {
        "class": "Entity",
        "vectorizer": "none",  # Use none vectorizer
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
                "description": "The text of the entity"
            },
            {
                "name": "type",
                "dataType": ["string"],
                "description": "The type of entity (PERSON, ORGANIZATION, etc.)"
            },
            {
                "name": "occurrences",
                "dataType": ["int"],
                "description": "Number of times this entity appears"
            },
            {
                "name": "metadata",
                "dataType": ["text"],
                "description": "Additional metadata for the entity in JSON format"
            }
        ]
    }
    
    # Relationship class
    relationship_class = {
        "class": "Relationship",
        "vectorizer": "none",  # Use none vectorizer
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
                "name": "metadata",
                "dataType": ["text"],
                "description": "Additional metadata for the relationship in JSON format"
            }
        ]
    }
    
    # Create schema classes
    try:
        # Check and delete existing classes if needed
        if client.schema.contains({"class": "Document"}):
            logger.info("Deleting existing Document class")
            client.schema.delete_class("Document")
            
        if client.schema.contains({"class": "Entity"}):
            logger.info("Deleting existing Entity class")
            client.schema.delete_class("Entity")
            
        if client.schema.contains({"class": "Relationship"}):
            logger.info("Deleting existing Relationship class")
            client.schema.delete_class("Relationship")
            
        # Create new classes
        client.schema.create_class(entity_class)
        logger.info("Created Entity class")
        
        client.schema.create_class(relationship_class)
        logger.info("Created Relationship class")
        
        client.schema.create_class(class_obj)
        logger.info("Created Document class")
        
        return True
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        return False

# Main function
if __name__ == "__main__":
    success = create_schema()
    if success:
        logger.info("Schema created successfully")
        print("SUCCESS: Weaviate schema created with none vectorizer")
    else:
        logger.error("Failed to create schema")
        print("ERROR: Failed to create Weaviate schema")
