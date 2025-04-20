#!/usr/bin/env python3
"""
Enhanced Test Script for GraphRAG Entity Extractor

This script tests the entity extraction capability of GraphRAG using different models:
- llama3
- gemma3:12b

Usage:
    python test_enhanced_entity_extraction.py

"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from src
from src.graphrag_engine.knowledge_graph.entity_extractor import EntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_test_document(file_path):
    """Load test document from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def test_entity_extraction(document, model_name):
    """Test entity extraction with specified model."""
    logger.info(f"Testing entity extraction with model: {model_name}")
    
    entity_types = [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", 
        "MONEY", "PERCENT", "FACILITY", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "CONCEPT"
    ]
    
    extractor = EntityExtractor(
        extraction_method="ollama",
        model_name=model_name,
        entity_types=entity_types
    )
    
    # Measure execution time
    start_time = time.time()
    entities = extractor.extract_entities(document)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Count entities by type
    entity_counts = {}
    for entity in entities:
        entity_type = entity.get("type", "UNKNOWN")
        if entity_type not in entity_counts:
            entity_counts[entity_type] = 0
        entity_counts[entity_type] += 1
    
    logger.info(f"Found {len(entities)} entities in {execution_time:.2f} seconds")
    logger.info(f"Entity types: {entity_counts}")
    
    return {
        "model": model_name,
        "execution_time": execution_time,
        "total_entities": len(entities),
        "entity_counts": entity_counts,
        "entities": entities
    }

def main():
    """Main function to run the test."""
    # Create results directory
    results_dir = Path(project_root) / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Load test document
    test_doc_path = Path(project_root) / "test_data" / "test_document.txt"
    document = load_test_document(test_doc_path)
    
    # Define models to test
    models = ["llama3.2", "gemma3:12b"]
    
    # Run tests
    results = []
    for model in models:
        try:
            result = test_entity_extraction(document, model)
            results.append(result)
        except Exception as e:
            logger.error(f"Error testing {model}: {e}")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"enhanced_entity_extraction_{timestamp}.json"
    
    # Save results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n--- Entity Extraction Summary ---")
    for result in results:
        print(f"Model: {result['model']}")
        print(f"- Execution time: {result['execution_time']:.2f} seconds")
        print(f"- Total entities: {result['total_entities']}")
        print(f"- Entity types: {result['entity_counts']}")
        print()
        
        # Print sample entities (first 5 of each type)
        print("Sample entities detected:")
        entity_types = list(result["entity_counts"].keys())
        
        for entity_type in entity_types:
            entities_of_type = [e for e in result["entities"] if e["type"] == entity_type]
            print(f"  {entity_type} (showing up to 5):")
            
            for i, entity in enumerate(entities_of_type[:5]):
                print(f"    - {entity['text']}")
            
            print()

if __name__ == "__main__":
    main()
