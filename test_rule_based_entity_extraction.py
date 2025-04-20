#!/usr/bin/env python3
"""
Rule-Based Entity Extraction Test Script

This script tests the rule-based entity extraction capability of GraphRAG.
Since we're encountering issues with Ollama, this uses the built-in rule-based approach.

Usage:
    python test_rule_based_entity_extraction.py

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

def test_rule_based_extraction(file_path):
    """Test rule-based entity extraction."""
    document = load_test_document(file_path)
    logger.info(f"Testing rule-based entity extraction on {file_path}")
    
    # Create extractor with rule_based method
    extractor = EntityExtractor(
        extraction_method="rule_based",
        entity_types=[
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", 
            "MONEY", "PERCENT", "EMAIL", "PHONE", "URL"
        ]
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
        "file": os.path.basename(file_path),
        "extraction_method": "rule_based",
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
    
    # Get all test documents
    test_doc_dir = Path(project_root) / "test_data"
    test_docs = list(test_doc_dir.glob("*.txt"))
    
    if not test_docs:
        logger.error(f"No test documents found in {test_doc_dir}")
        return
    
    # Run tests on each document
    results = []
    for doc_path in test_docs:
        try:
            result = test_rule_based_extraction(doc_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"rule_based_extraction_{timestamp}.json"
    
    # Save results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n--- Rule-Based Entity Extraction Summary ---")
    for result in results:
        print(f"File: {result['file']}")
        print(f"- Execution time: {result['execution_time']:.2f} seconds")
        print(f"- Total entities: {result['total_entities']}")
        print(f"- Entity types: {result['entity_counts']}")
        
        # Print all entities by type
        entity_types = list(result["entity_counts"].keys())
        
        for entity_type in entity_types:
            entities_of_type = [e for e in result["entities"] if e["type"] == entity_type]
            print(f"\n  {entity_type} entities ({len(entities_of_type)}):")
            
            for entity in entities_of_type:
                print(f"    - {entity['text']}")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
