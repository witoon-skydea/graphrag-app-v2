#!/usr/bin/env python3
"""
Test Script for GraphRAG Knowledge Graph Builder

This script tests the complete knowledge graph building process with different models:
- mxbai-embed-large
- gemma3:12b

It compares the quality and performance of the resulting knowledge graphs.

Usage:
    python test_gemma_knowledge_graph.py

"""

import os
import sys
import json
import time
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from src
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_test_documents(directory):
    """Load all test documents from a directory."""
    documents = {}
    dir_path = Path(directory)
    
    for file_path in dir_path.glob("*.txt"):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents[doc_id] = content
    
    return documents

def test_knowledge_graph_building(documents, model_name):
    """Test knowledge graph building with specified model."""
    logger.info(f"Testing knowledge graph building with model: {model_name}")
    
    # Create temporary directory for the graph
    graph_dir = Path(project_root) / "test_results" / f"kg_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize graph builder
    graph_builder = KnowledgeGraphBuilder(
        extraction_method="ollama",
        identification_method="ollama",
        model_name=model_name,
        similarity_threshold=0.85,
        confidence_threshold=0.5,
        persist_path=str(graph_dir)
    )
    
    # Measure execution time
    start_time = time.time()
    result = graph_builder.process_documents(
        documents,
        batch_size=5,
        extract_entities=True,
        identify_relationships=True,
        cross_document_relationships=True
    )
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get graph statistics
    stats = graph_builder.graph_statistics()
    
    # Export graph for visualization
    network_file = graph_dir / "network_data.json"
    graph_builder.export_to_network_format(str(network_file))
    
    logger.info(f"Knowledge graph built in {execution_time:.2f} seconds")
    logger.info(f"Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
    
    # Add execution time to stats
    stats["execution_time"] = execution_time
    stats["model"] = model_name
    stats["processing_result"] = result
    
    return stats, str(graph_dir)

def main():
    """Main function to run the test."""
    # Create results directory
    results_dir = Path(project_root) / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Load test documents
    test_docs_dir = Path(project_root) / "test_data"
    documents = load_test_documents(test_docs_dir)
    
    if not documents:
        logger.error(f"No documents found in {test_docs_dir}. Please add some .txt files.")
        return
    
    logger.info(f"Loaded {len(documents)} documents for testing")
    
    # Define models to test
    models = ["mxbai-embed-large", "gemma3:12b"]
    
    # Run tests
    results = []
    graph_paths = []
    
    for model in models:
        try:
            stats, graph_path = test_knowledge_graph_building(documents, model)
            results.append(stats)
            graph_paths.append(graph_path)
        except Exception as e:
            logger.error(f"Error testing {model}: {e}")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"knowledge_graph_comparison_{timestamp}.json"
    
    # Save results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n--- Knowledge Graph Building Summary ---")
    for result in results:
        print(f"Model: {result['model']}")
        print(f"- Execution time: {result['execution_time']:.2f} seconds")
        print(f"- Total nodes: {result['total_nodes']}")
        print(f"- Total edges: {result['total_edges']}")
        print(f"- Node types: {result['node_types']}")
        print(f"- Edge types: {result['edge_types']}")
        print(f"- Average degree: {result['average_degree']:.2f}")
        print()
    
    # Write comparison report
    comparison_report = results_dir / f"model_comparison_report_{timestamp}.md"
    with open(comparison_report, 'w', encoding='utf-8') as f:
        f.write("# Knowledge Graph Model Comparison Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Environment\n\n")
        f.write(f"- Documents processed: {len(documents)}\n")
        f.write(f"- Models tested: {', '.join(models)}\n\n")
        
        f.write("## Results Summary\n\n")
        
        # Performance comparison table
        f.write("### Performance Metrics\n\n")
        f.write("| Model | Execution Time (s) | Memory Usage | CPU Usage |\n")
        f.write("|-------|-------------------|--------------|----------|\n")
        for result in results:
            f.write(f"| {result['model']} | {result['execution_time']:.2f} | N/A | N/A |\n")
        f.write("\n")
        
        # Knowledge graph comparison table
        f.write("### Knowledge Graph Metrics\n\n")
        f.write("| Model | Total Nodes | Total Edges | Average Degree | Documents |\n")
        f.write("|-------|-------------|------------|----------------|----------|\n")
        for result in results:
            f.write(f"| {result['model']} | {result['total_nodes']} | {result['total_edges']} | {result['average_degree']:.2f} | {result['total_documents']} |\n")
        f.write("\n")
        
        # Node type comparison
        f.write("### Node Types Extracted\n\n")
        all_node_types = set()
        for result in results:
            all_node_types.update(result['node_types'].keys())
        
        f.write("| Entity Type |")
        for model in models:
            f.write(f" {model} |")
        f.write("\n")
        
        f.write("|------------|")
        for _ in models:
            f.write("------------|")
        f.write("\n")
        
        for node_type in sorted(all_node_types):
            f.write(f"| {node_type} |")
            for result in results:
                count = result['node_types'].get(node_type, 0)
                f.write(f" {count} |")
            f.write("\n")
        f.write("\n")
        
        # Edge type comparison
        f.write("### Relationship Types Identified\n\n")
        all_edge_types = set()
        for result in results:
            all_edge_types.update(result['edge_types'].keys())
        
        f.write("| Relationship Type |")
        for model in models:
            f.write(f" {model} |")
        f.write("\n")
        
        f.write("|------------------|")
        for _ in models:
            f.write("------------|")
        f.write("\n")
        
        for edge_type in sorted(all_edge_types):
            f.write(f"| {edge_type} |")
            for result in results:
                count = result['edge_types'].get(edge_type, 0)
                f.write(f" {count} |")
            f.write("\n")
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("Based on the comparison above, we can draw the following conclusions:\n\n")
        
        # Analyze which model performed better
        if len(results) >= 2:
            # Determine which model found more entities
            model_nodes = [(r['model'], r['total_nodes']) for r in results]
            model_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Determine which model found more relationships
            model_edges = [(r['model'], r['total_edges']) for r in results]
            model_edges.sort(key=lambda x: x[1], reverse=True)
            
            # Determine which model was faster
            model_times = [(r['model'], r['execution_time']) for r in results]
            model_times.sort(key=lambda x: x[1])
            
            f.write(f"1. **Entity Extraction:** {model_nodes[0][0]} identified more entities ({model_nodes[0][1]} vs {model_nodes[1][1]}).\n")
            f.write(f"2. **Relationship Identification:** {model_edges[0][0]} identified more relationships ({model_edges[0][1]} vs {model_edges[1][1]}).\n")
            f.write(f"3. **Performance:** {model_times[0][0]} was faster ({model_times[0][1]:.2f}s vs {model_times[1][1]:.2f}s).\n\n")
            
            # Overall recommendation
            f.write("### Recommendation\n\n")
            
            if model_nodes[0][0] == model_edges[0][0]:
                better_model = model_nodes[0][0]
                f.write(f"The {better_model} model performed better overall in terms of entity and relationship extraction.\n")
            else:
                f.write("The results are mixed, with different models performing better in different aspects.\n")
            
            if model_times[0][0] != model_nodes[0][0]:
                f.write(f"Consider using {model_nodes[0][0]} for better quality and {model_times[0][0]} for better performance.\n")
        
        f.write("\n*Note: This is an automated analysis. Manual review of the quality of extracted entities and relationships is recommended.*\n")
    
    logger.info(f"Comparison report saved to {comparison_report}")

if __name__ == "__main__":
    main()
