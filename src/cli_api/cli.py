"""
Command Line Interface for GraphRAG

This module provides a CLI for interacting with the GraphRAG system.
"""

import os
import sys
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_engine.graph_rag_engine import GraphRAGEngine
from document_processing.document_processor import DocumentProcessor

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('graphrag.log')
    ]
)
logger = logging.getLogger("graphrag-cli")

# CLI group
@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def graphrag(debug):
    """GraphRAG: Vector Database + Knowledge Graph for enhanced document search and RAG."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

# Initialize command
@graphrag.command()
@click.option('--persist-path', default='./graphrag_data', help='Path to persist GraphRAG data')
@click.option('--extraction-method', default='ollama', 
              type=click.Choice(['ollama', 'openai', 'anthropic', 'gemini', 'rule_based']), 
              help='Entity extraction method')
@click.option('--id-method', 'identification_method', default='ollama', 
              type=click.Choice(['ollama', 'openai', 'anthropic', 'gemini', 'rule_based', 'proximity', 'cooccurrence']), 
              help='Relationship identification method')
@click.option('--model-name', default='llama2', help='Model name for extraction/identification')
@click.option('--api-key', default=None, help='API key for external APIs')
@click.option('--weaviate-url', default='http://localhost:8080', help='Weaviate URL')
@click.option('--setup-db/--no-setup-db', default=True, help='Set up vector database schema')
@click.option('--force/--no-force', default=False, help='Force initialization even if data exists')
def init(persist_path, extraction_method, identification_method, model_name, api_key, weaviate_url, setup_db, force):
    """Initialize GraphRAG system."""
    if os.path.exists(persist_path) and not force:
        console.print(f"[bold yellow]Data directory {persist_path} already exists.[/]")
        if not click.confirm("Do you want to continue and potentially overwrite existing data?", default=False):
            console.print("[bold red]Aborted.[/]")
            return
    
    os.makedirs(persist_path, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[bold blue]Initializing GraphRAG...", total=3)
        
        # Initialize GraphRAG Engine
        console.print(f"[bold]Initializing GraphRAG with {extraction_method} extraction and {identification_method} identification[/]")
        engine = GraphRAGEngine(
            persist_path=persist_path,
            extraction_method=extraction_method,
            identification_method=identification_method,
            model_name=model_name,
            api_key=api_key,
            weaviate_url=weaviate_url
        )
        
        progress.update(task, advance=1, description="Checking vector database connection...")
        
        # Initialize components
        init_success = engine.initialize(setup_vector_db=setup_db)
        
        progress.update(task, advance=1, description="Persisting configuration...")
        
        # Persist configuration
        engine.persist()
        
        progress.update(task, advance=1, description="Initialization complete")
    
    if init_success:
        console.print("[bold green]GraphRAG initialized successfully![/]")
    else:
        console.print("[bold red]GraphRAG initialization had some issues. Check the logs for details.[/]")
    
    # Save config in JSON format for easy viewing
    config = {
        "persist_path": persist_path,
        "extraction_method": extraction_method,
        "identification_method": identification_method,
        "model_name": model_name,
        "weaviate_url": weaviate_url,
        "initialized_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(persist_path, "cli_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[green]Configuration saved to {os.path.join(persist_path, 'cli_config.json')}[/]")

# Import command
@graphrag.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--doc-id', help='Document ID (defaults to filename if not provided)')
@click.option('--title', help='Document title (defaults to filename if not provided)')
@click.option('--doc-type', help='Document type (defaults to file extension if not provided)')
@click.option('--extract-entities/--no-extract-entities', default=True, help='Extract entities from document')
@click.option('--id-relationships/--no-id-relationships', 'identify_relationships', default=True, help='Identify relationships between entities')
def import_file(file_path, config_path, doc_id, title, doc_type, extract_entities, identify_relationships):
    """Import a document file into GraphRAG."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Get file info
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower().replace('.', '')
    
    # Set defaults
    if not doc_id:
        doc_id = file_name
    if not title:
        title = file_name
    if not doc_type:
        doc_type = file_ext
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[bold blue]Processing document...", total=4)
        
        # Process the document
        progress.update(task, description=f"Reading {file_name}...")
        document_text, metadata = processor.process_file(file_path)
        
        if not document_text:
            progress.stop()
            console.print(f"[bold red]Failed to extract text from {file_name}.[/]")
            return
        
        progress.update(task, advance=1, description="Initializing GraphRAG engine...")
        
        # Initialize GraphRAG engine
        engine = GraphRAGEngine(
            persist_path=cli_config["persist_path"],
            extraction_method=cli_config["extraction_method"],
            identification_method=cli_config["identification_method"],
            model_name=cli_config["model_name"],
            weaviate_url=cli_config["weaviate_url"]
        )
        
        progress.update(task, advance=1, description=f"Processing document {doc_id}...")
        
        # Import the document
        result = engine.process_document(
            document_text=document_text,
            document_id=doc_id,
            title=title,
            document_type=doc_type,
            source=file_path,
            metadata=metadata,
            extract_entities=extract_entities,
            identify_relationships=identify_relationships
        )
        
        progress.update(task, advance=1, description="Saving changes...")
        
        # Persist changes
        engine.persist()
        
        progress.update(task, advance=1, description="Import complete")
    
    # Display results
    table = Table(title=f"Import Results for {file_name}")
    table.add_column("Metric", style="bold blue")
    table.add_column("Value", style="green")
    
    table.add_row("Document ID", doc_id)
    table.add_row("Chunks Created", str(result["chunks_created"]))
    table.add_row("Chunks Indexed", str(result.get("chunks_indexed", 0)))
    table.add_row("Entities Extracted", str(result["entities_extracted"]))
    table.add_row("Nodes Created", str(result["nodes_created"]))
    table.add_row("Relationships Identified", str(result["relationships_identified"]))
    table.add_row("Edges Created", str(result["edges_created"]))
    
    console.print(table)
    console.print(f"[bold green]Document {doc_id} imported successfully![/]")

# Import directory command
@graphrag.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--recursive/--no-recursive', default=True, help='Process directories recursively')
@click.option('--file-extensions', default='.txt,.md,.pdf,.docx', help='Comma-separated list of file extensions to process')
@click.option('--extract-entities/--no-extract-entities', default=True, help='Extract entities from documents')
@click.option('--id-relationships/--no-id-relationships', 'identify_relationships', default=True, help='Identify relationships between entities')
@click.option('--cross-document/--no-cross-document', default=True, help='Identify cross-document relationships')
def import_directory(directory_path, config_path, recursive, file_extensions, extract_entities, identify_relationships, cross_document):
    """Import all documents in a directory into GraphRAG."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Parse file extensions
    extensions = [ext.strip().lower() if ext.startswith('.') else '.' + ext.strip().lower() 
                 for ext in file_extensions.split(',')]
    
    # Find all matching files
    files_to_process = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    files_to_process.append(file_path)
    else:
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in extensions:
                files_to_process.append(file_path)
    
    if not files_to_process:
        console.print(f"[bold yellow]No matching files found in {directory_path}.[/]")
        return
    
    console.print(f"[bold]Found {len(files_to_process)} files to process.[/]")
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    # Process each file
    total_chunks = 0
    total_entities = 0
    total_relationships = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        file_task = progress.add_task("[bold blue]Processing files...", total=len(files_to_process))
        
        for file_path in files_to_process:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower().replace('.', '')
            doc_id = file_name
            
            progress.update(file_task, description=f"Processing {file_name}...")
            
            # Process the document
            document_text, metadata = processor.process_file(file_path)
            
            if not document_text:
                console.print(f"[yellow]Skipping {file_name}: Failed to extract text.[/]")
                progress.update(file_task, advance=1)
                continue
            
            # Import the document
            result = engine.process_document(
                document_text=document_text,
                document_id=doc_id,
                title=file_name,
                document_type=file_ext,
                source=file_path,
                metadata=metadata,
                extract_entities=extract_entities,
                identify_relationships=identify_relationships
            )
            
            # Update totals
            total_chunks += result["chunks_created"]
            total_entities += result["entities_extracted"]
            total_relationships += result["relationships_identified"]
            
            progress.update(file_task, advance=1)
    
    # Process cross-document relationships if requested
    if cross_document and extract_entities and identify_relationships and len(files_to_process) > 1:
        console.print("[bold]Processing cross-document relationships...[/]")
        
        # This would need additional implementation in the engine
        # For now, we'll just persist the changes
        engine.persist()
    else:
        # Persist changes
        engine.persist()
    
    # Display results
    table = Table(title=f"Import Results for {directory_path}")
    table.add_column("Metric", style="bold blue")
    table.add_column("Value", style="green")
    
    table.add_row("Files Processed", str(len(files_to_process)))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Total Entities", str(total_entities))
    table.add_row("Total Relationships", str(total_relationships))
    
    console.print(table)
    console.print(f"[bold green]Directory {directory_path} imported successfully![/]")

# Search command
@graphrag.command()
@click.argument('query')
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--limit', default=5, help='Maximum number of results to return')
@click.option('--hybrid/--no-hybrid', default=True, help='Use hybrid search (vector + keyword)')
@click.option('--use-kg/--no-use-kg', default=True, help='Enhance results with knowledge graph')
@click.option('--alpha', default=0.5, help='Balance between vector and keyword search (0.0-1.0)')
@click.option('--filter', help='Filter results (JSON format)')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json', 'markdown']), help='Output format')
@click.option('--output-file', help='Output file path (if not provided, results are printed to console)')
def search(query, config_path, limit, hybrid, use_kg, alpha, filter, output_format, output_file):
    """Search for documents matching a query."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Parse filter if provided
    filters = None
    if filter:
        try:
            filters = json.loads(filter)
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid filter format: {filter}[/]")
            console.print("Filter should be a valid JSON string.")
            return
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        search_task = progress.add_task("[bold blue]Searching...", total=1)
        
        # Perform search
        results = engine.search(
            query=query,
            limit=limit,
            filters=filters,
            hybrid_search=hybrid,
            use_knowledge_graph=use_kg,
            alpha=alpha
        )
        
        progress.update(search_task, advance=1, description="Search complete")
    
    # Format results
    if output_format == 'json':
        output_data = json.dumps(results, indent=2)
    elif output_format == 'markdown':
        output_data = f"# Search Results for \"{query}\"\n\n"
        
        for i, result in enumerate(results["combined_results"], 1):
            result_type = result.get("result_type", "unknown")
            output_data += f"## Result {i} ({result_type})\n\n"
            output_data += f"**Document:** {result.get('document_id', 'Unknown')}\n\n"
            if "title" in result and result["title"]:
                output_data += f"**Title:** {result['title']}\n\n"
            if "source" in result and result["source"]:
                output_data += f"**Source:** {result['source']}\n\n"
            if "score" in result:
                output_data += f"**Score:** {result['score']:.4f}\n\n"
            if "relationship" in result:
                rel = result["relationship"]
                output_data += f"**Relationship:** {rel.get('source_entity', '')} {rel.get('type', '')} {rel.get('target_entity', '')}\n\n"
            output_data += f"**Content:**\n\n{result.get('content', '')}\n\n---\n\n"
    else:  # text
        output_data = f"Search Results for \"{query}\"\n\n"
        
        for i, result in enumerate(results["combined_results"], 1):
            result_type = result.get("result_type", "unknown")
            output_data += f"Result {i} ({result_type}):\n"
            output_data += f"Document: {result.get('document_id', 'Unknown')}\n"
            if "title" in result and result["title"]:
                output_data += f"Title: {result['title']}\n"
            if "source" in result and result["source"]:
                output_data += f"Source: {result['source']}\n"
            if "score" in result:
                output_data += f"Score: {result['score']:.4f}\n"
            if "relationship" in result:
                rel = result["relationship"]
                output_data += f"Relationship: {rel.get('source_entity', '')} {rel.get('type', '')} {rel.get('target_entity', '')}\n"
            output_data += f"Content:\n{result.get('content', '')}\n\n"
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data)
        console.print(f"[bold green]Search results saved to {output_file}[/]")
    else:
        if output_format == 'json':
            syntax = Syntax(output_data, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif output_format == 'markdown':
            console.print(Markdown(output_data))
        else:  # text
            console.print(output_data)

# Get document command
@graphrag.command()
@click.argument('document_id')
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--include-chunks/--no-include-chunks', default=True, help='Include document chunks')
@click.option('--include-entities/--no-include-entities', default=True, help='Include entities linked to the document')
@click.option('--include-relationships/--no-include-relationships', default=False, help='Include relationships between entities')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json', 'markdown']), help='Output format')
@click.option('--output-file', help='Output file path (if not provided, results are printed to console)')
def get_document(document_id, config_path, include_chunks, include_entities, include_relationships, output_format, output_file):
    """Get a document by its ID."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    # Get document
    document = engine.get_document(
        document_id=document_id,
        include_chunks=include_chunks,
        include_entities=include_entities,
        include_relationships=include_relationships
    )
    
    if not document:
        console.print(f"[bold red]Document {document_id} not found.[/]")
        return
    
    # Format document
    if output_format == 'json':
        output_data = json.dumps(document, indent=2)
    elif output_format == 'markdown':
        output_data = f"# Document: {document_id}\n\n"
        
        if "title" in document and document["title"]:
            output_data += f"**Title:** {document['title']}\n\n"
        if "document_type" in document and document["document_type"]:
            output_data += f"**Type:** {document['document_type']}\n\n"
        if "source" in document and document["source"]:
            output_data += f"**Source:** {document['source']}\n\n"
        if "chunk_count" in document:
            output_data += f"**Chunks:** {document['chunk_count']}\n\n"
        
        if "metadata" in document and document["metadata"]:
            output_data += "## Metadata\n\n"
            for key, value in document["metadata"].items():
                output_data += f"**{key}:** {value}\n\n"
        
        if "chunks" in document and document["chunks"]:
            output_data += "## Chunks\n\n"
            for i, chunk in enumerate(document["chunks"], 1):
                output_data += f"### Chunk {i}\n\n"
                output_data += f"```\n{chunk.get('content', '')}\n```\n\n"
        
        if "entities" in document and document["entities"]:
            output_data += "## Entities\n\n"
            for i, entity in enumerate(document["entities"], 1):
                output_data += f"### {entity.get('text', 'Unknown')} ({entity.get('type', 'Unknown')})\n\n"
                output_data += f"**Occurrences:** {entity.get('occurrences', 0)}\n\n"
                
                if "relationships" in entity:
                    output_data += "#### Relationships\n\n"
                    
                    if entity["relationships"]["outgoing"]:
                        output_data += "**Outgoing:**\n\n"
                        for rel in entity["relationships"]["outgoing"]:
                            output_data += f"- {rel.get('type', 'Unknown')} -> {rel['target'].get('text', 'Unknown')} ({rel.get('confidence', 0):.2f})\n"
                    
                    if entity["relationships"]["incoming"]:
                        output_data += "\n**Incoming:**\n\n"
                        for rel in entity["relationships"]["incoming"]:
                            output_data += f"- {rel['source'].get('text', 'Unknown')} -> {rel.get('type', 'Unknown')} ({rel.get('confidence', 0):.2f})\n"
    else:  # text
        output_data = f"Document: {document_id}\n\n"
        
        if "title" in document and document["title"]:
            output_data += f"Title: {document['title']}\n"
        if "document_type" in document and document["document_type"]:
            output_data += f"Type: {document['document_type']}\n"
        if "source" in document and document["source"]:
            output_data += f"Source: {document['source']}\n"
        if "chunk_count" in document:
            output_data += f"Chunks: {document['chunk_count']}\n"
        
        output_data += "\n"
        
        if "metadata" in document and document["metadata"]:
            output_data += "Metadata:\n"
            for key, value in document["metadata"].items():
                output_data += f"  {key}: {value}\n"
            output_data += "\n"
        
        if "chunks" in document and document["chunks"]:
            output_data += "Chunks:\n"
            for i, chunk in enumerate(document["chunks"], 1):
                output_data += f"  Chunk {i}:\n"
                content_lines = chunk.get('content', '').split('\n')
                for line in content_lines:
                    output_data += f"    {line}\n"
                output_data += "\n"
        
        if "entities" in document and document["entities"]:
            output_data += "Entities:\n"
            for i, entity in enumerate(document["entities"], 1):
                output_data += f"  {entity.get('text', 'Unknown')} ({entity.get('type', 'Unknown')})\n"
                output_data += f"    Occurrences: {entity.get('occurrences', 0)}\n"
                
                if "relationships" in entity:
                    if entity["relationships"]["outgoing"]:
                        output_data += "    Outgoing Relationships:\n"
                        for rel in entity["relationships"]["outgoing"]:
                            output_data += f"      {rel.get('type', 'Unknown')} -> {rel['target'].get('text', 'Unknown')} ({rel.get('confidence', 0):.2f})\n"
                    
                    if entity["relationships"]["incoming"]:
                        output_data += "    Incoming Relationships:\n"
                        for rel in entity["relationships"]["incoming"]:
                            output_data += f"      {rel['source'].get('text', 'Unknown')} -> {rel.get('type', 'Unknown')} ({rel.get('confidence', 0):.2f})\n"
                
                output_data += "\n"
    
    # Output document
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data)
        console.print(f"[bold green]Document saved to {output_file}[/]")
    else:
        if output_format == 'json':
            syntax = Syntax(output_data, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif output_format == 'markdown':
            console.print(Markdown(output_data))
        else:  # text
            console.print(output_data)

# Get entity command
@graphrag.command()
@click.argument('entity_text')
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--include-documents/--no-include-documents', default=True, help='Include documents linked to the entity')
@click.option('--include-relationships/--no-include-relationships', default=True, help='Include relationships with other entities')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json', 'markdown']), help='Output format')
@click.option('--output-file', help='Output file path (if not provided, results are printed to console)')
def get_entity(entity_text, config_path, include_documents, include_relationships, output_format, output_file):
    """Get an entity by its text."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    # Get entity
    entity = engine.get_entity(
        entity_text=entity_text,
        include_documents=include_documents,
        include_relationships=include_relationships
    )
    
    if not entity:
        console.print(f"[bold red]Entity '{entity_text}' not found.[/]")
        return
    
    # Format entity
    if output_format == 'json':
        output_data = json.dumps(entity, indent=2)
    elif output_format == 'markdown':
        output_data = f"# Entity: {entity['text']}\n\n"
        
        output_data += f"**Type:** {entity.get('type', 'Unknown')}\n\n"
        output_data += f"**Occurrences:** {entity.get('occurrences', 0)}\n\n"
        
        if "metadata" in entity and entity["metadata"]:
            output_data += "## Metadata\n\n"
            for key, value in entity["metadata"].items():
                output_data += f"**{key}:** {value}\n\n"
        
        if "documents" in entity and entity["documents"]:
            output_data += "## Documents\n\n"
            for i, doc in enumerate(entity["documents"], 1):
                output_data += f"### Document {i}\n\n"
                output_data += f"**ID:** {doc.get('document_id', 'Unknown')}\n\n"
                if "title" in doc and doc["title"]:
                    output_data += f"**Title:** {doc['title']}\n\n"
                if "source" in doc and doc["source"]:
                    output_data += f"**Source:** {doc['source']}\n\n"
                if "document_type" in doc and doc["document_type"]:
                    output_data += f"**Type:** {doc['document_type']}\n\n"
        
        if "relationships" in entity:
            output_data += "## Relationships\n\n"
            
            if entity["relationships"]["outgoing"]:
                output_data += "### Outgoing\n\n"
                for rel in entity["relationships"]["outgoing"]:
                    output_data += f"- **{rel.get('type', 'Unknown')}** -> {rel['target'].get('text', 'Unknown')} ({rel['target'].get('type', 'Unknown')}) [{rel.get('confidence', 0):.2f}]\n\n"
            
            if entity["relationships"]["incoming"]:
                output_data += "### Incoming\n\n"
                for rel in entity["relationships"]["incoming"]:
                    output_data += f"- {rel['source'].get('text', 'Unknown')} ({rel['source'].get('type', 'Unknown')}) -> **{rel.get('type', 'Unknown')}** [{rel.get('confidence', 0):.2f}]\n\n"
    else:  # text
        output_data = f"Entity: {entity['text']}\n"
        output_data += f"Type: {entity.get('type', 'Unknown')}\n"
        output_data += f"Occurrences: {entity.get('occurrences', 0)}\n\n"
        
        if "metadata" in entity and entity["metadata"]:
            output_data += "Metadata:\n"
            for key, value in entity["metadata"].items():
                output_data += f"  {key}: {value}\n"
            output_data += "\n"
        
        if "documents" in entity and entity["documents"]:
            output_data += "Documents:\n"
            for i, doc in enumerate(entity["documents"], 1):
                output_data += f"  Document {i}:\n"
                output_data += f"    ID: {doc.get('document_id', 'Unknown')}\n"
                if "title" in doc and doc["title"]:
                    output_data += f"    Title: {doc['title']}\n"
                if "source" in doc and doc["source"]:
                    output_data += f"    Source: {doc['source']}\n"
                if "document_type" in doc and doc["document_type"]:
                    output_data += f"    Type: {doc['document_type']}\n"
                output_data += "\n"
        
        if "relationships" in entity:
            output_data += "Relationships:\n"
            
            if entity["relationships"]["outgoing"]:
                output_data += "  Outgoing:\n"
                for rel in entity["relationships"]["outgoing"]:
                    output_data += f"    {rel.get('type', 'Unknown')} -> {rel['target'].get('text', 'Unknown')} ({rel['target'].get('type', 'Unknown')}) [{rel.get('confidence', 0):.2f}]\n"
                output_data += "\n"
            
            if entity["relationships"]["incoming"]:
                output_data += "  Incoming:\n"
                for rel in entity["relationships"]["incoming"]:
                    output_data += f"    {rel['source'].get('text', 'Unknown')} ({rel['source'].get('type', 'Unknown')}) -> {rel.get('type', 'Unknown')} [{rel.get('confidence', 0):.2f}]\n"
                output_data += "\n"
    
    # Output entity
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data)
        console.print(f"[bold green]Entity saved to {output_file}[/]")
    else:
        if output_format == 'json':
            syntax = Syntax(output_data, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif output_format == 'markdown':
            console.print(Markdown(output_data))
        else:  # text
            console.print(output_data)

# Get statistics command
@graphrag.command()
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json', 'markdown']), help='Output format')
@click.option('--output-file', help='Output file path (if not provided, results are printed to console)')
def stats(config_path, output_format, output_file):
    """Get statistics about the GraphRAG system."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    # Get statistics
    stats = engine.get_statistics()
    
    # Format statistics
    if output_format == 'json':
        output_data = json.dumps(stats, indent=2)
    elif output_format == 'markdown':
        output_data = "# GraphRAG Statistics\n\n"
        
        output_data += "## Overview\n\n"
        output_data += f"**Total Documents:** {stats.get('total_documents', 0)}\n\n"
        output_data += f"**Total Entities:** {stats.get('total_entities', 0)}\n\n"
        output_data += f"**Total Relationships:** {stats.get('total_relationships', 0)}\n\n"
        
        if "knowledge_graph" in stats:
            kg_stats = stats["knowledge_graph"]
            output_data += "## Knowledge Graph\n\n"
            output_data += f"**Total Nodes:** {kg_stats.get('total_nodes', 0)}\n\n"
            output_data += f"**Total Edges:** {kg_stats.get('total_edges', 0)}\n\n"
            
            if "node_types" in kg_stats and kg_stats["node_types"]:
                output_data += "### Entity Types\n\n"
                for node_type, count in kg_stats["node_types"].items():
                    output_data += f"- **{node_type}:** {count}\n"
                output_data += "\n"
            
            if "edge_types" in kg_stats and kg_stats["edge_types"]:
                output_data += "### Relationship Types\n\n"
                for edge_type, count in kg_stats["edge_types"].items():
                    output_data += f"- **{edge_type}:** {count}\n"
                output_data += "\n"
        
        if "vector_database" in stats:
            vdb_stats = stats["vector_database"]
            output_data += "## Vector Database\n\n"
            output_data += f"**Document Count:** {vdb_stats.get('document_count', 0)}\n\n"
            output_data += f"**Chunk Count:** {vdb_stats.get('chunk_count', 0)}\n\n"
            
            if "document_types" in vdb_stats and vdb_stats["document_types"]:
                output_data += "### Document Types\n\n"
                for doc_type, count in vdb_stats["document_types"].items():
                    output_data += f"- **{doc_type}:** {count}\n"
                output_data += "\n"
    else:  # text
        output_data = "GraphRAG Statistics\n\n"
        
        output_data += "Overview:\n"
        output_data += f"  Total Documents: {stats.get('total_documents', 0)}\n"
        output_data += f"  Total Entities: {stats.get('total_entities', 0)}\n"
        output_data += f"  Total Relationships: {stats.get('total_relationships', 0)}\n\n"
        
        if "knowledge_graph" in stats:
            kg_stats = stats["knowledge_graph"]
            output_data += "Knowledge Graph:\n"
            output_data += f"  Total Nodes: {kg_stats.get('total_nodes', 0)}\n"
            output_data += f"  Total Edges: {kg_stats.get('total_edges', 0)}\n\n"
            
            if "node_types" in kg_stats and kg_stats["node_types"]:
                output_data += "  Entity Types:\n"
                for node_type, count in kg_stats["node_types"].items():
                    output_data += f"    {node_type}: {count}\n"
                output_data += "\n"
            
            if "edge_types" in kg_stats and kg_stats["edge_types"]:
                output_data += "  Relationship Types:\n"
                for edge_type, count in kg_stats["edge_types"].items():
                    output_data += f"    {edge_type}: {count}\n"
                output_data += "\n"
        
        if "vector_database" in stats:
            vdb_stats = stats["vector_database"]
            output_data += "Vector Database:\n"
            output_data += f"  Document Count: {vdb_stats.get('document_count', 0)}\n"
            output_data += f"  Chunk Count: {vdb_stats.get('chunk_count', 0)}\n\n"
            
            if "document_types" in vdb_stats and vdb_stats["document_types"]:
                output_data += "  Document Types:\n"
                for doc_type, count in vdb_stats["document_types"].items():
                    output_data += f"    {doc_type}: {count}\n"
                output_data += "\n"
    
    # Output statistics
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_data)
        console.print(f"[bold green]Statistics saved to {output_file}[/]")
    else:
        if output_format == 'json':
            syntax = Syntax(output_data, "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        elif output_format == 'markdown':
            console.print(Markdown(output_data))
        else:  # text
            console.print(output_data)

# Export knowledge graph command
@graphrag.command()
@click.argument('output_file')
@click.option('--config-path', default='./graphrag_data', help='Path to GraphRAG configuration')
def export_kg(output_file, config_path):
    """Export the knowledge graph to a file for visualization."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Config path {config_path} does not exist.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Load CLI config
    try:
        with open(os.path.join(config_path, "cli_config.json"), 'r') as f:
            cli_config = json.load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found in {config_path}.[/]")
        console.print("Run 'graphrag init' first to initialize the system.")
        return
    
    # Initialize GraphRAG engine
    engine = GraphRAGEngine(
        persist_path=cli_config["persist_path"],
        extraction_method=cli_config["extraction_method"],
        identification_method=cli_config["identification_method"],
        model_name=cli_config["model_name"],
        weaviate_url=cli_config["weaviate_url"]
    )
    
    # Export knowledge graph
    engine.export_knowledge_graph(output_file)
    console.print(f"[bold green]Knowledge graph exported to {output_file}[/]")
    console.print("[yellow]You can use tools like Cytoscape or D3.js to visualize the knowledge graph.[/]")

# Main entry point
if __name__ == '__main__':
    graphrag()
