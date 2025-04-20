"""
Document Processor Module

This module processes documents of various types (txt, md, docx, pdf, csv, excel)
and extracts text and metadata from them.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import re
import tempfile

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes documents of various types and extracts text and metadata.
    
    Supported document types:
    - Text files (.txt, .md)
    - Word documents (.docx)
    - PDF files (.pdf)
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    """
    
    def __init__(self):
        """Initialize the DocumentProcessor."""
        self.supported_extensions = {
            ".txt": self._process_text_file,
            ".md": self._process_text_file,
            ".docx": self._process_docx_file,
            ".pdf": self._process_pdf_file,
            ".csv": self._process_csv_file,
            ".xlsx": self._process_excel_file,
            ".xls": self._process_excel_file
        }
        
        # Install required dependencies lazily
        self._docx_installed = False
        self._pdf_installed = False
        self._pandas_installed = False
        self._tesseract_installed = False
        self._pymupdf_installed = False
    
    def process_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a file and extract text and metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        # Validate file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "", {}
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if file type is supported
        if file_ext not in self.supported_extensions:
            logger.error(f"Unsupported file type: {file_ext}")
            return "", {}
        
        # Process file based on its type
        try:
            processor_func = self.supported_extensions[file_ext]
            return processor_func(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return "", {}
    
    def _process_text_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a text file (.txt, .md)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            file_stat = os.stat(file_path)
            metadata = {
                "file_size": file_stat.st_size,
                "created_at": file_stat.st_ctime,
                "modified_at": file_stat.st_mtime,
                "file_type": os.path.splitext(file_path)[1][1:].upper()
            }
            
            # Try to extract title from markdown files
            if file_path.endswith('.md'):
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if title_match:
                    metadata["title"] = title_match.group(1)
            
            return content, metadata
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return "", {}
    
    def _process_docx_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a Word document (.docx)."""
        if not self._docx_installed:
            try:
                # Lazy import
                global Document
                from docx import Document
                self._docx_installed = True
            except ImportError:
                logger.error("python-docx not installed. Install it with: pip install python-docx")
                return "", {"error": "python-docx not installed"}
        
        try:
            doc = Document(file_path)
            
            # Extract text from paragraphs
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text])
            
            # Extract metadata
            metadata = {
                "file_size": os.path.getsize(file_path),
                "created_at": os.path.getctime(file_path),
                "modified_at": os.path.getmtime(file_path),
                "file_type": "DOCX"
            }
            
            # Add title if available
            if doc.core_properties.title:
                metadata["title"] = doc.core_properties.title
            
            # Add author if available
            if doc.core_properties.author:
                metadata["author"] = doc.core_properties.author
            
            # Add other metadata
            if doc.core_properties.created:
                metadata["doc_created_at"] = doc.core_properties.created.strftime("%Y-%m-%d %H:%M:%S")
            if doc.core_properties.modified:
                metadata["doc_modified_at"] = doc.core_properties.modified.strftime("%Y-%m-%d %H:%M:%S")
            
            return text, metadata
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {e}")
            return "", {}
    
    def _process_pdf_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a PDF file (.pdf)."""
        # Try with PyPDF2 first
        if not self._pdf_installed:
            try:
                # Lazy import PyPDF2
                global PdfReader
                from PyPDF2 import PdfReader
                self._pdf_installed = True
            except ImportError:
                logger.warning("PyPDF2 not installed. Install it with: pip install PyPDF2")
        
        # Extracted text (to be filled)
        extracted_text = ""
        
        # Default metadata
        metadata = {
            "file_size": os.path.getsize(file_path),
            "created_at": os.path.getctime(file_path),
            "modified_at": os.path.getmtime(file_path),
            "file_type": "PDF"
        }
        
        # Try text extraction with PyPDF2 if available
        if self._pdf_installed:
            try:
                reader = PdfReader(file_path)
                page_texts = []
                
                # Extract text from each page
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_texts.append(page_text)
                
                # Add document info to metadata
                if reader.metadata:
                    if reader.metadata.title:
                        metadata["title"] = reader.metadata.title
                    if reader.metadata.author:
                        metadata["author"] = reader.metadata.author
                    if reader.metadata.creator:
                        metadata["creator"] = reader.metadata.creator
                    if reader.metadata.producer:
                        metadata["producer"] = reader.metadata.producer
                
                metadata["page_count"] = len(reader.pages)
                
                # Join all page texts
                extracted_text = "\n\n".join(page_texts)
                
                # If we got substantial text, return it
                if len(extracted_text) > 100:
                    return extracted_text, metadata
            except Exception as e:
                logger.warning(f"Error extracting text with PyPDF2 from {file_path}: {e}")
        
        # If PyPDF2 didn't work well or isn't installed, try with PyMuPDF
        if not self._pymupdf_installed:
            try:
                # Lazy import PyMuPDF
                global fitz
                import fitz
                self._pymupdf_installed = True
            except ImportError:
                logger.warning("PyMuPDF not installed. Install it with: pip install PyMuPDF")
        
        if self._pymupdf_installed:
            try:
                doc = fitz.open(file_path)
                page_texts = []
                
                # Add document info to metadata
                metadata["page_count"] = len(doc)
                if doc.metadata:
                    if "title" in doc.metadata and doc.metadata["title"]:
                        metadata["title"] = doc.metadata["title"]
                    if "author" in doc.metadata and doc.metadata["author"]:
                        metadata["author"] = doc.metadata["author"]
                    if "creator" in doc.metadata and doc.metadata["creator"]:
                        metadata["creator"] = doc.metadata["creator"]
                    if "producer" in doc.metadata and doc.metadata["producer"]:
                        metadata["producer"] = doc.metadata["producer"]
                
                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text:
                        page_texts.append(page_text)
                
                # Join all page texts
                extracted_text = "\n\n".join(page_texts)
                
                # If we got substantial text, return it
                if len(extracted_text) > 100:
                    return extracted_text, metadata
            except Exception as e:
                logger.warning(f"Error extracting text with PyMuPDF from {file_path}: {e}")
        
        # If text extraction is poor, try OCR if possible
        if (not extracted_text or len(extracted_text) < 100) and not self._tesseract_installed:
            try:
                # Lazy import for OCR
                global Image, pytesseract
                from PIL import Image
                import pytesseract
                self._tesseract_installed = True
            except ImportError:
                logger.warning("pytesseract not installed. Install it with: pip install pytesseract Pillow")
        
        if (not extracted_text or len(extracted_text) < 100) and self._tesseract_installed and self._pymupdf_installed:
            try:
                doc = fitz.open(file_path)
                page_texts = []
                
                # Process each page with OCR
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    
                    # Save to temporary image file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        pix.save(temp_filename)
                    
                    # Apply OCR
                    try:
                        img = Image.open(temp_filename)
                        page_text = pytesseract.image_to_string(img)
                        if page_text:
                            page_texts.append(page_text)
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_filename):
                            os.unlink(temp_filename)
                
                # Join all page texts
                extracted_text = "\n\n".join(page_texts)
                
                # Add OCR info to metadata
                metadata["extraction_method"] = "OCR"
            except Exception as e:
                logger.warning(f"Error performing OCR on {file_path}: {e}")
        
        return extracted_text, metadata
    
    def _process_csv_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a CSV file (.csv)."""
        if not self._pandas_installed:
            try:
                # Lazy import
                global pd
                import pandas as pd
                self._pandas_installed = True
            except ImportError:
                logger.error("pandas not installed. Install it with: pip install pandas")
                return "", {"error": "pandas not installed"}
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Extract metadata
            metadata = {
                "file_size": os.path.getsize(file_path),
                "created_at": os.path.getctime(file_path),
                "modified_at": os.path.getmtime(file_path),
                "file_type": "CSV",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist()
            }
            
            # Convert DataFrame to text representation
            text_rows = []
            
            # Add header
            text_rows.append("# " + ", ".join(df.columns))
            text_rows.append("")
            
            # Add rows
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                text_rows.append(row_text)
            
            return "\n".join(text_rows), metadata
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return "", {}
    
    def _process_excel_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process an Excel file (.xlsx, .xls)."""
        if not self._pandas_installed:
            try:
                # Lazy import
                global pd
                import pandas as pd
                self._pandas_installed = True
            except ImportError:
                logger.error("pandas not installed. Install it with: pip install pandas openpyxl")
                return "", {"error": "pandas not installed"}
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Extract metadata
            metadata = {
                "file_size": os.path.getsize(file_path),
                "created_at": os.path.getctime(file_path),
                "modified_at": os.path.getmtime(file_path),
                "file_type": "EXCEL",
                "sheet_count": len(excel_file.sheet_names),
                "sheets": excel_file.sheet_names
            }
            
            # Process each sheet
            all_sheet_texts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                sheet_rows = []
                
                # Add sheet header
                sheet_rows.append(f"## Sheet: {sheet_name}")
                sheet_rows.append("")
                
                # Add column header
                sheet_rows.append("# " + ", ".join(df.columns))
                sheet_rows.append("")
                
                # Add rows
                for idx, row in df.iterrows():
                    row_text = f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                    sheet_rows.append(row_text)
                
                sheet_rows.append("")
                all_sheet_texts.append("\n".join(sheet_rows))
            
            return "\n".join(all_sheet_texts), metadata
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {e}")
            return "", {}
