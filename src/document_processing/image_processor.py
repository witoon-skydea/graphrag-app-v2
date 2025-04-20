"""
Image Processor Module

This module processes images and performs OCR (Optical Character Recognition)
to extract text from image files and images embedded in documents.
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Processes images and performs OCR to extract text.
    
    Supported image types:
    - PNG files (.png)
    - JPEG files (.jpg, .jpeg)
    - GIF files (.gif)
    - TIFF files (.tiff, .tif)
    - BMP files (.bmp)
    - WebP files (.webp)
    """
    
    def __init__(self, language: str = "eng+tha"):
        """
        Initialize the ImageProcessor.
        
        Args:
            language: Language(s) for OCR. Example: "eng" for English, "tha" for Thai,
                     "eng+tha" for both English and Thai.
        """
        self.language = language
        self.supported_extensions = {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"}
        
        # Install required dependencies lazily
        self._tesseract_installed = False
        self._pillow_installed = False
        self._easyocr_installed = False
    
    def process_image(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process an image file and extract text using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        # Validate file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "", {}
        
        # Get file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Check if image type is supported
        if file_ext not in self.supported_extensions:
            logger.error(f"Unsupported image type: {file_ext}")
            return "", {}
        
        # Process image with Tesseract OCR
        text_tesseract = self._process_with_tesseract(image_path)
        
        # Process image with EasyOCR as backup
        text_easyocr = ""
        if not text_tesseract or len(text_tesseract) < 50:
            text_easyocr = self._process_with_easyocr(image_path)
        
        # Use whichever result is better
        extracted_text = text_tesseract if len(text_tesseract) > len(text_easyocr) else text_easyocr
        
        # Extract metadata
        file_stat = os.stat(image_path)
        metadata = {
            "file_size": file_stat.st_size,
            "created_at": file_stat.st_ctime,
            "modified_at": file_stat.st_mtime,
            "file_type": file_ext[1:].upper(),
            "ocr_method": "Tesseract" if len(text_tesseract) > len(text_easyocr) else "EasyOCR",
            "ocr_language": self.language
        }
        
        # Add image dimensions if possible
        if self._pillow_installed:
            try:
                with Image.open(image_path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["format"] = img.format
                    metadata["mode"] = img.mode
            except Exception as e:
                logger.warning(f"Error getting image dimensions: {e}")
        
        return extracted_text, metadata
    
    def _process_with_tesseract(self, image_path: str) -> str:
        """Process an image with Tesseract OCR."""
        if not self._tesseract_installed:
            try:
                # Lazy import
                global Image, pytesseract
                from PIL import Image
                import pytesseract
                self._tesseract_installed = True
                self._pillow_installed = True
            except ImportError:
                logger.warning("pytesseract not installed. Install it with: pip install pytesseract Pillow")
                return ""
        
        if not self._tesseract_installed:
            return ""
        
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.language)
            return text
        except Exception as e:
            logger.warning(f"Error processing image with Tesseract OCR: {e}")
            return ""
    
    def _process_with_easyocr(self, image_path: str) -> str:
        """Process an image with EasyOCR as backup."""
        if not self._easyocr_installed:
            try:
                # Lazy import
                global Reader
                from easyocr import Reader
                self._easyocr_installed = True
            except ImportError:
                logger.warning("easyocr not installed. Install it with: pip install easyocr")
                return ""
        
        if not self._easyocr_installed:
            return ""
        
        try:
            # Parse languages for EasyOCR
            languages = self.language.split('+')
            
            # Map language codes to EasyOCR supported codes
            language_map = {
                "eng": "en",
                "tha": "th",
                "chi_sim": "ch_sim",
                "chi_tra": "ch_tra",
                "jpn": "ja",
                "kor": "ko",
                "fra": "fr",
                "deu": "de",
                "rus": "ru",
                "spa": "es",
                "por": "pt"
            }
            
            easyocr_langs = [language_map.get(lang, lang) for lang in languages if lang in language_map]
            
            # Default to English if no supported languages
            if not easyocr_langs:
                easyocr_langs = ["en"]
            
            reader = Reader(easyocr_langs)
            result = reader.readtext(image_path)
            
            # Extract text from result
            texts = [item[1] for item in result]
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"Error processing image with EasyOCR: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images (optional)
            
        Returns:
            List of dictionaries with image information
        """
        # Check if PyMuPDF is installed
        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF not installed. Install it with: pip install PyMuPDF")
            return []
        
        # Validate file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        # Create temporary directory for extracted images if not provided
        if not output_dir:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            doc = fitz.open(pdf_path)
            image_infos = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate filename
                    filename = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                    image_path = os.path.join(output_dir, filename)
                    
                    # Save image
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Process image with OCR
                    extracted_text, metadata = self.process_image(image_path)
                    
                    image_data = {
                        "page_num": page_num + 1,
                        "image_idx": img_idx + 1,
                        "path": image_path,
                        "extracted_text": extracted_text,
                        "metadata": metadata
                    }
                    
                    image_infos.append(image_data)
            
            return image_infos
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            return []
