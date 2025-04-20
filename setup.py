#!/usr/bin/env python3
"""
GraphRAG Setup Script

This script handles the installation of the GraphRAG package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphrag",
    version="0.1.0",
    author="Witoon Pongsilathong",
    author_email="witoon@example.com",
    description="GraphRAG: Vector Database + Knowledge Graph for enhanced document search and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/witoonpongsilathong/graphrag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "click>=8.1.7",
        "tqdm>=4.66.1",
        "pytest>=7.4.3",
        "python-docx>=1.0.1",
        "PyPDF2>=3.0.1",
        "pdfplumber>=0.10.3",
        "pytesseract>=0.3.10",
        "PyMuPDF>=1.23.2",
        "pandas>=2.1.1",
        "openpyxl>=3.1.2",
        "Pillow>=10.0.1",
        "easyocr>=1.7.1",
        "requests>=2.31.0",
        "numpy>=1.26.0",
        "openai>=1.3.0",
        "anthropic>=0.5.0",
        "google-generativeai>=0.3.1",
        "weaviate-client>=4.3.3",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "typer>=0.9.0",
        "rich>=13.6.0",
    ],
    entry_points={
        "console_scripts": [
            "graphrag=main:graphrag",
        ],
    },
)
