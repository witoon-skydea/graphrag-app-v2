# GraphRAG System Design Document

## 1. ภาพรวมของระบบ

GraphRAG คือระบบจัดการและค้นหาเอกสาร ที่ผสมผสานเทคโนโลยี Vector Database และ Knowledge Graph เพื่อเพิ่มประสิทธิภาพในการจัดเก็บและค้นหาข้อมูลจากเอกสารหลากหลายประเภท ทั้ง docx, txt, md, pdf, csv และ excel โดยรองรับทั้งข้อความและรูปภาพ

### สถาปัตยกรรมโดยรวม

```
┌────────────────────────────────────┐
│          Document Processing       │
│ ┌──────────┐ ┌──────────┐ ┌──────┐│
│ │   Text   │ │  Image   │ │Struct││
│ │Extraction│ │Processing│ │ Data ││
│ └────┬─────┘ └────┬─────┘ └──┬───┘│
└──────┼──────────────────────┼─────┘
       │                      │
┌──────▼──────────────────────▼─────┐
│          GraphRAG Engine          │
│ ┌──────────┐      ┌─────────────┐ │
│ │  Vector  │◄────►│Knowledge    │ │
│ │ Database │      │Graph Builder│ │
│ └──────────┘      └─────────────┘ │
│       ▲                  ▲        │
└───────┼──────────────────┼────────┘
        │                  │
┌───────▼──────────────────▼────────┐
│         Embedding Module          │
│ ┌────────────┐    ┌─────────────┐ │
│ │Local Models│    │External APIs│ │
│ │  (Ollama)  │    │   (OpenAI)  │ │
│ └────────────┘    └─────────────┘ │
└────────────────────────────────────┘
        │                  │
┌───────▼──────────────────▼────────┐
│            CLI / API              │
└────────────────────────────────────┘
```

### เป้าหมายของระบบ

1. จัดการและนำเข้าเอกสารหลากหลายรูปแบบสู่ระบบ Vector Database
2. สร้างและจัดการ Knowledge Graph เพื่อเชื่อมโยงความสัมพันธ์ระหว่างเอกสาร
3. รองรับการ Embedding ทั้งแบบ Local (Ollama) และ API (OpenAI, Anthropic, Gemini, OpenRouter)
4. มี CLI Interface สำหรับการใช้งานขั้นพื้นฐาน
5. เตรียมพร้อมสำหรับการพัฒนา Web UI ในอนาคต (Phase 2)

### ข้อจำกัดและข้อกำหนด

- ใช้งานบน macbook air m2 ram 24GB
- ใช้ Weaviate เป็น Vector Database แบบ Open Source ติดตั้งบน Local
- ใช้ Python เป็นภาษาหลักในการพัฒนา

## 2. รายละเอียดแต่ละโมดูล

### Document Processing Module

โมดูลนี้ทำหน้าที่แปลงเอกสารหลากหลายรูปแบบให้เป็นข้อความที่สามารถนำไป Embedding และเก็บใน Vector Database ได้

#### Text Extraction
รับผิดชอบในการแปลงเอกสารประเภทข้อความให้เป็นรูปแบบที่จัดการได้:

- **docx**: ใช้ `python-docx` library
- **txt, md**: อ่านโดยตรงด้วย Python
- **PDF (ข้อความ)**: ใช้ `PyPDF2` หรือ `pdfplumber`

#### Image Processing
จัดการกับเอกสารที่มีรูปภาพหรือเป็นรูปภาพล้วน:

- **PDF (รูปภาพ)**: ใช้ `pytesseract` หรือ `EasyOCR` สำหรับ OCR
- **รูปภาพใน PDF**: ใช้ `PyMuPDF` (fitz) ดึงรูปภาพและใช้ OCR

#### Structured Data
แปลงข้อมูลที่มีโครงสร้างให้เป็นข้อความหรือ Knowledge Graph:

- **CSV, Excel**: ใช้ `pandas` และ `openpyxl` อ่านและแปลงข้อมูล
- แปลงโครงสร้างตารางให้เป็นข้อความหรือความสัมพันธ์

### Embedding Module

โมดูลนี้รับผิดชอบการสร้าง Vector Embeddings จากข้อความ โดยรองรับ 2 แบบ:

#### Local Embedding
ใช้ Ollama เป็น Local LLM สำหรับสร้าง Embeddings:

- รองรับโมเดลต่างๆ เช่น `llama2`, `nomic-embed-text`
- มี API wrapper สำหรับเรียกใช้ Ollama ได้สะดวก

#### External API Embedding
รองรับการใช้งาน API จากผู้ให้บริการภายนอก:

- **OpenAI**: ใช้ `text-embedding-3-small`
- **Anthropic**: รองรับ API ของ Claude model
- **Gemini**: รองรับ API ของ Google Gemini
- **OpenRouter**: รองรับการเชื่อมต่อผ่าน OpenRouter API

### GraphRAG Engine

หัวใจของระบบที่เชื่อมต่อและจัดการ Vector Database และ Knowledge Graph:

#### Vector Database (Weaviate)
จัดการการเก็บและค้นหา Vector Embeddings:

- สร้างและจัดการ Schema ใน Weaviate
- รองรับการค้นหาแบบ Vector Search และ Hybrid Search
- จัดการการเก็บข้อมูลต้นฉบับและ Metadata

#### Knowledge Graph Builder
สร้างและจัดการความสัมพันธ์ระหว่างเอกสาร:

- สร้างความเชื่อมโยงระหว่าง Entities ในเอกสาร
- เชื่อมโยงเอกสารที่มีเนื้อหาเกี่ยวข้องกัน
- ใช้ LLM ช่วยในการวิเคราะห์ความสัมพันธ์

### CLI / API Interface

ส่วนติดต่อกับผู้ใช้และระบบภายนอก:

#### Command Line Interface
CLI สำหรับการใช้งานพื้นฐาน:

- คำสั่งสำหรับนำเข้าเอกสาร: `graphrag import <path>`
- คำสั่งสำหรับค้นหา: `graphrag search <query>`
- คำสั่งจัดการระบบ: `graphrag manage <command>`

#### REST API
API สำหรับการเชื่อมต่อกับระบบภายนอก:

- Endpoints สำหรับนำเข้าเอกสาร
- Endpoints สำหรับค้นหาและดึงข้อมูล
- Endpoints สำหรับจัดการระบบ

## 3. รายละเอียดการจัดการเอกสารแต่ละประเภท

### การจัดการเอกสารข้อความ (.txt, .md, .docx)

```python
# ตัวอย่างโค้ดการจัดการเอกสารข้อความ
def process_text_document(file_path):
    if file_path.endswith('.txt') or file_path.endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    elif file_path.endswith('.docx'):
        from docx import Document
        doc = Document(file_path)
        content = '\n'.join([para.text for para in doc.paragraphs])
    
    # แบ่งเอกสารเป็น chunks
    chunks = chunk_text(content, chunk_size=1000, overlap=200)
    
    return chunks

def chunk_text(text, chunk_size=1000, overlap=200):
    """แบ่งข้อความเป็น chunks ขนาดเท่าๆ กัน พร้อม overlap"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # หากไม่ใช่ chunk สุดท้าย ให้พยายามตัดที่ช่องว่าง
        if end < len(text):
            # หาช่องว่างล่าสุด
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:  # หากไม่พบช่องว่าง ให้ตัดที่ chunk_size เลย
                end = start + chunk_size
        
        chunks.append(text[start:end])
        start = end - overlap  # ใช้ overlap
    
    return chunks
```

### การจัดการ PDF (ทั้งข้อความและรูปภาพ)

```python
# ตัวอย่างโค้ดการจัดการ PDF
def process_pdf(file_path):
    import pdfplumber
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    
    # ลองดึงข้อความก่อน
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    
    # ถ้าไม่มีข้อความหรือมีน้อยเกินไป ให้ใช้ OCR
    if not text_content or len(''.join(text_content)) < 100:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang='tha+eng')
            text_content.append(text)
    
    # แบ่งเป็น chunks
    chunks = []
    for page_text in text_content:
        chunks.extend(chunk_text(page_text))
    
    return chunks
```

### การจัดการ CSV/Excel

```python
# ตัวอย่างโค้ดการจัดการ CSV/Excel
def process_structured_data(file_path):
    import pandas as pd
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    
    # แปลงข้อมูลจากตารางเป็นข้อความ
    text_representation = []
    for idx, row in df.iterrows():
        row_text = f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
        text_representation.append(row_text)
    
    # สร้าง knowledge graph จากความสัมพันธ์ในข้อมูล
    graph_relationships = extract_relationships_from_df(df)
    
    return {
        'text_chunks': text_representation,
        'relationships': graph_relationships
    }

def extract_relationships_from_df(df):
    """สกัดความสัมพันธ์จาก DataFrame สำหรับสร้าง Knowledge Graph"""
    relationships = []
    
    # สมมติให้แต่ละแถวเป็น Entity และคอลัมน์เป็น Properties
    column_names = df.columns.tolist()
    
    # สร้างความสัมพันธ์ระหว่าง Entities ที่มี Property คล้ายกัน
    for i, row1 in df.iterrows():
        entity1 = f"Row_{i}"
        for j, row2 in df.iloc[i+1:].iterrows():
            entity2 = f"Row_{j}"
            
            # ตรวจสอบความคล้ายกันระหว่าง Entities
            similarities = []
            for col in column_names:
                if row1[col] == row2[col] and pd.notna(row1[col]):
                    similarities.append(col)
            
            if len(similarities) > 0:
                relationship = {
                    'source': entity1,
                    'target': entity2,
                    'relationship': f"similar_by_{','.join(similarities)}",
                    'weight': len(similarities) / len(column_names)
                }
                relationships.append(relationship)
    
    return relationships
```

## 4. การเชื่อมต่อกับ Weaviate และการสร้าง Knowledge Graph

### การเชื่อมต่อและสร้าง Schema

```python
# ตัวอย่างโค้ดการเชื่อมต่อกับ Weaviate
import weaviate
from weaviate.auth import AuthApiKey
import os

def setup_weaviate():
    client = weaviate.Client(
        url="http://localhost:8080",
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # สำหรับการใช้ OpenAI integration
        }
    )
    
    # ตรวจสอบหรือสร้าง schema
    if not client.schema.contains({"class": "Document"}):
        class_obj = {
            "class": "Document",
            "vectorizer": "text2vec-transformers",  # หรือใช้ "text2vec-openai" สำหรับ OpenAI
            "moduleConfig": {
                "text2vec-transformers": {
                    "poolingStrategy": "masked_mean"
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the document chunk"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Source file path"
                },
                {
                    "name": "chunkIndex",
                    "dataType": ["int"],
                    "description": "Index of this chunk in the document"
                },
                {
                    "name": "documentType",
                    "dataType": ["string"],
                    "description": "Type of document (pdf, docx, etc.)"
                },
                {
                    "name": "relatesTo",
                    "dataType": ["Document"],
                    "description": "Related document chunks"
                }
            ]
        }
        client.schema.create_class(class_obj)
    
    return client
```

### การสร้าง Knowledge Graph

```python
# ตัวอย่างโค้ดการสร้างความสัมพันธ์ใน Knowledge Graph
def create_document_relationships(client, doc_ids, similarity_threshold=0.7):
    """สร้างความสัมพันธ์ระหว่างเอกสารโดยอ้างอิงจากความคล้ายกัน"""
    # ดึงข้อมูลเอกสารทั้งหมด
    results = []
    for doc_id in doc_ids:
        result = client.data_object.get_by_id(
            doc_id, 
            class_name="Document",
            with_vector=True
        )
        if result:
            results.append(result)
    
    # หาความสัมพันธ์โดยใช้ความคล้ายของ vector
    relationships = []
    for i, doc1 in enumerate(results):
        for j, doc2 in enumerate(results[i+1:], i+1):
            # คำนวณ similarity ระหว่าง vectors
            similarity = calculate_similarity(doc1["vector"], doc2["vector"])
            
            if similarity > similarity_threshold:
                relationships.append({
                    "from_id": doc1["id"],
                    "to_id": doc2["id"],
                    "similarity": similarity
                })
    
    # บันทึกความสัมพันธ์กลับไปที่ Weaviate
    for rel in relationships:
        client.data_object.reference.add(
            from_class_name="Document",
            from_uuid=rel["from_id"],
            from_property_name="relatesTo",
            to_class_name="Document",
            to_uuid=rel["to_id"]
        )
    
    return relationships

def calculate_similarity(vec1, vec2):
    """คำนวณความคล้ายกันระหว่าง vectors ด้วย cosine similarity"""
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm
    
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```

## 5. การใช้ Ollama สำหรับ Local Embedding

```python
# ตัวอย่างโค้ดการใช้ Ollama สำหรับ local embedding
import requests
import numpy as np

def get_embedding_ollama(text, model="llama2"):
    """Get embedding from Ollama local model"""
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": model,
            "prompt": text
        }
    )
    
    if response.status_code == 200:
        return np.array(response.json()["embedding"])
    else:
        raise Exception(f"Error calling Ollama API: {response.text}")

def import_document_with_local_embedding(file_path, weaviate_client):
    """นำเข้าเอกสารและสร้าง embedding ด้วย Ollama"""
    # ประมวลผลเอกสาร
    if file_path.endswith(('.txt', '.md', '.docx')):
        chunks = process_text_document(file_path)
        doc_type = file_path.split('.')[-1]
    elif file_path.endswith('.pdf'):
        chunks = process_pdf(file_path)
        doc_type = 'pdf'
    elif file_path.endswith(('.csv', '.xlsx', '.xls')):
        result = process_structured_data(file_path)
        chunks = result['text_chunks']
        doc_type = 'structured'
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # สร้าง embedding และนำเข้า Weaviate
    doc_ids = []
    for i, chunk in enumerate(chunks):
        # สร้าง embedding โดย Ollama
        embedding = get_embedding_ollama(chunk)
        
        # นำเข้า Weaviate
        doc_id = weaviate_client.data_object.create(
            class_name="Document",
            data_object={
                "content": chunk,
                "source": file_path,
                "chunkIndex": i,
                "documentType": doc_type
            },
            vector=embedding.tolist()
        )
        doc_ids.append(doc_id)
    
    # สร้างความสัมพันธ์ระหว่างเอกสาร (สำหรับข้อมูลโครงสร้าง)
    if file_path.endswith(('.csv', '.xlsx', '.xls')):
        import_relationships(weaviate_client, result['relationships'], doc_ids)
    
    return doc_ids

def import_relationships(client, relationships, doc_ids):
    """นำเข้าความสัมพันธ์ที่สกัดได้จากข้อมูลโครงสร้าง"""
    # ทำการเชื่อมโยงความสัมพันธ์ระหว่างเอกสาร
    for rel in relationships:
        source_idx = int(rel['source'].split('_')[1])
        target_idx = int(rel['target'].split('_')[1])
        
        if source_idx < len(doc_ids) and target_idx < len(doc_ids):
            client.data_object.reference.add(
                from_class_name="Document",
                from_uuid=doc_ids[source_idx],
                from_property_name="relatesTo",
                to_class_name="Document",
                to_uuid=doc_ids[target_idx]
            )
```

## 6. โครงสร้าง CLI

```python
# ตัวอย่างโค้ดสำหรับ CLI
import click
import os

@click.group()
def cli():
    """GraphRAG CLI for managing documents and knowledge graph"""
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Import directory recursively')
@click.option('--embedding-model', default='local', type=click.Choice(['local', 'openai', 'anthropic', 'gemini']))
def import_docs(path, recursive, embedding_model):
    """Import documents to GraphRAG"""
    click.echo(f"Importing documents from {path} with {embedding_model} embedding model")
    
    # ตรวจสอบว่าเป็นไฟล์หรือโฟลเดอร์
    if os.path.isfile(path):
        files = [path]
    else:
        if recursive:
            files = []
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(('.txt', '.md', '.docx', '.pdf', '.csv', '.xlsx', '.xls')):
                        files.append(os.path.join(root, filename))
        else:
            files = [os.path.join(path, f) for f in os.listdir(path) 
                   if os.path.isfile(os.path.join(path, f)) and 
                   f.endswith(('.txt', '.md', '.docx', '.pdf', '.csv', '.xlsx', '.xls'))]
    
    # เชื่อมต่อ Weaviate
    client = setup_weaviate()
    
    # นำเข้าเอกสาร
    with click.progressbar(files, label='Importing documents') as bar:
        for file in bar:
            if embedding_model == 'local':
                import_document_with_local_embedding(file, client)
            else:
                import_document_with_api_embedding(file, client, embedding_model)

@cli.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results to return')
@click.option('--model', default='local', type=click.Choice(['local', 'openai', 'anthropic', 'gemini']))
def search(query, limit, model):
    """Search documents with GraphRAG"""
    click.echo(f"Searching for: {query}")
    
    # เชื่อมต่อ Weaviate
    client = setup_weaviate()
    
    # สร้าง embedding สำหรับคำค้นหา
    if model == 'local':
        query_embedding = get_embedding_ollama(query).tolist()
    else:
        query_embedding = get_api_embedding(query, model)
    
    # ทำการค้นหา
    result = client.query.get(
        "Document", ["content", "source", "chunkIndex", "documentType"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.7
    }).with_limit(limit).do()
    
    # แสดงผลลัพธ์
    if "data" in result and "Get" in result["data"] and "Document" in result["data"]["Get"]:
        docs = result["data"]["Get"]["Document"]
        for i, doc in enumerate(docs):
            click.echo(f"\n--- Result {i+1} ---")
            click.echo(f"Source: {doc['source']}")
            click.echo(f"Content: {doc['content'][:200]}...")
    else:
        click.echo("No results found")

@cli.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results to return')
def graph_search(query, limit):
    """Search documents using knowledge graph relationships"""
    click.echo(f"Graph searching for: {query}")
    
    # เชื่อมต่อ Weaviate
    client = setup_weaviate()
    
    # ทำการค้นหาด้วย GraphQL query ที่ใช้ความสัมพันธ์
    graphql_query = """
    {
      Get {
        Document(
          nearText: {
            concepts: ["QUERY_PLACEHOLDER"]
          }
          limit: LIMIT_PLACEHOLDER
        ) {
          content
          source
          chunkIndex
          documentType
          relatesTo {
            ... on Document {
              content
              source
              chunkIndex
            }
          }
        }
      }
    }
    """.replace("QUERY_PLACEHOLDER", query).replace("LIMIT_PLACEHOLDER", str(limit))
    
    result = client.query.raw(graphql_query)
    
    # แสดงผลลัพธ์
    if "data" in result and "Get" in result["data"] and "Document" in result["data"]["Get"]:
        docs = result["data"]["Get"]["Document"]
        for i, doc in enumerate(docs):
            click.echo(f"\n--- Result {i+1} ---")
            click.echo(f"Source: {doc['source']}")
            click.echo(f"Content: {doc['content'][:200]}...")
            
            if "relatesTo" in doc and doc["relatesTo"]:
                click.echo("\nRelated Documents:")
                for j, related in enumerate(doc["relatesTo"]):
                    click.echo(f"  {j+1}. {related['source']} (Chunk {related['chunkIndex']})")
    else:
        click.echo("No results found")

if __name__ == '__main__':
    cli()
```

## 7. ข้อกำหนดและการติดตั้ง

### ความต้องการของระบบ

- **ฮาร์ดแวร์**: MacBook Air M2 with 24GB RAM
- **ระบบปฏิบัติการ**: macOS

### การติดตั้ง Weaviate

```bash
# ติดตั้ง Weaviate ด้วย Docker
mkdir weaviate && cd weaviate
curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml"
docker-compose up -d
```

### การติดตั้ง Ollama

```bash
# ติดตั้ง Ollama จากเว็บไซต์ ollama.ai
# หรือใช้คำสั่ง brew
brew install ollama

# ดาวน์โหลดโมเดลที่จำเป็น
ollama pull llama2
ollama pull nomic-embed-text
```

### Python Dependencies

สร้างไฟล์ `requirements.txt`:

```
weaviate-client>=3.25.0
click>=8.0.0
python-docx>=0.8.11
PyPDF2>=3.0.0
pdfplumber>=0.7.0
pytesseract>=0.3.9
pillow>=9.0.0
pandas>=1.3.0
openpyxl>=3.0.9
requests>=2.28.0
numpy>=1.20.0
pymupdf>=1.19.0
langchain>=0.0.200
rich>=13.0.0
python-dotenv>=1.0.0
```

### การติดตั้งโปรเจค

```bash
# สร้างและเข้าสู่ virtual environment
python -m venv venv
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# ติดตั้งโปรเจคในโหมด development
pip install -e .
```

## 8. การเพิ่มประสิทธิภาพสำหรับ MacBook Air M2

เนื่องจากใช้งานบน MacBook Air M2 ที่มี RAM 24GB ควรมีการปรับแต่งดังนี้:

### การจัดการหน่วยความจำ

- จำกัดการใช้ RAM ของ Docker containers:
  ```yaml
  # ตัวอย่างการปรับแต่ง docker-compose.yml
  services:
    weaviate:
      image: semitechnologies/weaviate:latest
      deploy:
        resources:
          limits:
            memory: 12G  # จำกัดการใช้ RAM ไม่เกิน 12GB
  ```

- ปรับขนาด chunk ของเอกสารให้เหมาะสม:
  ```python
  # ปรับขนาด chunk ลงมาเหลือ 500-700 tokens เพื่อลดการใช้หน่วยความจำ
  chunks = chunk_text(content, chunk_size=600, overlap=100)
  ```

### การเพิ่มประสิทธิภาพ Apple Silicon

- ใช้ libraries ที่ optimize สำหรับ Apple Silicon:
  ```bash
  # ติดตั้ง tensorflow สำหรับ Mac
  pip install tensorflow-macos
  
  # ติดตั้ง torch สำหรับ Mac
  pip install torch torchvision torchaudio
  ```

- ใช้ประโยชน์จาก M-series neural engine:
  ```python
  # ตัวอย่างการตั้งค่า PyTorch สำหรับ MPS
  import torch
  
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model = model.to(device)
  ```

### การทำงานแบบขนาน

- ใช้ multiprocessing สำหรับการประมวลผลเอกสารขนาดใหญ่:
  ```python
  from multiprocessing import Pool
  
  def process_files_parallel(file_list, num_workers=4):
      with Pool(processes=num_workers) as pool:
          results = pool.map(process_file, file_list)
      return results
  ```

- ทำ batch processing สำหรับการนำเข้าเอกสารจำนวนมาก:
  ```python
  def batch_import(files, batch_size=10):
      for i in range(0, len(files), batch_size):
          batch = files[i:i+batch_size]
          process_files_parallel(batch)
  ```

## 9. แผนการพัฒนาในอนาคต (Phase 2)

### Web UI

#### ส่วนนำเข้าเอกสาร
- ระบบ drag-and-drop สำหรับอัปโหลดไฟล์
- แสดงความคืบหน้าและสถานะการนำเข้า
- จัดการ metadata ของเอกสาร

#### ส่วนค้นหาและแสดงผล
- แสดงผลการค้นหาพร้อมไฮไลต์คำสำคัญ
- แสดงความสัมพันธ์ระหว่างเอกสารด้วย graph visualization
- ระบบกรองและเรียงลำดับผลลัพธ์

#### ส่วนจัดการระบบ
- จัดการ API keys สำหรับบริการต่างๆ
- ตั้งค่าและปรับแต่งโมเดล embedding
- ตรวจสอบการใช้ทรัพยากรและประสิทธิภาพ

### การปรับปรุงประสิทธิภาพ

- เพิ่มความเร็วในการค้นหาด้วย index optimization
- ลดการใช้หน่วยความจำด้วยเทคนิคการทำ streaming และ lazy loading
- เพิ่มความแม่นยำในการค้นหาด้วยการปรับปรุงอัลกอริทึมการจับคู่

### Multi-modal Support

- รองรับการค้นหาด้วยรูปภาพ (image-to-text search)
- รองรับการค้นหาด้วยเสียง (speech-to-text search)
- การวิเคราะห์และจัดกลุ่มเนื้อหารูปภาพอัตโนมัติ

## 10. การใช้งานเบื้องต้น

### ติดตั้ง Weaviate และ Ollama

```bash
# ติดตั้ง Weaviate ด้วย Docker
mkdir weaviate && cd weaviate
curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml"
docker-compose up -d

# ติดตั้ง Ollama จากเว็บไซต์และดาวน์โหลดโมเดล
ollama pull llama2
ollama pull nomic-embed-text
```

### ติดตั้ง GraphRAG CLI

```bash
# สร้างและเข้าสู่ virtual environment
python -m venv venv
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# ติดตั้งโปรเจคในโหมด development
pip install -e .
```

### นำเข้าเอกสาร

```bash
# นำเข้าเอกสารเดี่ยว
graphrag import /path/to/document.pdf --embedding-model local

# นำเข้าโฟลเดอร์
graphrag import /path/to/documents --recursive --embedding-model local
```

### ค้นหาข้อมูล

```bash
# ค้นหาด้วย vector search ปกติ
graphrag search "ข้อมูลเกี่ยวกับการใช้งาน GraphRAG" --limit 5 --model local

# ค้นหาด้วย graph search
graphrag graph-search "ข้อมูลเกี่ยวกับการใช้งาน GraphRAG" --limit 5
```

## สรุป

ระบบ GraphRAG ที่ออกแบบนี้จะช่วยให้สามารถจัดการเอกสารหลากหลายรูปแบบได้อย่างมีประสิทธิภาพ โดยใช้ Weaviate เป็นฐานข้อมูล vector และสร้างความสัมพันธ์แบบ knowledge graph เพื่อเพิ่มประสิทธิภาพการค้นหา รองรับการทำงานทั้งแบบ local และผ่าน API สามารถพัฒนาต่อยอดเป็น Web UI ได้ในอนาคต

การใช้ Ollama เป็น local LLM สำหรับ embedding ช่วยลดการพึ่งพา external API และเพิ่มความเป็นส่วนตัวของข้อมูล ในขณะที่ยังรองรับการใช้ API จาก OpenAI, Anthropic, Gemini, และ OpenRouter ในกรณีที่ต้องการประสิทธิภาพที่สูงขึ้น

โค้ดและแนวทางที่นำเสนอในเอกสารนี้เป็นพื้นฐานสำหรับการพัฒนาระบบ GraphRAG ซึ่งสามารถนำไปต่อยอดและปรับแต่งเพิ่มเติมตามความต้องการเฉพาะได้ในอนาคต