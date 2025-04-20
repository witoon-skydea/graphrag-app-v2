# GraphRAG

GraphRAG คือระบบจัดการและค้นหาเอกสาร ที่ผสมผสานเทคโนโลยี Vector Database และ Knowledge Graph เพื่อเพิ่มประสิทธิภาพในการจัดเก็บและค้นหาข้อมูลจากเอกสารหลากหลายประเภท ทั้ง docx, txt, md, pdf, csv และ excel โดยรองรับทั้งข้อความและรูปภาพ

## สถาปัตยกรรมโดยรวม

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

## คุณสมบัติหลัก

- รองรับการนำเข้าเอกสารหลายประเภท (docx, txt, md, pdf, csv, excel)
- สกัดข้อความจากรูปภาพด้วย OCR
- สร้าง Knowledge Graph ที่เชื่อมโยงเอกสารและ entities
- ผสมผสานการค้นหาแบบ Vector Search และ Knowledge Graph
- รองรับการ Embedding ทั้งแบบ Local (Ollama) และ External API (OpenAI, Anthropic, Gemini)
- มี CLI สำหรับการใช้งานขั้นพื้นฐาน

## ความต้องการของระบบ

- Python 3.8+
- Weaviate (สำหรับ Vector Database)
- Ollama (สำหรับ Local Embedding)
- Tesseract OCR (สำหรับประมวลผล OCR)

## การติดตั้ง

1. โคลนโปรเจค:
   ```bash
   git clone https://github.com/yourusername/graphrag.git
   cd graphrag
   ```

2. ติดตั้ง dependency:
   ```bash
   pip install -r requirements.txt
   ```

3. ติดตั้ง Weaviate (ดูคำแนะนำการติดตั้งที่ [Weaviate Documentation](https://weaviate.io/developers/weaviate/installation))

4. ติดตั้ง Ollama (ถ้าต้องการใช้ Local Embedding):
   ```bash
   # macOS or Linux
   curl https://ollama.ai/install.sh | sh
   
   # ดาวน์โหลดโมเดล
   ollama pull llama2
   ```

5. ติดตั้ง Tesseract OCR:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu
   sudo apt install tesseract-ocr
   sudo apt install libtesseract-dev
   ```

## การใช้งาน

### การใช้งาน CLI

1. เริ่มต้นระบบ:
   ```bash
   python -m graphrag init --extraction-method ollama --id-method ollama --model-name llama2
   ```

2. นำเข้าเอกสาร:
   ```bash
   python -m graphrag import-file /path/to/document.pdf
   ```

3. นำเข้าเอกสารทั้งไดเรกทอรี:
   ```bash
   python -m graphrag import-directory /path/to/documents
   ```

4. ค้นหาเอกสาร:
   ```bash
   python -m graphrag search "คำค้นหาของคุณ"
   ```

5. ดูรายละเอียดเอกสาร:
   ```bash
   python -m graphrag get-document document_id
   ```

6. ดูรายละเอียด entity:
   ```bash
   python -m graphrag get-entity "entity_text"
   ```

7. ดูสถิติระบบ:
   ```bash
   python -m graphrag stats
   ```

8. ส่งออก Knowledge Graph:
   ```bash
   python -m graphrag export-kg knowledge_graph.json
   ```

## การใช้งานโปรแกรมเมอร์

```python
from graphrag_engine.graph_rag_engine import GraphRAGEngine

# สร้าง GraphRAG Engine
engine = GraphRAGEngine(
    extraction_method="ollama",
    identification_method="ollama",
    model_name="llama2",
    weaviate_url="http://localhost:8080"
)

# เริ่มต้นระบบ
engine.initialize()

# นำเข้าเอกสาร
result = engine.process_document(
    document_text="เนื้อหาเอกสาร",
    document_id="doc_1",
    title="ชื่อเอกสาร",
    document_type="text"
)

# ค้นหาเอกสาร
search_results = engine.search("คำค้นหา")

# ดึงข้อมูลเอกสาร
document = engine.get_document("doc_1")

# ดึงข้อมูล entity
entity = engine.get_entity("ชื่อ entity")
```

## ใบอนุญาต

โปรเจคนี้เผยแพร่ภายใต้ใบอนุญาต MIT

## ผู้พัฒนา

Witoon Pongsilathong
