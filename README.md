# GraphRAG - Knowledge Graph Builder for Document Processing

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

## คู่มือการใช้งานทีละขั้นตอน

### 1. การติดตั้งและการตั้งค่า

#### 1.1 ติดตั้ง Dependencies

การติดตั้ง dependencies ทั้งหมดสามารถทำได้ด้วยสคริปต์ `setup_test_environment.sh`:

```bash
# ให้สิทธิ์การทำงานกับไฟล์
chmod +x setup_test_environment.sh

# รันสคริปต์
./setup_test_environment.sh
```

หรือติดตั้งแต่ละส่วนด้วยตนเอง:

```bash
# ติดตั้ง Python dependencies
pip install -r requirements.txt

# ติดตั้ง Tesseract OCR (สำหรับ macOS)
brew install tesseract

# ติดตั้ง Tesseract OCR (สำหรับ Ubuntu)
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

#### 1.2 ติดตั้งและตั้งค่า Ollama

Ollama เป็นเครื่องมือสำหรับรัน LLM แบบ local ซึ่งใช้สำหรับสร้าง embeddings ในโปรเจคนี้

1. ติดตั้ง Ollama:
   ```bash
   # macOS หรือ Linux
   curl https://ollama.ai/install.sh | sh
   
   # Windows: ดาวน์โหลดและติดตั้งจาก https://ollama.ai/download
   ```

2. เริ่มการทำงานของ Ollama:
   ```bash
   ollama serve
   ```

3. ดาวน์โหลดโมเดล mxbai-embed-large สำหรับ embeddings:
   ```bash
   ollama pull mxbai-embed-large
   ```

#### 1.3 ติดตั้งและตั้งค่า Weaviate

Weaviate เป็น vector database ที่ใช้จัดเก็บและค้นหาข้อมูล

1. ติดตั้ง Weaviate โดยใช้ Docker:
   ```bash
   docker compose up -d
   ```
   (ต้องมีไฟล์ docker-compose.yml ที่กำหนดค่า Weaviate)

2. หรือติดตั้งโดยใช้ Docker โดยตรง:
   ```bash
   docker run -d -p 8080:8080 --name weaviate \
     -e QUERY_DEFAULTS_LIMIT=25 \
     -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
     -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
     -v weaviate_data:/var/lib/weaviate \
     semitechnologies/weaviate:1.19.6
   ```

### 2. การใช้งานระบบ GraphRAG

#### 2.1 เริ่มการทำงานของระบบ

การใช้งาน GraphRAG สามารถทำได้ผ่าน Python script หรือ Command Line Interface (CLI)

##### ผ่าน Python script:

สร้างไฟล์ Python เพื่อเริ่มการทำงานของระบบ:

```python
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.document_processing.document_processor import DocumentProcessor

# สร้าง Knowledge Graph Builder
kg_builder = KnowledgeGraphBuilder(
    extraction_method="ollama",          # วิธีการสกัด entities
    identification_method="ollama",      # วิธีการระบุความสัมพันธ์
    model_name="mxbai-embed-large",      # โมเดลที่ใช้
    similarity_threshold=0.85,           # ค่าความคล้ายคลึงสำหรับการจับคู่ entities
    confidence_threshold=0.5,            # ค่าความมั่นใจสำหรับความสัมพันธ์
    persist_path="my_knowledge_graph"    # เส้นทางสำหรับบันทึกข้อมูล
)

# สร้าง Document Processor
doc_processor = DocumentProcessor()

print("ระบบ GraphRAG พร้อมใช้งานแล้ว")
```

##### ผ่าน CLI:

```bash
python -m graphrag init --extraction-method ollama --id-method ollama --model-name mxbai-embed-large
```

#### 2.2 การประมวลผลเอกสาร

##### การประมวลผลไฟล์เอกสารเดี่ยวผ่าน Python script:

```python
# ประมวลผลเอกสารด้วย Document Processor
file_path = "เส้นทางไฟล์ของคุณ/เอกสาร.txt"
content, metadata = doc_processor.process_file(file_path)

# สร้าง Knowledge Graph จากเอกสาร
document_id = "doc_" + metadata.get("file_type", "unknown").lower()
result = kg_builder.process_document(
    document_text=content,
    document_id=document_id
)

print(f"ประมวลผลเอกสารเสร็จสิ้น: พบ {result['entities_extracted']} entities, "
      f"สร้าง {result['nodes_created']} nodes และ {result['edges_created']} edges")
```

##### การประมวลผลหลายเอกสารพร้อมกัน:

```python
# รวบรวมข้อมูลเอกสาร
documents = {}

# ประมวลผลไฟล์แรก
file1_path = "เส้นทางไฟล์ของคุณ/เอกสาร1.txt"
content1, metadata1 = doc_processor.process_file(file1_path)
documents["doc_txt_1"] = content1

# ประมวลผลไฟล์ที่สอง
file2_path = "เส้นทางไฟล์ของคุณ/เอกสาร2.pdf"
content2, metadata2 = doc_processor.process_file(file2_path)
documents["doc_pdf_2"] = content2

# ประมวลผลเอกสารทั้งหมดพร้อมกัน เพื่อหาความสัมพันธ์ระหว่างเอกสาร
result = kg_builder.process_documents(
    documents=documents,
    cross_document_relationships=True  # เปิดใช้งานการค้นหาความสัมพันธ์ระหว่างเอกสาร
)

print(f"ประมวลผลเอกสารทั้งหมดเสร็จสิ้น:")
print(f"- พบ {result['entities_extracted']} entities ทั้งหมด")
print(f"- สร้าง {result['nodes_created']} nodes ทั้งหมด")
print(f"- ระบุ {result['relationships_identified']} ความสัมพันธ์")
print(f"- สร้าง {result['edges_created']} เส้นเชื่อมทั้งหมด")
print(f"- พบ {result['cross_document_relationships']} ความสัมพันธ์ระหว่างเอกสาร")
```

##### การประมวลผลไฟล์เอกสารผ่าน CLI:

```bash
# ประมวลผลไฟล์เดี่ยว
python -m graphrag import-file /path/to/document.pdf

# ประมวลผลทั้งไดเรกทอรี
python -m graphrag import-directory /path/to/documents
```

#### 2.3 การสืบค้นและค้นหาข้อมูล

##### การค้นหา entities:

```python
# ค้นหา entities ประเภท PERSON
persons = kg_builder.query_entities(entity_type="PERSON", limit=10)
print(f"พบ {len(persons)} entities ประเภท PERSON:")
for i, person in enumerate(persons, 1):
    print(f"{i}. {person['text']} (ID: {person['id']})")

# ค้นหา entities โดยใช้ข้อความ
search_text = "ชื่อที่ต้องการค้นหา"
entities = kg_builder.query_entities(entity_text=search_text, limit=5)
print(f"ผลการค้นหาสำหรับ '{search_text}':")
for entity in entities:
    print(f"- {entity['text']} (ประเภท: {entity['type']})")
```

##### การค้นหาความสัมพันธ์:

```python
# ค้นหาความสัมพันธ์ของ entity
entity_text = "ชื่อ entity ที่ต้องการ"
relationships = kg_builder.get_entity_relationships(
    entity_text=entity_text,
    include_incoming=True,
    include_outgoing=True
)

# แสดงความสัมพันธ์ขาเข้า
print(f"ความสัมพันธ์ขาเข้าของ '{entity_text}':")
for rel in relationships["incoming"]:
    print(f"- {rel['source_text']} ({rel['source_type']}) {rel['type']} {entity_text}")

# แสดงความสัมพันธ์ขาออก
print(f"ความสัมพันธ์ขาออกของ '{entity_text}':")
for rel in relationships["outgoing"]:
    print(f"- {entity_text} {rel['type']} {rel['target_text']} ({rel['target_type']})")

# ค้นหาเส้นทางระหว่าง entities
source_text = "entity ต้นทาง"
target_text = "entity ปลายทาง"
paths = kg_builder.get_paths_between_entities(
    source_text=source_text,
    target_text=target_text,
    max_depth=3  # ความลึกสูงสุดของเส้นทาง
)

print(f"เส้นทางระหว่าง '{source_text}' ไปยัง '{target_text}':")
for i, path in enumerate(paths, 1):
    print(f"เส้นทางที่ {i}:")
    for edge in path:
        print(f"  {edge['source_text']} --[{edge['type']}]--> {edge['target_text']}")
```

##### การค้นหาผ่าน CLI:

```bash
# ค้นหาเอกสาร
python -m graphrag search "คำค้นหาของคุณ"

# ดูรายละเอียด entity
python -m graphrag get-entity "ชื่อ entity"

# ดูเอกสารที่เกี่ยวข้องกับ entity
python -m graphrag get-related-documents "ชื่อ entity"
```

#### 2.4 การวิเคราะห์และแสดงผล Knowledge Graph

##### การดูสถิติของ Knowledge Graph:

```python
# ดูสถิติของ Knowledge Graph
stats = kg_builder.graph_statistics()

print("สถิติของ Knowledge Graph:")
print(f"- จำนวน nodes ทั้งหมด: {stats['total_nodes']}")
print(f"- จำนวน edges ทั้งหมด: {stats['total_edges']}")
print(f"- จำนวนเอกสารที่ประมวลผล: {stats['total_documents']}")

# แสดงจำนวน nodes แยกตามประเภท
print("\nจำนวน nodes แยกตามประเภท:")
for node_type, count in stats['node_types'].items():
    print(f"- {node_type}: {count}")

# แสดงจำนวน edges แยกตามประเภท
print("\nจำนวน edges แยกตามประเภท:")
for edge_type, count in stats['edge_types'].items():
    print(f"- {edge_type}: {count}")
```

##### การส่งออก Knowledge Graph เพื่อการแสดงผล:

```python
# ส่งออก Knowledge Graph เป็นรูปแบบเครือข่าย
output_file = "knowledge_graph_network.json"
kg_builder.export_to_network_format(output_file)
print(f"ส่งออก Knowledge Graph ไปยัง {output_file} เรียบร้อยแล้ว")
```

##### การบันทึกและโหลด Knowledge Graph:

```python
# บันทึก Knowledge Graph
kg_builder.persist_graph()
print(f"บันทึก Knowledge Graph ไปยัง {kg_builder.persist_path} เรียบร้อยแล้ว")

# โหลด Knowledge Graph จากที่บันทึกไว้
new_kg_builder = KnowledgeGraphBuilder(
    extraction_method="ollama",
    identification_method="ollama",
    model_name="mxbai-embed-large",
    persist_path="my_knowledge_graph"
)
new_kg_builder.load_graph()
print(f"โหลด Knowledge Graph จาก {new_kg_builder.persist_path} เรียบร้อยแล้ว")
```

### 3. ตัวอย่างการใช้งานเต็มรูปแบบ

ตัวอย่างสคริปต์ที่ครบถ้วนสำหรับการประมวลผลเอกสารและสร้าง Knowledge Graph:

```python
import os
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.document_processing.document_processor import DocumentProcessor

# สร้าง Document Processor และ Knowledge Graph Builder
doc_processor = DocumentProcessor()
kg_builder = KnowledgeGraphBuilder(
    extraction_method="ollama",
    identification_method="ollama",
    model_name="mxbai-embed-large",
    similarity_threshold=0.85,
    confidence_threshold=0.5,
    persist_path="example_kg"
)

# กำหนดไดเรกทอรีที่มีเอกสาร
documents_dir = "/path/to/your/documents"
document_files = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]

# ประมวลผลเอกสารแต่ละไฟล์
print(f"พบ {len(document_files)} ไฟล์ในไดเรกทอรี")
documents = {}

for i, file_name in enumerate(document_files, 1):
    file_path = os.path.join(documents_dir, file_name)
    print(f"[{i}/{len(document_files)}] กำลังประมวลผลไฟล์: {file_name}")
    
    try:
        # ประมวลผลเอกสาร
        content, metadata = doc_processor.process_file(file_path)
        
        # เพิ่มเข้าใน documents dictionary
        doc_id = f"doc_{i}_{os.path.splitext(file_name)[0]}"
        documents[doc_id] = content
        
        print(f"  - ประมวลผลสำเร็จ: {len(content)} อักขระ")
    except Exception as e:
        print(f"  - เกิดข้อผิดพลาด: {e}")

# ประมวลผลเอกสารทั้งหมดเพื่อสร้าง Knowledge Graph
print("\nกำลังสร้าง Knowledge Graph จากเอกสารทั้งหมด...")
result = kg_builder.process_documents(
    documents=documents,
    cross_document_relationships=True
)

# แสดงผลลัพธ์
print("\nผลการสร้าง Knowledge Graph:")
print(f"- ประมวลผล {result['documents_processed']} เอกสาร")
print(f"- พบ {result['entities_extracted']} entities")
print(f"- สร้าง {result['nodes_created']} nodes")
print(f"- ระบุ {result['relationships_identified']} ความสัมพันธ์")
print(f"- สร้าง {result['edges_created']} edges")
print(f"- พบ {result['cross_document_relationships']} ความสัมพันธ์ระหว่างเอกสาร")

# บันทึก Knowledge Graph
kg_builder.persist_graph()
print(f"\nบันทึก Knowledge Graph ไปยัง {kg_builder.persist_path} เรียบร้อยแล้ว")

# ส่งออกเป็นรูปแบบเครือข่ายสำหรับการแสดงผล
kg_builder.export_to_network_format("knowledge_graph_network.json")
print("ส่งออก Knowledge Graph เป็นรูปแบบเครือข่ายเรียบร้อยแล้ว")

# แสดงสถิติของ Knowledge Graph
stats = kg_builder.graph_statistics()
print("\nสถิติของ Knowledge Graph:")
print(f"- จำนวน nodes ทั้งหมด: {stats['total_nodes']}")
print(f"- จำนวน edges ทั้งหมด: {stats['total_edges']}")

# แสดงประเภทของ nodes ที่พบมากที่สุด 5 อันดับแรก
print("\nประเภทของ nodes ที่พบมากที่สุด:")
node_types_sorted = sorted(stats['node_types'].items(), key=lambda x: x[1], reverse=True)[:5]
for node_type, count in node_types_sorted:
    print(f"- {node_type}: {count}")

# แสดงประเภทของ edges ที่พบมากที่สุด 5 อันดับแรก
print("\nประเภทของความสัมพันธ์ที่พบมากที่สุด:")
edge_types_sorted = sorted(stats['edge_types'].items(), key=lambda x: x[1], reverse=True)[:5]
for edge_type, count in edge_types_sorted:
    print(f"- {edge_type}: {count}")
```

### 4. การแก้ไขปัญหาที่พบบ่อย

#### 4.1 ปัญหาเกี่ยวกับ Ollama

```
ปัญหา: ไม่สามารถเชื่อมต่อกับ Ollama API ได้
สาเหตุ: Ollama อาจไม่ได้ทำงานหรือไม่ได้ติดตั้ง
การแก้ไข: 
1. ตรวจสอบว่า Ollama ทำงานอยู่หรือไม่ด้วยคำสั่ง:
   $ ollama list
2. หากไม่ทำงาน ให้เริ่ม Ollama:
   $ ollama serve
3. ตรวจสอบว่าโมเดล mxbai-embed-large ถูกติดตั้งหรือไม่:
   $ ollama list | grep mxbai-embed-large
4. หากไม่พบ ให้ดาวน์โหลดโมเดล:
   $ ollama pull mxbai-embed-large
```

#### 4.2 ปัญหาเกี่ยวกับ Dependencies

```
ปัญหา: ModuleNotFoundError: No module named 'xxx'
สาเหตุ: ไม่ได้ติดตั้ง package ที่จำเป็น
การแก้ไข:
1. ติดตั้ง package ที่จำเป็น:
   $ pip install xxx
2. หรือรันสคริปต์ setup_test_environment.sh:
   $ ./setup_test_environment.sh
```

#### 4.3 ปัญหาเกี่ยวกับ OCR

```
ปัญหา: ไม่สามารถทำ OCR ได้
สาเหตุ: ไม่ได้ติดตั้ง Tesseract OCR หรือติดตั้งไม่ถูกต้อง
การแก้ไข:
1. ติดตั้ง Tesseract OCR:
   # macOS
   $ brew install tesseract
   
   # Ubuntu
   $ sudo apt install tesseract-ocr
   $ sudo apt install libtesseract-dev
   
2. ตรวจสอบการติดตั้ง:
   $ tesseract --version
```

#### 4.4 ปัญหาเกี่ยวกับ Weaviate

```
ปัญหา: ไม่สามารถเชื่อมต่อกับ Weaviate ได้
สาเหตุ: Weaviate อาจไม่ได้ทำงานหรือไม่ได้ติดตั้ง
การแก้ไข:
1. ตรวจสอบว่า Weaviate container ทำงานอยู่หรือไม่:
   $ docker ps | grep weaviate
2. หากไม่ทำงาน ให้เริ่ม Weaviate:
   $ docker run -d -p 8080:8080 --name weaviate \
     -e QUERY_DEFAULTS_LIMIT=25 \
     -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
     -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
     -v weaviate_data:/var/lib/weaviate \
     semitechnologies/weaviate:1.19.6
```

## ความต้องการของระบบ

- Python 3.8+
- Weaviate (สำหรับ Vector Database)
- Ollama (สำหรับ Local Embedding)
- Tesseract OCR (สำหรับประมวลผล OCR)
- Docker (สำหรับรัน Weaviate)

## ใบอนุญาต

โปรเจคนี้เผยแพร่ภายใต้ใบอนุญาต MIT

## ผู้พัฒนา

Witoon Pongsilathong