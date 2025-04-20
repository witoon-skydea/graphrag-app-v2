#!/usr/bin/env python3
"""
ทดสอบสร้าง Knowledge Graph โดยใช้ไฟล์เอกสารจริง

ไฟล์ทดสอบที่ใช้:
1. ไฟล์ TXT: manual_skydea_cloudapi.txt - คู่มือการใช้งาน Skydea Cloud API
2. ไฟล์ DOCX: test_subject.docx
3. ไฟล์ PDF (ข้อความ): CV Visutthichai Busayapongpakdee_2024.pdf
4. ไฟล์ PDF (รูปภาพ): 20250324140235_001.pdf
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# เพิ่ม parent directory เข้าไปใน sys.path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# นำเข้าโมดูลจากโปรเจค
from src.document_processing.document_processor import DocumentProcessor
from src.document_processing.image_processor import ImageProcessor
from src.graphrag_engine.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.embedding.embedding_manager import EmbeddingManager

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("knowledge_graph_test.log")
    ]
)

logger = logging.getLogger("kg_test")

def test_setup():
    """ทดสอบการตั้งค่าเริ่มต้น"""
    logger.info("==== เริ่มการทดสอบการตั้งค่าเริ่มต้น ====")
    
    # ทดสอบการสร้าง EmbeddingManager ด้วย Ollama และโมเดล mxbai-embed-large
    try:
        embedding_manager = EmbeddingManager(
            embedding_source="ollama",
            model_name="mxbai-embed-large"
        )
        logger.info(f"สร้าง EmbeddingManager สำเร็จ: {embedding_manager}")
    except Exception as e:
        logger.error(f"ไม่สามารถสร้าง EmbeddingManager ได้: {e}")
        return None

    # ทดสอบการสร้าง Knowledge Graph Builder
    try:
        kg_builder = KnowledgeGraphBuilder(
            extraction_method="ollama",
            identification_method="ollama",
            model_name="mxbai-embed-large",
            similarity_threshold=0.85,
            confidence_threshold=0.5
        )
        logger.info(f"สร้าง KnowledgeGraphBuilder สำเร็จ: {kg_builder}")
        return kg_builder
    except Exception as e:
        logger.error(f"ไม่สามารถสร้าง KnowledgeGraphBuilder ได้: {e}")
        return None

def test_document_processing():
    """ทดสอบการประมวลผลเอกสาร"""
    logger.info("==== เริ่มการทดสอบการประมวลผลเอกสาร ====")
    
    # สร้าง DocumentProcessor และ ImageProcessor
    doc_processor = DocumentProcessor()
    img_processor = ImageProcessor(language="eng+tha")
    
    # กำหนดเส้นทางไฟล์ทดสอบ
    txt_file = "/Users/witoonpongsilathong/MCP_folder/mm_dev_mode/manual_skydea_cloudapi.txt"
    docx_file = "/Users/witoonpongsilathong/MCP_folder/test_subject.docx"
    pdf_text_file = "/Users/witoonpongsilathong/Library/CloudStorage/OneDrive-Personal/001one_2024/CV Visutthichai Busayapongpakdee_2024.pdf"
    pdf_image_file = "/Users/witoonpongsilathong/Downloads/20250324140235_001.pdf"
    
    results = {}
    
    # ทดสอบประมวลผลไฟล์ TXT
    if os.path.exists(txt_file):
        try:
            start_time = time.time()
            txt_content, txt_metadata = doc_processor.process_file(txt_file)
            processing_time = time.time() - start_time
            
            results["txt"] = {
                "status": "success",
                "file_path": txt_file,
                "content_length": len(txt_content),
                "metadata": txt_metadata,
                "processing_time": processing_time,
                "content": txt_content[:200] + "..." # แสดงเฉพาะส่วนต้นของเนื้อหา
            }
            logger.info(f"ประมวลผลไฟล์ TXT สำเร็จ: {len(txt_content)} อักขระ, ใช้เวลา {processing_time:.2f} วินาที")
        except Exception as e:
            logger.error(f"ไม่สามารถประมวลผลไฟล์ TXT ได้: {e}")
            results["txt"] = {"status": "error", "error": str(e)}
    else:
        logger.warning(f"ไม่พบไฟล์ TXT: {txt_file}")
        results["txt"] = {"status": "not_found"}
    
    # ทดสอบประมวลผลไฟล์ DOCX
    if os.path.exists(docx_file):
        try:
            start_time = time.time()
            docx_content, docx_metadata = doc_processor.process_file(docx_file)
            processing_time = time.time() - start_time
            
            results["docx"] = {
                "status": "success",
                "file_path": docx_file,
                "content_length": len(docx_content),
                "metadata": docx_metadata,
                "processing_time": processing_time,
                "content": docx_content[:200] + "..." if docx_content else "ไม่มีเนื้อหา"
            }
            logger.info(f"ประมวลผลไฟล์ DOCX สำเร็จ: {len(docx_content)} อักขระ, ใช้เวลา {processing_time:.2f} วินาที")
        except Exception as e:
            logger.error(f"ไม่สามารถประมวลผลไฟล์ DOCX ได้: {e}")
            results["docx"] = {"status": "error", "error": str(e)}
    else:
        logger.warning(f"ไม่พบไฟล์ DOCX: {docx_file}")
        results["docx"] = {"status": "not_found"}
    
    # ทดสอบประมวลผลไฟล์ PDF (ข้อความ)
    if os.path.exists(pdf_text_file):
        try:
            start_time = time.time()
            pdf_text_content, pdf_text_metadata = doc_processor.process_file(pdf_text_file)
            processing_time = time.time() - start_time
            
            results["pdf_text"] = {
                "status": "success",
                "file_path": pdf_text_file,
                "content_length": len(pdf_text_content),
                "metadata": pdf_text_metadata,
                "processing_time": processing_time,
                "content": pdf_text_content[:200] + "..." if pdf_text_content else "ไม่มีเนื้อหา"
            }
            logger.info(f"ประมวลผลไฟล์ PDF (ข้อความ) สำเร็จ: {len(pdf_text_content)} อักขระ, ใช้เวลา {processing_time:.2f} วินาที")
        except Exception as e:
            logger.error(f"ไม่สามารถประมวลผลไฟล์ PDF (ข้อความ) ได้: {e}")
            results["pdf_text"] = {"status": "error", "error": str(e)}
    else:
        logger.warning(f"ไม่พบไฟล์ PDF (ข้อความ): {pdf_text_file}")
        results["pdf_text"] = {"status": "not_found"}
    
    # ทดสอบประมวลผลไฟล์ PDF (รูปภาพ)
    if os.path.exists(pdf_image_file):
        try:
            start_time = time.time()
            # ประมวลผล PDF ปกติก่อน
            pdf_image_content, pdf_image_metadata = doc_processor.process_file(pdf_image_file)
            processing_time = time.time() - start_time
            
            # ทดสอบการดึงรูปภาพและ OCR ด้วย ImageProcessor
            ocr_start_time = time.time()
            try:
                extracted_images = img_processor.extract_images_from_pdf(pdf_image_file)
                ocr_processing_time = time.time() - ocr_start_time
                
                # แสดงข้อมูลการดึงรูปภาพ
                image_info = {
                    "image_count": len(extracted_images),
                    "ocr_processing_time": ocr_processing_time,
                    "image_details": []
                }
                
                for i, img_data in enumerate(extracted_images[:3]):  # แสดงเฉพาะ 3 รูปภาพแรก
                    image_info["image_details"].append({
                        "page": img_data["page_num"],
                        "index": img_data["image_idx"],
                        "text_length": len(img_data["extracted_text"]),
                        "sample_text": img_data["extracted_text"][:100] + "..." if img_data["extracted_text"] else "ไม่มีข้อความ"
                    })
            except Exception as e:
                image_info = {"error": str(e)}
            
            results["pdf_image"] = {
                "status": "success",
                "file_path": pdf_image_file,
                "content_length": len(pdf_image_content),
                "metadata": pdf_image_metadata,
                "processing_time": processing_time,
                "content": pdf_image_content[:200] + "..." if pdf_image_content else "ไม่มีเนื้อหา",
                "image_extraction": image_info
            }
            logger.info(f"ประมวลผลไฟล์ PDF (รูปภาพ) สำเร็จ: {len(pdf_image_content)} อักขระ, ใช้เวลา {processing_time:.2f} วินาที")
        except Exception as e:
            logger.error(f"ไม่สามารถประมวลผลไฟล์ PDF (รูปภาพ) ได้: {e}")
            results["pdf_image"] = {"status": "error", "error": str(e)}
    else:
        logger.warning(f"ไม่พบไฟล์ PDF (รูปภาพ): {pdf_image_file}")
        results["pdf_image"] = {"status": "not_found"}
    
    return results

def test_knowledge_graph_building(document_results):
    """ทดสอบการสร้าง Knowledge Graph จากเอกสาร"""
    logger.info("==== เริ่มการทดสอบการสร้าง Knowledge Graph ====")
    
    # สร้าง Knowledge Graph Builder
    kg_builder = KnowledgeGraphBuilder(
        extraction_method="ollama",
        identification_method="ollama",
        model_name="mxbai-embed-large",
        similarity_threshold=0.85,
        confidence_threshold=0.5,
        persist_path="kg_test_data"
    )
    
    # เตรียมเอกสารสำหรับประมวลผล
    documents = {}
    
    # เพิ่มเอกสารที่ประมวลผลสำเร็จเข้าไป
    for doc_type, result in document_results.items():
        if result["status"] == "success" and result.get("content"):
            doc_id = f"{doc_type}_{Path(result['file_path']).stem}"
            documents[doc_id] = result["content"]
    
    # ถ้าไม่มีเอกสารใดเลย ให้แจ้งและจบการทำงาน
    if not documents:
        logger.error("ไม่มีเอกสารที่ประมวลผลสำเร็จสำหรับสร้าง Knowledge Graph")
        return {"status": "error", "error": "ไม่มีเอกสารที่ประมวลผลสำเร็จ"}
    
    # ประมวลผลแต่ละเอกสารทีละไฟล์ก่อน
    individual_results = {}
    
    for doc_id, content in documents.items():
        try:
            logger.info(f"กำลังประมวลผลเอกสาร: {doc_id}")
            start_time = time.time()
            
            # ตัดเนื้อหาให้สั้นลงเพื่อการทดสอบเร็วขึ้น (10,000 อักขระแรก)
            if len(content) > 10000:
                logger.info(f"ตัดเนื้อหาของ {doc_id} จาก {len(content)} เป็น 10,000 อักขระ")
                content = content[:10000]
            
            result = kg_builder.process_document(
                document_text=content,
                document_id=doc_id
            )
            
            processing_time = time.time() - start_time
            
            individual_results[doc_id] = {
                "status": "success",
                "processing_time": processing_time,
                **result
            }
            
            logger.info(f"ประมวลผลเอกสาร {doc_id} สำเร็จ: พบ {result['entities_extracted']} entities, สร้าง {result['nodes_created']} nodes และ {result['edges_created']} edges, ใช้เวลา {processing_time:.2f} วินาที")
        except Exception as e:
            logger.error(f"ไม่สามารถประมวลผลเอกสาร {doc_id} ได้: {e}")
            individual_results[doc_id] = {"status": "error", "error": str(e)}
    
    # ทดสอบการค้นหา entities จากแต่ละประเภท
    entity_search = {}
    
    try:
        # ค้นหา entities ประเภท PERSON
        persons = kg_builder.query_entities(entity_type="PERSON", limit=10)
        entity_search["PERSON"] = {
            "count": len(persons),
            "samples": [{"text": node["text"], "id": node["id"], "documents": node.get("documents", [])} for node in persons[:5]]
        }
        
        # ค้นหา entities ประเภท ORGANIZATION
        organizations = kg_builder.query_entities(entity_type="ORGANIZATION", limit=10)
        entity_search["ORGANIZATION"] = {
            "count": len(organizations),
            "samples": [{"text": node["text"], "id": node["id"], "documents": node.get("documents", [])} for node in organizations[:5]]
        }
        
        # ค้นหา entities ประเภท URL
        urls = kg_builder.query_entities(entity_type="URL", limit=10)
        entity_search["URL"] = {
            "count": len(urls),
            "samples": [{"text": node["text"], "id": node["id"], "documents": node.get("documents", [])} for node in urls[:5]]
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดระหว่างการค้นหา entities: {e}")
        entity_search["error"] = str(e)
    
    # ทดสอบการประมวลผลเอกสารทั้งหมดพร้อมกัน (cross-document relationships)
    try:
        logger.info(f"กำลังประมวลผลทุกเอกสารเพื่อหาความสัมพันธ์ระหว่างเอกสาร")
        start_time = time.time()
        
        # สร้าง Knowledge Graph Builder ใหม่
        cross_kg_builder = KnowledgeGraphBuilder(
            extraction_method="ollama",
            identification_method="ollama",
            model_name="mxbai-embed-large",
            similarity_threshold=0.85,
            confidence_threshold=0.5,
            persist_path="kg_cross_doc_test"
        )
        
        # ประมวลผลเอกสารทุกไฟล์พร้อมกัน
        cross_doc_result = cross_kg_builder.process_documents(
            documents=documents,
            cross_document_relationships=True
        )
        
        processing_time = time.time() - start_time
        
        cross_doc_processing = {
            "status": "success",
            "processing_time": processing_time,
            **cross_doc_result
        }
        
        logger.info(f"ประมวลผลความสัมพันธ์ระหว่างเอกสารสำเร็จ: พบ {cross_doc_result['cross_document_relationships']} ความสัมพันธ์, สร้าง {cross_doc_result['cross_document_edges']} เส้นเชื่อม, ใช้เวลา {processing_time:.2f} วินาที")
        
        # ส่งออกเป็นรูปแบบ Network สำหรับการแสดงผล
        try:
            cross_kg_builder.export_to_network_format("kg_network_data.json")
            logger.info("ส่งออกข้อมูล Knowledge Graph เป็นรูปแบบเครือข่ายสำเร็จ")
        except Exception as e:
            logger.error(f"ไม่สามารถส่งออกข้อมูล Knowledge Graph ได้: {e}")
    except Exception as e:
        logger.error(f"ไม่สามารถประมวลผลความสัมพันธ์ระหว่างเอกสารได้: {e}")
        cross_doc_processing = {"status": "error", "error": str(e)}
    
    # รวมผลลัพธ์ทั้งหมด
    return {
        "individual_processing": individual_results,
        "entity_search": entity_search,
        "cross_document_processing": cross_doc_processing,
        "graph_statistics": kg_builder.graph_statistics()
    }

def run_test():
    """ดำเนินการทดสอบทั้งหมด"""
    # ทดสอบการตั้งค่าเริ่มต้น
    kg_builder = test_setup()
    if not kg_builder:
        logger.error("ไม่สามารถตั้งค่าได้ ยกเลิกการทดสอบ")
        return False
    
    # ทดสอบการประมวลผลเอกสาร
    document_results = test_document_processing()
    
    # ทดสอบการสร้าง Knowledge Graph
    kg_results = test_knowledge_graph_building(document_results)
    
    # บันทึกผลการทดสอบ
    test_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "document_processing": document_results,
        "knowledge_graph": kg_results
    }
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info("การทดสอบเสร็จสิ้น ผลลัพธ์ถูกบันทึกใน test_results.json")
    return True

def main():
    """ฟังก์ชันหลัก"""
    logger.info("เริ่มการทดสอบ GraphRAG Knowledge Graph Builder")
    run_test()

if __name__ == "__main__":
    main()
