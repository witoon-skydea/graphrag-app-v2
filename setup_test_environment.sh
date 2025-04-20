#!/bin/bash

# สคริปต์นี้ใช้ในการติดตั้งสภาพแวดล้อมสำหรับการทดสอบ GraphRAG

echo "=== เริ่มการตั้งค่าสภาพแวดล้อมสำหรับการทดสอบ GraphRAG ==="

# ตรวจสอบว่ามี pip หรือไม่
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
elif command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
else
    echo "ไม่พบ pip หรือ pip3 - กรุณาติดตั้ง Python และ pip ก่อน"
    exit 1
fi

# ตรวจสอบและติดตั้ง dependencies หลัก
echo "กำลังติดตั้ง dependencies หลัก..."
$PIP_CMD install numpy tqdm requests

# ตรวจสอบและติดตั้ง dependencies สำหรับการประมวลผลเอกสาร
echo "กำลังติดตั้ง dependencies สำหรับการประมวลผลเอกสาร..."
$PIP_CMD install python-docx PyPDF2 pymupdf pandas openpyxl

# ตรวจสอบและติดตั้ง dependencies สำหรับ OCR
echo "กำลังติดตั้ง dependencies สำหรับ OCR..."
$PIP_CMD install pytesseract Pillow

# ตรวจสอบ Tesseract OCR
if command -v tesseract &> /dev/null; then
    echo "Tesseract OCR ถูกติดตั้งแล้ว: $(tesseract --version | head -n 1)"
else
    echo "ไม่พบ Tesseract OCR - กรุณาติดตั้ง Tesseract OCR:"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  สำหรับ macOS: brew install tesseract"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  สำหรับ Ubuntu/Debian: sudo apt-get install tesseract-ocr"
        echo "  สำหรับ CentOS/RHEL: sudo yum install tesseract"
    elif [[ "$OSTYPE" == "msys"* ]]; then
        echo "  สำหรับ Windows: ดาวน์โหลดและติดตั้งจาก https://github.com/UB-Mannheim/tesseract/wiki"
    fi
fi

# ตรวจสอบ Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama ถูกติดตั้งแล้ว: $(ollama --version)"
    
    # ตรวจสอบว่า Ollama กำลังทำงานหรือไม่
    if curl -s http://localhost:11434/api/version &> /dev/null; then
        echo "Ollama กำลังทำงาน"
        
        # ตรวจสอบและดาวน์โหลดโมเดล mxbai-embed-large
        if ollama list | grep -q "mxbai-embed-large"; then
            echo "โมเดล mxbai-embed-large มีอยู่แล้ว"
        else
            echo "กำลังดาวน์โหลดโมเดล mxbai-embed-large..."
            ollama pull mxbai-embed-large
        fi
    else
        echo "Ollama ยังไม่ได้ทำงาน - กรุณาเริ่มต้น Ollama ด้วยคำสั่ง: ollama serve"
    fi
else
    echo "ไม่พบ Ollama - กรุณาติดตั้ง Ollama:"
    echo "  ดูคำแนะนำการติดตั้งได้ที่: https://ollama.ai/download"
fi

# ตรวจสอบและติดตั้ง dependencies สำหรับ Weaviate (ตัวเลือก)
echo "กำลังติดตั้ง Weaviate client..."
$PIP_CMD install weaviate-client

# อัปเดต requirements.txt
echo "กำลังอัปเดตไฟล์ requirements.txt..."
cat > requirements.txt << EOL
numpy>=1.20.0
tqdm>=4.60.0
requests>=2.25.1
python-docx>=0.8.11
PyPDF2>=2.10.0
pymupdf>=1.19.0
pytesseract>=0.3.8
Pillow>=8.2.0
pandas>=1.3.0
openpyxl>=3.0.7
weaviate-client>=3.0.0
EOL

echo "=== ตั้งค่าสภาพแวดล้อมเสร็จสิ้น ==="
echo "คุณสามารถเริ่มทดสอบด้วยคำสั่ง: python test_kg_with_real_docs.py"
