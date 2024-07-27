from typing import Any, Dict
import PyPDF2
from fastapi import APIRouter, File, UploadFile, HTTPException
import io
import re

from pydantic import BaseModel

router = APIRouter()

class ContactInfo(BaseModel):
    primary: Dict[str, Any]
    secondary: Dict[str, Any]

def parse_pdf_content(content: bytes) -> str:
    pdf_file = io.BytesIO(content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_contact_info(text: str) -> ContactInfo:
    # Split the text into sections
    sections = re.split(r'\n\s*\n', text)
    
    # Function to process each section
    def process_section(section: str) -> Dict[str, Any]:
        lines = section.split('\n')
        info = {}
        current_key = None
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                info[key] = value
                current_key = key
            elif current_key:
                # Append to previous value if it's a continuation
                info[current_key] += ' ' + line.strip()
        return info

    # Process primary and secondary contacts
    primary_info = {}
    secondary_info = {}
    
    for section in sections:
        if 'Primary' in section:
            primary_info = process_section(section)
        elif 'Name' in section and not primary_info:
            primary_info = process_section(section)
        elif 'Name' in section and primary_info:
            secondary_info = process_section(section)

    return ContactInfo(primary=primary_info, secondary=secondary_info)

@router.post("/parse-pdf-pypdf")
async def parse_pdf_pypdf(file: UploadFile = File(...)):
    content = await file.read()
    text = parse_pdf_content(content)
    print(text)
    # contact_info = extract_contact_info(text)
    return text

