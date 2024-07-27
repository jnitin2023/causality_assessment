from typing import Any, Dict
from fastapi import APIRouter, File, UploadFile, HTTPException
import camelot.io as camelot
import io
import tempfile
import os

from pydantic import BaseModel

router = APIRouter()

class ContactInfo(BaseModel):
    name: str
    address: str
    phone: str
    email: str

def parse_pdf_with_camelot(content: bytes) -> ContactInfo:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        tables = camelot.read_pdf(temp_file_path, pages='1', flavor='stream')
        
        if tables and len(tables) > 0:
            df = tables[0].df
            
            name = df.iloc[0, 1] if df.shape[0] > 0 and df.shape[1] > 1 else ""
            address = df.iloc[1, 1] if df.shape[0] > 1 and df.shape[1] > 1 else ""
            phone = df.iloc[2, 1] if df.shape[0] > 2 and df.shape[1] > 1 else ""
            email = df.iloc[3, 1] if df.shape[0] > 3 and df.shape[1] > 1 else ""

            return ContactInfo(
                name=name,
                address=address,
                phone=phone,
                email=email
            )
        else:
            return ContactInfo(name="", address="", phone="", email="")
    finally:
        os.unlink(temp_file_path)

@router.post("/parse-pdf/")
async def parse_pdf(file: UploadFile = File(...)):
    content = await file.read()
    # contact_info = parse_pdf_with_camelot(content)
    return content

