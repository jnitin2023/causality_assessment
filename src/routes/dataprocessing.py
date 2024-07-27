
from fastapi import APIRouter, File, HTTPException, UploadFile
import pdfplumber
import io
import re

router = APIRouter()

def extract_info(text):
    # Adjust patterns to handle multiline addresses
    name_pattern = re.compile(r"Name\s+([A-Za-z ]+)")
    address_pattern = re.compile(r"Address\s+((?:[A-Za-z0-9 ,.-]+\n?)+)")
    case_type_pattern = re.compile(r"Case Type\s+([A-Za-z ]+)")

    names = name_pattern.findall(text)
    addresses = address_pattern.findall(text)
    case_types = case_type_pattern.findall(text)

    # Remove extra newline characters from addresses
    addresses = [address.replace('\n', ' ').strip() for address in addresses]

    return {"Name": names, "Address": addresses, "Case Type": case_types}

@router.post("/read_pdf")
async def read_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    try:
        pdf_file = io.BytesIO(await file.read())
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
        print(text)
        # extracted_info = extract_info(text)
        return {"filename": file.filename, "extracted_info": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





