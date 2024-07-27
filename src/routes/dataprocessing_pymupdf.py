# main.py

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import re
from typing import Dict

router = APIRouter()

def extract_text_from_pdf(file):
    text = ""
    pdf_document = fitz.open("pdf", file)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

def extract_section(text: str, start_word: str, end_word: str) -> str:
    pattern = re.compile(f'{start_word}(.*?){end_word}', re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def parse_text_to_dict(text: str) -> Dict[str, str]:
    keys = ["Name", "Gender", "Address", "Phone", "Contact Phone", "Fax", "E-Mail", 
            "Class", "Misc", "Type", "Greeting", "Salutation", "Degree", "Specialty", 
            "National Identifier"]
    
    data_list = []
    current_dict = {}
    current_key = None
    contact_section = extract_section(text, "Contacts", "\nCase\nInformation")
    lines = contact_section.splitlines()
    for line in lines:
        # Check if the line starts with a key
        key_match = re.match(r"^(" + "|".join(keys) + r")\b", line)
        if key_match:
            current_key = key_match.group(0)
            value = line[len(current_key):].strip()
            if current_key == "Name" and current_dict:
                data_list.append(current_dict)
                current_dict = {}
            current_dict[current_key] = value
            # if current_key == "National Identifier":
            #     data_list.append(current_dict)
            #     current_dict = {}
        elif current_key:
            current_dict[current_key] += " " + line.strip()

    if current_dict:
        data_list.append(current_dict)
    
    return data_list

    # data_dict = {}
    # current_key = None
    # contact_section = extract_section(text, "Contacts", "\nCase\nInformation")
    # print(contact_section)
    # # text = re.findall(r'Contacts[\s\S]+?Case\sInformation', text)
    # lines = contact_section.splitlines()
    # for line in lines:
    #     # Check if the line starts with a key
    #     key_match = re.match(r"^(" + "|".join(keys) + r")\b", line)
    #     if key_match:
    #         current_key = key_match.group(0)
    #         value = line[len(current_key):].strip()
    #         data_dict[current_key] = value
    #     elif current_key:
    #         # Append to the current key's value if it spans multiple lines
    #         data_dict[current_key] += " " + line.strip()

    # return data_dict

@router.post("/extract_pdf_data/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)    
    # print(text)
    data_dict = parse_text_to_dict(text)
    
    return JSONResponse(content=data_dict)


