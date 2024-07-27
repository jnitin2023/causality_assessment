from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import re
import spacy
from typing import Dict, List

router = APIRouter()

nlp = spacy.load("en_core_web_sm")

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

def parse_text_with_ner(text: str) -> List[Dict[str, str]]:
    keys = ["Name", "Gender", "Address", "Phone", "Contact Phone", "Fax", "E-Mail", 
            "Class", "Misc", "Type", "Greeting", "Salutation", "Degree", "Specialty", 
            "National Identifier"]

    contact_section = extract_section(text, "Contacts", "\nCase\nInformation")
    doc = nlp(contact_section)
    
    # Initialize the pattern matcher
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    
    patterns = [[{"LOWER": key.lower()}] for key in keys]
    matcher.add("KEYS", patterns)
    
    data_list = []
    current_dict = {}
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]
        key = span.text.strip()
        next_line_start = end
        next_key_start = len(doc) if len(matches) == 0 else matches[0][1]  # Next key position or end of doc
        
        if key in keys:
            if current_dict and "National Identifier" in current_dict:
                data_list.append(current_dict)
                current_dict = {}
            current_dict[key] = ''
            while next_line_start < next_key_start:
                next_line = doc[next_line_start].text.strip()
                if not next_line:
                    break
                current_dict[key] += " " + next_line
                next_line_start += 1
    
    if current_dict:
        data_list.append(current_dict)
    
    return data_list

def identify_patterns(text: str) -> Dict[str, str]:
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

@router.post("/upload/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)  
    # contact_section = extract_section(text, "Contacts", "\nCase\nInformation")

    data_dict = parse_text_to_dict(text)
    nlp_entities = parse_text_with_ner(text)
    
    return JSONResponse(content={ "data_dict": data_dict, "nlp_entities": nlp_entities})


