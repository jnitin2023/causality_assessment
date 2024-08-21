from fastapi import APIRouter, HTTPException
from transformers import pipeline, BertTokenizer, BertForQuestionAnswering
from pydantic import BaseModel
from openai import OpenAI
import sacremoses
import torch
import os
import re

router = APIRouter()
# Set your OpenAI API key
# openai.api_key = ''

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

print('openai_api_key: ',openai_api_key)
client = OpenAI(
  api_key=openai_api_key,
)


# Define a Pydantic model for the input
class Report(BaseModel):
    report_text: str

@router.post("/assessText")
async def assess(report: Report):
    print(report.report_text)

# Define a route for the causality assessment
@router.post("/assess_with_gpt4o")
async def assess(report: str):
    try:
        # print(report)
        response = client.chat.completions.create(
           model="gpt-4o",
           messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Assess the causality for the following AE report:\n\n{report}\n\nAssessment:"}
            ]
        )
        assessment_text = response.choices[0].message.content
        print(assessment_text)
        # Use regex to find headers and their corresponding text
        pattern = re.compile(r"(\*\*.+?\*\*):?\s*(.+?)(?=(\*\*|$))", re.DOTALL)
        matches = pattern.findall(assessment_text)
        
        result = {}
        for match in matches:
            header = match[0].strip('*').strip()
            text = match[1].strip()
            result[header] = text
        
        if not result:
            raise ValueError("Assessment is incomplete or not properly formatted.")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/assess_with_gpt4o_mini")
async def assess(report: str):
    try:
        # print(report)
        response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Assess the causality for the following AE report:\n\n{report}\n\nAssessment:"}
            ]
        )
        assessment_text = response.choices[0].message.content
        print(assessment_text)
        # Use regex to find headers and their corresponding text
        pattern = re.compile(r"(\*\*.+?\*\*):?\s*(.+?)(?=(\*\*|$))", re.DOTALL)
        matches = pattern.findall(assessment_text)
        
        result = {}
        for match in matches:
            header = match[0].strip('*').strip()
            text = match[1].strip()
            result[header] = text
        
        if not result:
            raise ValueError("Assessment is incomplete or not properly formatted.")
        
        return result

        # Initialize the dictionary to store the structured data
        # assessment_data = {}
        # current_header = None
        # current_sub_header = None
        
        # # Split the response into lines
        # lines = assessment_text.split("\n")
        
        # for line in lines:
        #     line = line.strip()
            
        #     if line.startswith("###"):
        #         # Extract header by stripping the "###" and leading/trailing spaces
        #         current_header = line.strip("#").strip()
        #         assessment_data[current_header] = {}
        #         current_sub_header = None
        #     elif line.startswith("**"):
        #         # Extract sub-header by stripping the "**" and leading/trailing spaces
        #         if current_header:
        #             current_sub_header = line.strip("*").strip()
        #             assessment_data[current_header][current_sub_header] = ""
        #     elif current_sub_header:
        #         # Append line to the current sub-header's value
        #         assessment_data[current_header][current_sub_header] += (line + " ")
        #     elif current_header:
        #         # Append line to the current header's value if no sub-header is present
        #         if current_sub_header is None:
        #             if "Text" not in assessment_data[current_header]:
        #                 assessment_data[current_header]["Text"] = ""
        #             assessment_data[current_header]["Text"] += (line + " ")

        # # Clean up extra spaces
        # for header in assessment_data:
        #     for sub_header in assessment_data[header]:
        #         assessment_data[header][sub_header] = assessment_data[header][sub_header].strip()

        # return assessment_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def segment_text(text):
    segments = {}
    # segments['patient_info'] = text.split('Patient Information:')[1].split('Symptoms:')[0].strip()
    segments['narrative'] = text.split('Narrative:')[1].strip()
    return segments

qa_pipeline = pipeline('question-answering', model='dmis-lab/biobert-base-cased-v1.1')

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

@router.post("/assess_with_bioBERT")
async def assess(report: str):
    segments = segment_text(report)
    question = "What is the patient's diagnosis?"
    answer = answer_question(question, segments['narrative'])
    print(answer)
    return answer

# Load pre-trained BioBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def answer_question_biobert(question, context):
    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][start_index:end_index]))
        return answer
    
class Query(BaseModel):
    question: str
    context: str

@router.post("/ask_biobert")
def ask_question(query: Query):
    try:
        print(f"Received question: {query.question}")
        print(f"Received context: {query.context}")
        answer = answer_question_biobert(query.question, query.context)
        
        return {"question": query.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------BioGPT Example --------------------------------------

from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import set_seed
import json

@router.post("/ask_bioGPT_large")
def get_biogpt_response_large(query: Query):
    # Load the model and tokenizer 
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt-large')
    model = BioGptForCausalLM.from_pretrained('microsoft/biogpt-large')
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    set_seed(42)
    
    # Format the input text for the model
    # input_text = f"Context: {context} Question: {question}"
    input_text= f""" question: {query.question}
                    context: {query.context}
                    answer: the answer to the question given the context is """
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024*2)
    
    # generated_data= generator(input_text, max_new_tokens=500, num_return_sequences=1, do_sample=True)
    generated_data = model.generate(
            inputs['input_ids'], 
            max_new_tokens=500, 
            num_return_sequences=5, 
            do_sample=True,      # Enable sampling for more diverse output
            top_k=50,            # Use top_k sampling
            top_p=0.95,          # Or use nucleus sampling with top_p
            temperature=0.7      # Control the randomness of predictions    
        )
    
    # print(generated_data)
    # parsed_data = json.loads(generated_text)
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True).split("answer:")[1].strip()
        for output in generated_data
    ]

    # Return the list of generated answers
    print(generated_texts)

    
    # Generate the response using the model
    # outputs = model.generate(inputs['input_ids'], max_new_tokens=500, num_return_sequences=1, return_dict_in_generate=True)
    
    # # Print generated token IDs and their corresponding text
    # print("Generated Token IDs:", outputs.sequences)
    # print("Generated Token Text:", tokenizer.convert_ids_to_tokens(outputs.sequences[0]))
    
    # # Decode the response and return it
    # response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return generated_texts

@router.post("/ask_bioGPT")
def get_biogpt_response(query: Query):
    # Load the tokenizer and model
    tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
    model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')

    # Optionally set a random seed for reproducibility
    set_seed(42)

    # Define the input prompt
    input_text = (
        f"context: {query.context}\n"
        f"question: {query.question}\n"
        f"answer: "
    )

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)

    # Generate the response
    with torch.no_grad():
        generated_data = model.generate(
            inputs['input_ids'], 
            max_new_tokens=200, 
            num_return_sequences=5, 
            do_sample=True,      # Enable sampling for more diverse output
            top_k=50,            # Use top_k sampling
            top_p=0.95,          # Or use nucleus sampling with top_p
            temperature=0.7      # Control the randomness of predictions    
        )
    # Decode the generated texts and collect them in a list
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True).split("answer:")[1].strip()
        for output in generated_data
    ]

    # Return the list of generated answers
    print(generated_texts)

    return generated_texts