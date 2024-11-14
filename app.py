from fastapi import FastAPI, HTTPException, File, UploadFile, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import pandas as pd
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from typing import List
import time
import random
import httpx
import asyncio
import shutil
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import date
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from passlib.context import CryptContext
import secrets
import time
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
import uuid

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows selected origins only
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


API_KEY = os.environ.get("OPENAI_API_KEY")
subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
endpoint = "https://my-ocr-image.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))



AVAILABLE_MODELS = {
    'gpt-4': 'gpt-4',
    'gpt-3.5': 'gpt-3.5-turbo'
}
DEFAULT_MODEL = 'gpt-3.5'

start_time_ai = 0
start_time_ocr = 0
end_time_ai = 0
end_time_ocr = 0

processed_data = []  # Store processed data to be retrieved by GET request
processing_complete = False  # Flag to indicate when processing is complete

request_data_store = {}

logs = {}
ai_money = 0



def ocr_cost(calls):
    return calls * 0.084076

async def calculate_cost(input_tokens, output_tokens):
    global ai_money
    ai_money += input_tokens * 0.002532 + output_tokens * 0.005064

async def delay_between_requests():
    delay = random.uniform(1, 5)
    print(f"Delaying for {delay:.2f} seconds to prevent rate limit errors.")
    await asyncio.sleep(delay)

async def get_openai_response(prompt, model_name=DEFAULT_MODEL, retries=3):
        headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        }
        data = {
            'model': AVAILABLE_MODELS[model_name],
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1500,
            'temperature': 0.5
        }
        async with httpx.AsyncClient() as client:
            for attempt in range(retries):
                try:
                    print(1)
                    response = await client.post('https://api.openai.com/v1/chat/completions', 
                              headers=headers, 
                              json=data, 
                              timeout=30)                    
                    print(response.raise_for_status())
                    response_json = response.json()
                    print(response_json)
                    input_tokens = response_json['usage']['prompt_tokens']
                    output_tokens = response_json['usage']['completion_tokens']
                    await calculate_cost(input_tokens, output_tokens)
                    return response_json['choices'][0]['message']['content']
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        print("Rate limit exceeded. Retrying after delay...")
                        await delay_between_requests()
                    else:
                        print(f"HTTP error occurred: {e}")
                        raise
                except httpx.RequestError as e:
                    print(f"Error during API request: {e}")
                    raise
        print("Failed to get response after multiple attempts. Returning None.")

def sno():
    with open('variable', 'r') as lt:
        no = int(lt.readline())
        no += 1
        with open('variable', 'w') as ltw:
            ltw.write(str(no))
        with open('fixed', 'r') as ft:
            code = str(ft.readline())
            sno = str(code) + str(no)
    return sno

async def process_invoicing(invoice_texts, model_name=DEFAULT_MODEL):
    all_data = []
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available. Using default model '{DEFAULT_MODEL}' instead.")
        model_name = DEFAULT_MODEL
    responses = []
    no_of_invoices = len(invoice_texts)
    print("Abou to process")
    cost_ocr = ocr_cost(no_of_invoices)
    logs["ocr_cost"] = cost_ocr 
    all_ids = []
    for ocr_output in invoice_texts:
        # Print OCR text for debugging purposes
        print(len(ocr_output))        
        prompt = f"""
        The following text is extracted from an invoice:
        {ocr_output}

        Please extract the following information and provide it in a structured JSON format with the fields:
        - invoice_number: The unique invoice number, usually labeled with "Invoice No."
        - invoice_date: Date of the invoice, usually labeled as "Invoice Date" and ensure to have date in the format DD/MM/YYYY.
        - vendor_name: The seller or supplier name, typically found before "GSTIN" or at the top of the invoice.
        - vendor_address: The address of the vendor.
        - vendor_gst: GST Identification Number (GSTIN) of the vendor, usually labeled with "GST No." or "GSTIN."
        - vendor_pan: PAN (Permanent Account Number) of the vendor if available.
        - buyer_name: The name of the buyer or recipient, typically following "Ship to" or "Buyer".
        - buyer_gst: GST Identification Number of the buyer.
        - shipping_address: Address to which the goods or services are being shipped.
        - site_name: Site or project name where services/goods are provided, if applicable.
        - line_items: Extract line items (as an array of objects) containing:
            - description: Description of the product or service.
            - hsn_sac_code: HSN or SAC code associated with the item.
            - quantity: Quantity of items or services, usually a numeric value followed by units like "CUM", "KG", etc.
            - cumulative_quantity: The cumulative quantity value, usually labeled as "Cumulative Qty" or similar.
            - rate: Rate per unit, usually a monetary value in ₹ (INR). Avoid taking cumulative quantity values as rates.
            - amount: Total amount for the line item.
        - tax_details: An array of objects containing tax details:
            - tax_type: Type of tax (CGST, SGST, IGST, etc.)
            - rate: Tax rate in percentage.
            - amount: Amount of tax charged.
        - total_amount: Total amount payable after all taxes.
        - other_charges: Any additional charges such as transport or handling charges.
        - other_charges_amount: The amount for other charges.

        Important:
        - Ensure that "Vendor" and "Buyer" details are not confused. Vendor is the seller, typically mentioned first, and is associated with "GSTIN" or "PAN".
        - Avoid confusing cumulative quantities with rates. Quantities are usually numeric values with units like "CUM", "KG", or "L". Rates are monetary values with currency symbols like "₹" or "$".
        - If any fields are not found, return "None" as the value.
        """
        response_content = await get_openai_response(prompt, model_name)
        print(response_content)
        print(len(response_content))
        if not response_content:
            continue
        try:
            invoice_data = json.loads(response_content)
        except json.JSONDecodeError:
            print("Error: Could not parse the response as JSON.")
            print("Response:", response_content)
            continue
        if isinstance(invoice_data, str):
            print("Error: Unexpected response format. Response was a string instead of JSON.")
            print("Response:", invoice_data)
            continue

        # Extract summary data with checks for missing details
        invoice_id  = sno()
        all_ids.append(invoice_id)
        summary_data = {
            "Output ID": invoice_id,
            "Invoice Number": invoice_data.get("invoice_number", "not found"),
            "Invoice Date": invoice_data.get("invoice_date", "not found"),
            "Vendor Name": invoice_data.get("vendor_name", "not found"),
            "Vendor Address": invoice_data.get("vendor_address", "not found"),
            "Vendor GST": invoice_data.get("vendor_gst", "not found"),
            "Vendor PAN": invoice_data.get("vendor_pan", "not found"),
            "Buyer GST": invoice_data.get("buyer_gst", "not found"),
            "Shipping Address": invoice_data.get("shipping_address", "not found"),
            "Site/Project Name": invoice_data.get("site_name", "not found"),
            "Total Amount": invoice_data.get("total_amount", "not found"),
            "Other Charges": invoice_data.get("other_charges", "not found"),
            "Other Charges Amount": invoice_data.get("other_charges_amount", "not found")
        }

        # Handle tax details if available
        tax_details = invoice_data.get("tax_details", [])
        if isinstance(tax_details, list):
            for i, tax in enumerate(tax_details):
                if isinstance(tax, dict):
                    summary_data[f"Tax Type {i+1}"] = tax.get("tax_type", "not found")
                    summary_data[f"Tax Rate {i+1} (%)"] = tax.get("rate", "not found")
                    summary_data[f"Tax Amount {i+1}"] = tax.get("amount", "not found")

        # Handle line items if available
        line_items = invoice_data.get("line_items", [])
        if isinstance(line_items, list):
            for i, item in enumerate(line_items, start=1):
                if isinstance(item, dict):
                    summary_data[f"Description {i}"] = item.get("description", "not found")
                    summary_data[f"HSN/SAC Code {i}"] = item.get("hsn_sac_code", "not found")
                    summary_data[f"Quantity {i}"] = item.get("quantity", "not found")
                    summary_data[f"Cumulative Quantity {i}"] = item.get("cumulative_quantity", "not found")
                    summary_data[f"Rate {i}"] = item.get("rate", "not found")
                    summary_data[f"Amount {i}"] = item.get("amount", "not found")

        all_data.append(summary_data)
    logs["invoice_output_ids"] = all_ids
    return pd.DataFrame(all_data)

# Function to extract text from image using Azure OCR
async def extract_text_from_image(image_path, retries=3):
    for attempt in range(retries):
        try:
            with open(image_path, "rb") as image_stream:
                read_response = computervision_client.read_in_stream(image=image_stream, raw=True)

            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]
            
            await delay_between_requests()

            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                await asyncio.sleep(1)

            full_text = ""

            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        full_text += line.text + "\n"
                return full_text.strip()
            else:
                print("OCR failed. Retrying...")
                delay_between_requests()

        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            delay_between_requests()

    raise Exception("Failed to extract text from image after multiple retries.")

@app.post("/process_invoices")
async def batch_process_invoices(invoice_files: List[UploadFile] = File(...), model_name: str = DEFAULT_MODEL):
    global processing_complete
    logs["time stamp"] = date.today().strftime("%Y-%m-%d")
    request_id = str(uuid.uuid4())
    request_data_store[request_id] = {"status": "processing", "data": None}
    processing_complete = False  # Reset the flag at the start of processing
    file_paths = []


    for file in invoice_files:
        filename = f"temp_{file.filename}"
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(filename)

    try:
        print("Ocr start")
        ocr_start_time = time.time()
        ocr_outputs = await asyncio.gather(*[process_image(file_path) for file_path in file_paths])
        ocr_end_time = time.time()
        total_ocr_time = ocr_end_time - ocr_start_time
        logs["ocr_time"] = total_ocr_time
        print("Ocr Done")

        ai_start_time = time.time()
        print("inside")
        results = await process_invoicing(ocr_outputs,model_name=model_name)
        print("outside")
        ai_end_time = time.time()
        total_ai_time = ai_end_time - ai_start_time
        logs["ai_time"] = total_ai_time

        logs["total_time"] = total_ocr_time + total_ai_time

        logs["ai_cost"] = ai_money
        logs["total_cost"] = logs["ocr_cost"] + logs["ai_cost"]
        logs["time stamp"] = date.today().strftime("%Y-%m-%d")

        print(logs)
        print("Invoice data extracted successfully.")
        print(results)
        invoice_data_dict = results.replace({np.nan: None}).to_dict(orient='records')
        processed_data.extend(invoice_data_dict)  # Store processed data for GET request
        processing_complete = True  # Set the flag when processing is complete
        request_data_store[request_id]["status"] = "completed"
        request_data_store[request_id]["data"] = invoice_data_dict
        df = pd.DataFrame([logs])
        file_path = "logs_output.xlsx"

        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets["Sheet1"].max_row)
        else:
            df.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")

        return JSONResponse(content={"request_id": request_id, "invoice_data": invoice_data_dict})    
    except Exception as e:
        request_data_store[request_id]["status"] = "error"
        request_data_store[request_id]["error"] = str(e)
        return HTTPException(status_code=500, detail=str(e))
    finally:
        for path in file_paths:
            os.remove(path)

async def process_image(file_path: str):
    start_time = time.time()
    ocr_text = await extract_text_from_image(file_path)
    print(f"OCR extraction took {time.time() - start_time:.2f} seconds")
    return ocr_text

@app.get("/get_processed_invoices")
async def get_processed_invoices():
    try:
        if not processing_complete:
            raise HTTPException(status_code=202, detail="Processing not complete. Please try again later.")
        if not processed_data:
            raise HTTPException(status_code=404, detail="No processed data available.")
        return JSONResponse(content={"invoice_data": processed_data})
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
@app.get("/get_processed_invoices/{request_id}")
async def get_processed_invoices(request_id: str):
    try:
        request_data = request_data_store.get(request_id)
        if not request_data:
            raise HTTPException(status_code=404, detail="Request not found")
        if request_data["status"] == "processing":
            raise HTTPException(status_code=202, detail="Processing not complete")
        if request_data["status"] == "error":
            raise HTTPException(status_code=500, detail=request_data["error"])
        return {"invoice_data": request_data["data"]}       
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@app.post("/send_email")
async def send_email( email_list: str = Form(...), file: UploadFile = File(...)):
    EMAIL_ADDRESS = "bharathworks2u@gmail.com"
    EMAIL_PASSWORD = "yiyiqhgtovmjoxeq"
    emails = email_list.split(',')
    file_data = await file.read()
    file_name = file.filename

    msg = EmailMessage()
    msg['Subject'] = 'Invoice Processing Result'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ', '.join(emails)

    msg.set_content('This is an email with an attached invoice processing Excel file.')
    msg.add_attachment(file_data, maintype='application', subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=file_name)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return JSONResponse(content={"message": "Email sent successfully!"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Hey!! Welcome to the BluOrgin AI's Invoice Application Processor! add /docs to proceed"}
