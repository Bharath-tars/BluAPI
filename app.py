from fastapi import FastAPI, HTTPException, File, UploadFile, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
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
from datetime import date
from fastapi import FastAPI, Depends, HTTPException, status, Request
import time
import uuid
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
import time
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
import re



SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "$2b$12$dCpkIfEbHLNiOOCQUrFoNeSWLCYPsabGcUu.xmu1XNTiohE1.0ObK",  # password: "testpassword123"
        "full_name": "Test User",
        "email": "test@gmail.com",
        "disabled": False

    }
}

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



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

CredentialCertificate = os.environ.get('CREDENTIALCERTIFICATE')
firebase_credentials_dict = json.loads(CredentialCertificate)

cred = credentials.Certificate(firebase_credentials_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://bluorigin-859f2-default-rtdb.asia-southeast1.firebasedatabase.app/'
})


AVAILABLE_MODELS = {
    'gpt-4': 'gpt-4',
    'gpt-3.5': 'gpt-3.5-turbo'
}
DEFAULT_MODEL = 'gpt-4'

start_time_ai = 0
start_time_ocr = 0
end_time_ai = 0
end_time_ocr = 0


logs = {}
ai_money = 0



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(fake_db, username: str):
    if username in fake_db:
        user_data = fake_db[username]
        return UserInDB(**user_data)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False

    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception

        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception

    user = get_user(fake_db, username=token_data.username)
    if user is None:
        raise credential_exception

    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}



def ocr_cost(calls):
    return calls * 0.084076

async def calculate_cost(input_tokens, output_tokens):
    global ai_money
    if logs["model_name"] == "gpt-4":
        ai_money += input_tokens * 0.002532 + output_tokens * 0.005064
    elif logs["model_name"] == "gpt-3.5":
        ai_money += input_tokens * 0.00012672 + output_tokens * 0.00025344
    else:
        ai_money += input_tokens * 0.0012 + output_tokens * 0.0024

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
        return None

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
    print("About to process")
    all_ids = []
    responses= []
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

        **Please provide only the JSON content, without any code block markers, explanations, or extra text. Start directly with open brackets and end with closed brackets, formatted as plain JSON.**
        """
        response_content = await get_openai_response(prompt, model_name)
        print(response_content)
        print(len(response_content))
        responses.append(response_content)
        if not response_content:
            continue
        try:
            cleaned_response = re.sub(r'```json\n|\n```', '', response_content)
            invoice_data = json.loads(cleaned_response)
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
            "Output_ID": invoice_id,
            "Invoice_Number": invoice_data.get("invoice_number", "not found"),
            "Invoice_Date": invoice_data.get("invoice_date", "not found"),
            "Vendor_Name": invoice_data.get("vendor_name", "not found"),
            "Vendor_Address": invoice_data.get("vendor_address", "not found"),
            "Vendor_GST": invoice_data.get("vendor_gst", "not found"),
            "Vendor_PAN": invoice_data.get("vendor_pan", "not found"),
            "Buyer_GST": invoice_data.get("buyer_gst", "not found"),
            "Shipping_Address": invoice_data.get("shipping_address", "not found"),
            "Site_or_Project_Name": invoice_data.get("site_name", "not found"),
            "Total_Amount": invoice_data.get("total_amount", "not found"),
            "Other_Charges": invoice_data.get("other_charges", "not found"),
            "Other_Charges_Amount": invoice_data.get("other_charges_amount", "not found")
        }

        # Handle tax details if available
        tax_details = invoice_data.get("tax_details", [])
        if isinstance(tax_details, list):
            for i, tax in enumerate(tax_details):
                if isinstance(tax, dict):
                    summary_data[f"Tax_Type_{i+1}"] = tax.get("tax_type", "not found")
                    summary_data[f"Tax_Rate_{i+1}"] = tax.get("rate", "not found")
                    summary_data[f"Tax_Amount_{i+1}"] = tax.get("amount", "not found")

        # Handle line items if available
        line_items = invoice_data.get("line_items", [])
        if isinstance(line_items, list):
            for i, item in enumerate(line_items, start=1):
                if isinstance(item, dict):
                    summary_data[f"Description_{i}"] = item.get("description", "not found")
                    summary_data[f"HSN_or_SAC_Code_{i}"] = item.get("hsn_sac_code", "not found")
                    summary_data[f"Quantity_{i}"] = item.get("quantity", "not found")
                    summary_data[f"Cumulative_Quantity_{i}"] = item.get("cumulative_quantity", "not found")
                    summary_data[f"Rate_{i}"] = item.get("rate", "not found")
                    summary_data[f"Amount_{i}"] = item.get("amount", "not found")

        all_data.append(summary_data)
    logs["invoice_output_ids"] = all_ids
    logs["openai_outputs"] = responses
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
async def batch_process_invoices(invoice_files: List[UploadFile] = File(...), model_name: str = DEFAULT_MODEL, current_user: User = Depends(get_current_user)):
    if current_user.disabled: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    global processing_complete
    request_id = str(uuid.uuid4())
    processing_complete = False  
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
        logs["time_stamp"] = date.today().strftime("%Y-%m-%d")
        logs["invoice_count"] = len(ocr_outputs)
        logs["model_name"] = model_name
        total_ocr_time = ocr_end_time - ocr_start_time
        logs["ocr_time"] = total_ocr_time
        logs["ocr_cost"] = ocr_cost(len(ocr_outputs))
        logs["ocr_outputs"] = ocr_outputs
        print("Ocr Done")
    

        ai_start_time = time.time()
        print("inside")
        results = await process_invoicing(ocr_outputs,model_name=model_name)
        print("outside")
        ai_end_time = time.time()
        total_ai_time = ai_end_time - ai_start_time
        logs["ai_time"] = total_ai_time
        logs["ai_cost"] = ai_money

        logs["total_time"] = total_ocr_time + total_ai_time
        logs["total_cost"] = logs["ocr_cost"] + logs["ai_cost"]
        print("Invoice data extracted successfully.")
        print(results)

        invoice_data_dict = results.replace({np.nan: "NULL"}).to_dict(orient='records')
        processing_complete = True  # Set the flag when processing is complete
        logs["invoices_result"] = invoice_data_dict
        refa = db.reference('output_data/' + request_id)

        refa.set({
                "time_stamp": logs["time_stamp"],
                "result": invoice_data_dict,
        })
        return JSONResponse(content={"request_id": request_id, "invoice_data": invoice_data_dict}) 
       
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
    finally:
        for path in file_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error while removing file {path}: {str(e)}")

        output_id = request_id
        ref = db.reference('logs/' + output_id)
        ref.set({
                "time_stamp": logs["time_stamp"],
                "output_id": output_id,
                "content" : logs
        })
        print("Data Pushed to Firebase")

async def process_image(file_path: str):
    start_time = time.time()
    ocr_text = await extract_text_from_image(file_path)
    print(f"OCR extraction took {time.time() - start_time:.2f} seconds")
    return ocr_text

    
@app.get("/get_processed_invoices/{request_id}")
async def get_processed_invoices(request_id: str, current_user: User = Depends(get_current_user)):
    if current_user.disabled:  # Example of checking user permissions
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    try:
        refa = db.reference(f'output_data/{request_id}')
        print("Request loaded")        
        data = refa.get()
        print("Data Fetched Successfully")
        if data and 'result' in data:
            return JSONResponse(
                content={"request_id": request_id, "invoice_data": data['result']}
            )
        elif data:
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "error": "'result' key not found in the data."
                },
                status_code=404
            )
        else:
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "error": f"No data found for request_id: {request_id}"
                },
                status_code=404
            )
        
    except Exception as e:
        return JSONResponse(
            content={
                "request_id": request_id,
                "error": str(e),
                "message": "An error occurred while fetching the data."
            },
            status_code=500
        )

@app.get("/")
async def root():
    return {"message": "Hey Just Welcome to the BluOrgin FirebaseAI's Invoice Application Processor! add /upload-invoice to the URL to upload an invoice image."}


# password = "testpassword123"
# hashed_password = get_password_hash(password)
# print(hashed_password)
