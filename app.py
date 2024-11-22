from fastapi import FastAPI, HTTPException, File, UploadFile, Form, status, Depends
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
from pydantic import BaseModel, EmailStr
import re
from typing import Optional, Dict, Any, List
from collections import Counter
from statistics import mean
import mimetypes


if not firebase_admin._apps:
    CredentialCertificate = os.environ.get('CREDENTIALCERTIFICATE')
    firebase_credentials_dict = json.loads(CredentialCertificate)
    cred = credentials.Certificate(firebase_credentials_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://bluorigin-859f2-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    
# Security configurations
SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

API_KEY = os.environ.get("OPENAI_API_KEY")
subscription_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
endpoint = "https://my-ocr-image.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Models for data validation
class UserBase(BaseModel):
    username: str
    email: EmailStr
    disabled: Optional[bool] = False
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserInDB(UserBase):
    hashed_password: str
    disabled: bool = False  # Default is False, no need for Optional
    login_count: int = 0  # Default is 0
    logout_count: int = 0  # Default is 0
    time_spent: Dict[str, float] = {}  # Default to an empty dictionary


class StatisticsResponse(BaseModel):
    avg_ocr_time_per_invoice: float
    avg_ai_time_per_invoice: float
    avg_ocr_cost_per_invoice: float
    avg_ai_cost_per_invoice: float
    total_invoices_processed: int
    total_time_spent: float
    total_cost: float
    most_used_model: str
    gpt_3_5_usage_percentage: float
    gpt_4_usage_percentage: float
    avg_cost_per_invoice: float
    avg_time_taken_per_invoice: float

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows selected origins only
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)



# Utility functions
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Firebase helpers
def get_user_from_firebase(username: str) -> Optional[dict]:
    users_ref = db.reference("/users")
    user_data = users_ref.child(username).get()
    if user_data and "time_spent" not in user_data:
        user_data["time_spent"] = {}
    return user_data

def get_user_by_email(email: str) -> Optional[dict]:
    users_ref = db.reference("/users")
    users = users_ref.get() or {}
    for user in users.values():
        if user.get("email") == email:
            return user
    return None

def add_user_to_firebase(user: Dict[str, Any]):
    users_ref = db.reference("/users")
    users_ref.child(user["username"]).set(user)

def update_user_login(username: str):
    try:
        user_ref = db.reference(f"/users/{username}")
        now = datetime.utcnow().isoformat()

        # Fetch existing user data
        user_data = user_ref.get()
        current_login_count = user_data.get("login_count", 0) if user_data else 0

        # Increment login count and update
        user_ref.update({
            "login_count": current_login_count + 1,
            "last_login": now
        })
    except Exception as e:
        print(f"Error updating user login: {e}")

def update_user_logout(username: str):
    try:
        user_ref = db.reference(f"/users/{username}")
        user_data = user_ref.get()

        if user_data is None:
            raise ValueError(f"No data found for username: {username}")

        now = datetime.utcnow()
        last_login = user_data.get("last_login")
        time_spent = user_data.get("time_spent", {})

        # Ensure `time_spent` is a dictionary
        if not isinstance(time_spent, dict):
            time_spent = {}

        session_duration = 0
        if last_login:
            try:
                last_login_time = datetime.fromisoformat(last_login)
                session_duration = (now - last_login_time).total_seconds()
            except ValueError:
                print(f"Invalid `last_login` format for user {username}: {last_login}")

        # Determine the current month (Year-Month)
        current_month = now.strftime("%Y-%m")
        monthly_time = time_spent.get(current_month, 0)

        # Update monthly time spent
        updated_monthly_time = monthly_time + session_duration
        time_spent[current_month] = updated_monthly_time

        # Retrieve current logout count, increment it, and update
        logout_count = user_data.get("logout_count", 0)
        logout_count += 1

        # Update the user's data in the database
        user_ref.update({
            "logout_count": logout_count,  # Manually increment logout count
            "last_logout": now.isoformat(),
            "time_spent": time_spent
        })
    except Exception as e:
        print(f"Error updating user logout: {e}")



def revoke_token(token: str):
    blacklist_ref = db.reference("/token_blacklist")
    blacklist_ref.push({
        "token": token,
        "revoked_at": datetime.utcnow().isoformat()
    })


def is_token_revoked(token: str) -> bool:
    blacklist_ref = db.reference("/token_blacklist")
    tokens = blacklist_ref.get() or {}  # Fetch all revoked tokens; default to an empty dict
    for entry in tokens.values():
        if entry.get("token") == token:
            return True
    return False




# Dependencies
def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    
    if is_token_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_data = get_user_from_firebase(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found.")
    if not isinstance(user_data.get("time_spent", {}), dict):
        user_data["time_spent"] = {}
    return UserInDB(**user_data)

def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    if current_user.disabled:
        raise HTTPException(status_code=403, detail="Inactive user.")
    return current_user


# API routes
# Signup route
@app.post("/signup", status_code=201)
async def signup(user: UserCreate):
    existing_user = get_user_from_firebase(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists.")

    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False,
        "created_at": datetime.utcnow().isoformat(),
        "login_count": 0,
        "logout_count": 0,
        "time_spent": {},  # Ensure time_spent is a dictionary
        "last_login": None,
        "last_logout": None
    }
    add_user_to_firebase(user_data)
    return {"message": "User created successfully."}


# Login route
@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_data = get_user_from_firebase(form_data.username)
    if not user_data:
        raise HTTPException(status_code=401, detail="Incorrect username")
    if not verify_password(form_data.password, user_data["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    if user_data.get("disabled", False):
        raise HTTPException(status_code=403, detail="User account is disabled.")

    update_user_login(form_data.username)
    db.reference(f"/users/{form_data.username}/activity_logs/logins").push(datetime.utcnow().isoformat())

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}


# Logout route
@app.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_active_user),token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token.")
        timestamp = datetime.utcnow().isoformat()
        users_ref = db.reference(f"/users/{username}/activity_logs")
        users_ref.child("logouts").push(timestamp)
        update_user_logout(current_user.username)
        revoke_token(token)
        return {"message": "Logged out successfully."}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate token.")



# User profile route
@app.get("/me", response_model=UserBase)
async def read_current_user(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user


# Admin routes
@app.get("/activity_logs/{username}")
async def get_activity_logs(username: str, token: str = Depends(oauth2_scheme)):
    user_data = get_user_from_firebase(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found.")
    return db.reference(f"/users/{username}/activity_logs").get() or {"logins": [], "logouts": []}

# Admin route to get user stats
@app.get("/user/{current_user.username}/stats")
async def get_user_stats(current_user: UserInDB = Depends(get_current_active_user), token: str = Depends(oauth2_scheme)):
    if current_user.disabled: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    try:
        user_data = get_user_from_firebase(current_user.username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")
        
        time_spent_data = user_data.get("time_spent", {})
        if not isinstance(time_spent_data, dict):
            time_spent_data = {}

        total_time_spent_seconds = sum(time_spent_data.values())
        total_time_spent_minutes = total_time_spent_seconds / 60  # Convert to minutes

        current_month = datetime.utcnow().strftime("%Y-%m")
        time_spent_current_month_seconds = time_spent_data.get(current_month, 0)
        time_spent_current_month_minutes = time_spent_current_month_seconds / 60  # Convert to minutes

        return {
            "message": "User statistics retrieved successfully.",
            "login_count": user_data.get("login_count", 0),
            "logout_count": user_data.get("logout_count", 0),
            "time_spent_current_month_minutes": round(time_spent_current_month_minutes, 2),  # Time spent this month
            "total_time_spent_minutes": round(total_time_spent_minutes, 2),  # Total time spent across all months
            "monthly_time_spent": {
                month: round(time_spent_seconds / 60, 2)  
                for month, time_spent_seconds in time_spent_data.items()
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "login_count": 0,
            "logout_count": 0,
            "time_spent_current_month_minutes": 0,
            "total_time_spent_minutes": 0,
            "monthly_time_spent": {}
        }





#Normal processes
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


async def rearrange_data(data):
    key_string = ""
    for key in data:
        key_string += key + " "
    data["key_string"] = key_string
    return data


async def validate_and_format_data(data):
    def format_amount(amount):
        try:
            amount = re.sub(r'[^\d.]+', '', amount)  # Remove non-numeric characters
            formatted_amount = f"₹ {float(amount):,.2f}"
            return formatted_amount
        except ValueError:
            return "not found"

    def format_tax_rate(rate):
        try:
            rate = re.sub(r'[^\d.]+', '', rate)  
            if rate:
                formatted_rate = f"{float(rate):.2f}%"
                return formatted_rate
            return "not found"
        except ValueError:
            return "not found"

    for key in data.keys():
        if "amount" in key.lower():
            data[key] = format_amount(data[key])
        if "rate" in key.lower() and "tax" in key.lower():
            data[key] = format_tax_rate(data[key])
    
    return data


async def calculate_total_amount_before_tax(data):
    def extract_numeric_amount(amount_str):
        try:
            return float(re.sub(r'[^\d.]', '', amount_str))  
        except ValueError:
            return 0.0

    total_amount = extract_numeric_amount(data.get("Total_Amount", "0"))    
    tax_keys = [key for key in data.keys() if key.startswith("Tax_Amount")]
    total_tax_amount = sum(extract_numeric_amount(data.get(key, "0")) for key in tax_keys)    
    total_amount_before_tax = total_amount - total_tax_amount    
    formatted_amount_before_tax = f"₹ {total_amount_before_tax:,.2f}"
    return formatted_amount_before_tax



async def convert_keys_to_lowercase(data):
    if isinstance(data, dict):
        return {key.lower(): await convert_keys_to_lowercase(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [await convert_keys_to_lowercase(item) for item in data]
    else:
        return data

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
        - site_name: Site or project name where services/goods are provided, always ending with "site" or "project if applicable.
        - line_items: Extract line items (as an array of objects) containing:
            - description: Simplified item description (e.g., "Steel Test" from "Steel Test J-D-13-14/01").
            - hsn_sac_code: 6-digit SAC code or 8-digit HSN code. Ignore codes of incorrect length or invalid characters.
            - quantity: Numeric value of items, often with units (e.g., "CUM", "KG").
            - cumulative_quantity: The cumulative quantity value, usually labeled as "Cumulative Qty."
            - rate: Rate per unit, formatted as ₹ X,XXX.XX.
            - amount: Total amount for the line item, formatted as ₹ X,XXX.XX.
        - tax_details: An array of objects containing tax details:
            - tax_type: Type of tax (CGST, SGST, IGST, etc.).
            - rate: Tax rate in percentage (e.g., "14.00%"). If extracted as '14.00', append '%'.
            - amount: Amount of tax charged, formatted as ₹ X,XXX.XX.
        - grand_total_amount: Total amount payable after all taxes and other charges, formatted as ₹ X,XXX.XX.
        - other_charges: Any additional charges such as transport or handling charges.
        - other_charges_amount: The amount for other charges, formatted as ₹ X,XXX.XX.

        Important:
        - Ensure that "Vendor" and "Buyer" details are not confused. Vendor is the seller, typically mentioned first, and is associated with "GSTIN" or "PAN".
        - Avoid confusing cumulative quantities with rates. Quantities are usually numeric values with units like "CUM", "KG", or "L". Rates are monetary values with currency symbols like "₹" or "$".
        - If any fields are not found, return "Not found" as the value in string format instead of null or None.
        - In the Json responce, if any key name has more than one word, please use underscore(_) instead of space.

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


        invoice_data = await convert_keys_to_lowercase(invoice_data)
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
            "Total_Amount": invoice_data.get("grand_total_amount", "not found"),
            "Other_Charges": invoice_data.get("other_charges", "not found"),
            "Other_Charges_Amount": invoice_data.get("other_charges_amount", "not found"),
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

        summary_data["total_amount_before_tax"] = await calculate_total_amount_before_tax(summary_data)
        validated_data = await validate_and_format_data(summary_data)
        rearrangedata = await rearrange_data(validated_data)
        all_data.append(rearrangedata)

    logs["invoice_output_ids"] = all_ids
    logs["openai_outputs"] = responses
    return all_data
    

# Function to extract text from image using Azure OCR
async def extract_text_from_image(image_path, retries=3):
    for attempt in range(retries):
        try:
            mime_type, _ = mimetypes.guess_type(image_path)
            with open(image_path, "rb") as file_stream:
                if mime_type in ["application/pdf"]:
                    # Process PDF files
                    read_response = computervision_client.read_in_stream(image=file_stream, raw=True)
                elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                    # Process image files
                    read_response = computervision_client.read_in_stream(image=file_stream, raw=True)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

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



async def process_image(file_path: str):
    start_time = time.time()
    ocr_text = await extract_text_from_image(file_path)
    print(f"OCR extraction took {time.time() - start_time:.2f} seconds")
    return ocr_text


@app.post("/process_invoices")
async def batch_process_invoices(invoice_files: List[UploadFile] = File(...), model_name: str = DEFAULT_MODEL, current_user: UserInDB = Depends(get_current_active_user),token: str = Depends(oauth2_scheme)):
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

        # invoice_data_dict = results.replace({np.nan: "NULL"}).to_dict(orient='records')
        processing_complete = True  # Set the flag when processing is complete
        logs["invoices_result"] = results
        try:
            refa = db.reference('output_data/' + request_id)
            refa.set({
                    "time_stamp": logs["time_stamp"],
                    "request_id": request_id,
                    "result": results,
            })
            print("Written results to firebase")

            # Capture user details and update the user profile with the new request_id
            user_ref = db.reference(f'/users/{current_user.username}')
            user_data = user_ref.get()  # Get the current data
            current_request_ids = user_data.get("request_ids", []) if user_data else []
            if request_id not in current_request_ids:
                current_request_ids.append(request_id)

            user_ref.update({
                "request_ids": current_request_ids  # Update the list with the new request_id
            })
            print(f"Request ID {request_id} assigned to user {current_user.username}")

            return JSONResponse(content={"request_id": request_id, "invoice_data": results}) 
        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))
          
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
        print("Removed files")
        output_id = request_id
        ref = db.reference('logs/' + output_id)
        ref.set({
                "time_stamp": logs["time_stamp"],
                "output_id": output_id,
                "content" : logs
        })
        print("Data Pushed to Firebase")


async def original_format(data):
    final_list = []
    for db_dict in data:  # Loop over the list
        if not isinstance(db_dict, dict):  # Ensure each item is a dictionary
            continue
        
        inside_dict = {}
        key_string = db_dict.get("key_string", "")  # Safely get 'key_string'
        keys = key_string.split()
        
        for key in keys:
            if key == "key_string":
                continue
            if key in db_dict:
                inside_dict[key] = db_dict[key]
        
        final_list.append(inside_dict)

    return final_list

    
@app.get("/get_processed_invoices/{request_id}")
async def get_processed_invoices(request_id: str, current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.disabled:  # Example of checking user permissions
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    
    try:
        refa = db.reference(f'output_data/{request_id}')
        print("Request loaded")        
        data = refa.get()
        print("Data Fetched Successfully")
        if data and 'result' in data:
            db_data = data['result']
            real_data = await original_format(db_data)
            return JSONResponse(
                content={"request_id": request_id, "invoice_data": real_data}
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


# Statistics route
@app.get("/statistics/{current_user.username}", response_model=StatisticsResponse)
async def get_statistics(current_user: UserInDB = Depends(get_current_active_user), token: str = Depends(oauth2_scheme)):   
    if current_user.disabled:  
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    try:
        user_ref = db.reference(f"users/{current_user.username}")
        user_data = user_ref.get()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_request_ids = user_data.get("request_ids", [])
        if not user_request_ids:
            raise HTTPException(status_code=404, detail="No request IDs found for this user")

        logs_ref = db.reference("logs")
        logs_data = logs_ref.get()
        if not logs_data:
            raise HTTPException(status_code=404, detail="No logs found in the database")

        filtered_logs = [logs_data[req_id] for req_id in user_request_ids if req_id in logs_data]

        if not filtered_logs:
            raise HTTPException(status_code=404, detail="No logs found for the user's request IDs")

        ocr_times, ai_times, ocr_costs, ai_costs, invoice_counts = [], [], [], [], []
        total_time, total_cost = 0, 0
        model_usage = Counter()

        for log in filtered_logs:
            content = log.get("content", {})
            ocr_times.append(content.get("ocr_time", 0))
            ai_times.append(content.get("ai_time", 0))
            ocr_costs.append(content.get("ocr_cost", 0))
            ai_costs.append(content.get("ai_cost", 0))
            invoice_counts.append(content.get("invoice_count", 0))
            total_time += content.get("total_time", 0)
            total_cost += content.get("total_cost", 0)
            model_usage[content.get("model_name", "unknown")] += 1

        total_invoices = sum(invoice_counts)
        avg_ocr_time_per_invoice = mean(ocr_times) if ocr_times else 0
        avg_ai_time_per_invoice = mean(ai_times) if ai_times else 0
        avg_ocr_cost_per_invoice = mean(ocr_costs) if ocr_costs else 0
        avg_ai_cost_per_invoice = mean(ai_costs) if ai_costs else 0

        most_used_model, most_used_count = model_usage.most_common(1)[0] if model_usage else ("unknown", 0)
        gpt_3_5_count = model_usage.get("gpt-3.5", 0)
        gpt_4_count = model_usage.get("gpt-4", 0)
        gpt_3_5_usage_percentage = (gpt_3_5_count / sum(model_usage.values()) * 100) if model_usage else 0
        gpt_4_usage_percentage = (gpt_4_count / sum(model_usage.values()) * 100) if model_usage else 0
        avg_cost_per_invoice = total_cost / total_invoices if total_invoices else 0
        avg_time_taken_per_invoice = total_time / total_invoices if total_invoices else 0

        return StatisticsResponse(
            avg_ocr_time_per_invoice=avg_ocr_time_per_invoice,
            avg_ai_time_per_invoice=avg_ai_time_per_invoice,
            avg_ocr_cost_per_invoice=avg_ocr_cost_per_invoice,
            avg_ai_cost_per_invoice=avg_ai_cost_per_invoice,
            total_invoices_processed=total_invoices,
            total_time_spent=total_time,
            total_cost=total_cost,
            most_used_model=most_used_model,
            gpt_3_5_usage_percentage=gpt_3_5_usage_percentage,
            gpt_4_usage_percentage=gpt_4_usage_percentage,
            avg_cost_per_invoice=avg_cost_per_invoice,
            avg_time_taken_per_invoice=avg_time_taken_per_invoice
        )
    except Exception as e:
        return StatisticsResponse(
            avg_ocr_time_per_invoice=0,
            avg_ai_time_per_invoice=0,
            avg_ocr_cost_per_invoice=0,
            avg_ai_cost_per_invoice=0,
            total_invoices_processed=0,
            total_time_spent=0,
            total_cost=0,
            most_used_model="unknown",
            gpt_3_5_usage_percentage=0,
            gpt_4_usage_percentage=0,
            avg_cost_per_invoice=0,
            avg_time_taken_per_invoice=0
        )


# User history route
@app.get("/user_history/{current_user.username}", response_model=Dict[str, List[Dict]])
async def user_history(current_user: UserInDB = Depends(get_current_active_user), token: str = Depends(oauth2_scheme)):
    if current_user.disabled: 
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    try:
        user_ref = db.reference(f"users/{current_user.username}")
        user_data = user_ref.get()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_request_ids = user_data.get("request_ids", [])
        if not user_request_ids:
            raise HTTPException(status_code=404, detail="No request IDs found for this user")

        outputs_ref = db.reference("output_data")
        outputs_data = outputs_ref.get()
        if not outputs_data:
            raise HTTPException(status_code=404, detail="No output data found in the database")

        filtered_outputs = [outputs_data[req_id] for req_id in user_request_ids if req_id in outputs_data]

        if not filtered_outputs:
            raise HTTPException(status_code=404, detail="No output data found for the user's request IDs")

        results = []
        for output in filtered_outputs:
            result_data = output.get("result", [])
            if isinstance(result_data, list):
                results.extend(result_data)

        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Hey Just Welcome to the BluOrgin FirebaseAI's Invoice Application Processor! add /upload-invoice to the URL to upload an invoice image."}
