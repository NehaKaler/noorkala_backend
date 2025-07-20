# backend/main.py

import os
import time
import base64
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, Form, File, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import replicate
import google.generativeai as genai 
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi import Request
from pathlib import Path
from pydantic import BaseModel
from typing import List
# from google.cloud import translate_v3 as translate
# from google.cloud import texttospeech
# from google.oauth2 import service_account

import google.generativeai as genai

load_dotenv()
app = FastAPI()

# Initialize router
router = APIRouter()

# app.include_router(translate.router)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Set up credentials
# SERVICE_ACCOUNT_FILE = "service-account.json"
# credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

# static_path = os.path.join(os.getcwd() + "/static")
# app.mount("/static", StaticFiles(directory=static_path), name="static")

# ---------------- CONFIG ----------------
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

business_model = genai.GenerativeModel("gemini-2.0-flash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#  request schema 
class ProductData(BaseModel):
    product: str
    category: str
    price: float
    units_sold: int

class SalesData(BaseModel):
    sales_data: List[ProductData]
    
# Language mappings
language_code_map = {
    "Hindi": "hi",
    "English": "en",
    "Tamil": "ta",
    "Bengali": "bn",
    "Marathi": "mr"
}
language_voice_map = {
    "Hindi": "hi-IN",
    "English": "en-US",
    "Tamil": "ta-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN"
}
    
# --------------- UTILS ------------------

# def translate_and_optionally_speak(text: str, lang: str, speak: bool = False):
#     lang = lang.strip().capitalize()
#     target_lang_code = language_code_map.get(lang, "en")
#     voice_code = language_voice_map.get(lang, "en-US")

#     # Translate using Translate v3 API
#     translate_client = translate.TranslationServiceClient(credentials=credentials)
#     parent = f"projects/{credentials.project_id}/locations/global"

#     response = translate_client.translate_text(
#         contents=[text],
#         target_language_code=target_lang_code,
#         parent=parent
#     )
#     translated = response.translations[0].translated_text

#     audio_path = None

#     if speak:
#         tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
#         synthesis_input = texttospeech.SynthesisInput(text=translated)
#         voice = texttospeech.VoiceSelectionParams(
#             language_code=voice_code,
#             ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#         )
#         audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

#         tts_response = tts_client.synthesize_speech(
#             input=synthesis_input,
#             voice=voice,
#             audio_config=audio_config
#         )

#         audio_path = f"tts_output_{target_lang_code}.mp3"
#         with open(audio_path, "wb") as out:
#             out.write(tts_response.audio_content)

#     return {
#         "translated": translated,
#         "audio_path": audio_path
#     }


def generate_enhancement_prompt(theme=str, occasion=str, productType=str, region=str, gender=None, feedback=None):
    prompt_parts = [
        f"From the given image, extract and identify the dress and Create a visually stunning product image from the given image to be featured for {productType.lower()}",
        f"styled with a {theme.lower()} theme" if theme else "",
        f"designed specifically for {occasion.lower()}" if occasion else "",
        f"featuring cultural elements and motifs from {region.lower()}" if region else "",
        "Ensure the product looks elegant, well-lit, and appealing to customers.",
        "Use soft, diffused lighting, a clean background, and subtle highlights.",
        "Keep the product centered and clearly visible, avoiding clutter.",
        "Highlight textures, colors, and craftsmanship to attract buyers."
    ]
    if gender:
        # prompt_parts.append(f"Include a {gender.lower()} model to showcase the product.")
        prompt_parts.append(f"Showcase the new generated dress as being weared by a model of gender {gender.lower()}")
    if feedback:
        prompt_parts.append(f"Improve based on this feedback: {feedback}")
    # Filter out empty strings
    prompt = ". ".join([part for part in prompt_parts if part])
    return prompt


def generate_caption_prompt(theme: str, captionLength: str, captionType: str, tone: str):
    prompt_parts = [
        f"Given the image input, Write a single-line product caption for the image that is catchy, elegant, and emotionally appealing.",
        f"The caption must reflect and resonate with the vibe of {theme.lower()}." if theme else "",
        f"The caption tone should be {tone.lower()}." if tone else "",
        f"The caption length should be {captionLength.lower()}." if captionLength else "",
        f"The caption should focus on {captionType.lower()}." if captionType else "",
        "Focus on words that evoke tradition, beauty, or celebration.",
        "Make it suitable for social media or online stores, and persuasive enough to attract a buyer.",
        
    ]

    return " ".join([p for p in prompt_parts if p])


def upload_to_imgbb(file: UploadFile) -> str:
    encoded = base64.b64encode(file.file.read()).decode("utf-8")
    response = requests.post("https://api.imgbb.com/1/upload", data={
        "key": IMGBB_API_KEY,
        "image": encoded
    })
    print("image BB:", response)
    data = response.json()
    return data['data']['url']

def call_enhance_api(prompt: str, image_url: str):
    print("starting image enhancement")
    if not REPLICATE_API_TOKEN:
        raise ValueError("Replicate API token is not set or invalid.")
    print("replicate token ", REPLICATE_API_TOKEN)
    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        print("replicate client")
        output = client.run(
            "black-forest-labs/flux-kontext-pro",
            input={"prompt": prompt, "input_image": image_url, "output_format": "jpg"}
        )
        print("replicate output: ", output)
        if not output:
            print("❌ Error: No output received from Replicate.")
            return None
        # Save locally
        print("replicate output: ", output)
        filename = f"enhanced_{int(time.time())}.jpg"
        print(filename)
        save_path = os.path.join("static", filename)
        try:
            image_data = requests.get(output).content
        except Exception as e:
            print("❌ Error downloading image from Replicate URL:", e)
            return None
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_data = requests.get(output).content
        with open(save_path, "wb") as f:
            f.write(image_data)
        return f"{filename}"
    except replicate.exceptions.AuthenticationError:
        print("Replicate authentication error")
        raise ValueError("Invalid Replicate API token. Please check your credentials.")
    except Exception as e:
        print("Exception occured with replicate")
        raise RuntimeError(f"Error while calling replicate API: {e}")

def call_caption_api(image: UploadFile, prompt: str):
    genai.configure(api_key=GOOGLE_API_KEY)
    pil_image = Image.open(image.file)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content([prompt, pil_image])
    print("gemini-response: ", response)
    return response.text

# --------------- ROUTES ------------------

@app.post("/generate-image")
async def generate_image(
    image: UploadFile = File(...),
    theme: str = Form(...),
    occasion: str = Form(...),
    productType: str = Form(...),
    region: str = Form(...),
    gender: str = Form(...),
    feedback: Optional[str] = Form(None),
):
    
    try:
        print("Received image:", image.filename)
        prompt = generate_enhancement_prompt(theme, occasion, productType, region, gender=gender, feedback=None)
        print("Prompt:", prompt)
        imgbb_url = upload_to_imgbb(image)
        print("ImageBBUrl:" + imgbb_url)
        enhanced_file_name = call_enhance_api(prompt, imgbb_url)
        # return {"enhanced_image": enhanced_path}
        return {
            "image_url": enhanced_file_name
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get-enhanced-image/{file_name}")
def get_enhanced_image(file_name: str):
    # Get full path to the image using pathlib
    image_file_path = Path(__file__).resolve().parent / "static" / file_name
    
    # ✅ Print for debug
    print("Serving file:", image_file_path)

    # ✅ Return the image
    return FileResponse(
        path=os.path.join(image_file_path),
        media_type="image/png",  # or "image/jpeg" depending on your file
        filename=file_name 
    )


@app.post("/regenerate-with-feedback")
async def regenerate_with_feedback(
    image: UploadFile = File(...),
    theme: str = Form(""),
    occasion: str = Form(""),
    productType: str = Form(""),
    region: str = Form(""),
    gender: str = Form(""),
    feedback: str = Form("")
):  
    try:
        print("Received image:", image.filename)
        prompt = generate_enhancement_prompt(theme, occasion, productType, region, gender=gender, feedback=feedback)
        print("Prompt:", prompt)
        imgbb_url = upload_to_imgbb(image)
        print("ImageBBUrl:" + imgbb_url)
        enhanced_file_name = call_enhance_api(prompt, imgbb_url)
        # return {"enhanced_image": enhanced_path}
        return {
            "image_url": enhanced_file_name
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    # # Step 1: Reconstruct original prompt
    # base_prompt = f"A {gender} model wearing a {productType} for {occasion}, theme: {theme}, region: {region}."

    # # Step 2: Combine with feedback
    # improved_prompt = f"{base_prompt} {feedback.strip()}"
    
    # # Step 3: Call image generation model with improved_prompt
    # result = call_image_model(prompt=improved_prompt, image=image)

    # # Optional: save (base_prompt, feedback, improved_prompt) to file/db

    return {
        "image_url": result["image_url"],
        "caption": result["caption"]
    }


@app.post("/generate-caption")
async def caption(
    image: UploadFile= File(...),
    theme: str = Form(...),
    captionType: str = Form(...),
    captionLength: str = Form(...),
    tone: str = Form(...)
):
    try:
        prompt = generate_caption_prompt(theme, captionLength, captionType, tone)
        print("caption-prompt: ", prompt)
        caption = call_caption_api(image, prompt)
        print("caption: ", caption)
        return {"caption": caption}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/business-report")
async def generate_business_insight(data: SalesData):
    try:
        # Combine insights from multiple products
        report_sections = []
        for item in data.sales_data:
            prompt = (
                f"You are a business analyst. Generate a short business insight report for the following product:\n\n"
                f"- Product: {item.product}\n"
                f"- Category: {item.category}\n"
                f"- Price: ₹{item.price}\n"
                f"- Units Sold: {item.units_sold}\n\n"
                f"Include insights on performance, improvement tips, and potential trends.\n"
                f"Make sure the report is in a readable and presentable form and has really good indentation and has suitable headings and sub-headings too."
            )
            result = business_model.generate_content(prompt)
            report_sections.append(result.text)

        full_report = "\n\n---\n\n".join(report_sections)

        return {"report": full_report}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating business report: {str(e)}")


# @router.post("/translate_and_speak")
# async def handle_translate_speak(request: Request):
#     data = await request.json()
#     text = data.get("text")
#     lang = data.get("lang")
#     speak = data.get("speak", False)

#     result = translate_and_optionally_speak(text, lang, speak)
    
#     response_data = {
#         "translated": result["translated"],
#     }

#     if result["audio_path"]:
#         response_data["audio_url"] = f"/static/audio/{result['audio_path'].split('/')[-1]}"

#     return response_data
