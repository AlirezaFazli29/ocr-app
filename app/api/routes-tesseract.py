import io
import base64
import pytesseract
from PIL import Image
from fastapi import (
    Form,
    FastAPI,
    UploadFile,
    HTTPException,
)
from .schemas import (
    Language,
    OCRJsonRequest,
)
from fastapi.responses import JSONResponse


app = FastAPI(title="Tesseract OCR Service")


@app.get(
        path = "/",
        tags = [
            "Health",
        ],
)
async def root():
    """
    Root endpoint for the tesseract OCR service.

    This function serves as a health check for the application.
    When a GET request is made to the root endpoint ("/"), 
    it returns a simple message indicating that the OCR service is running.

    Returns:
        dict: A dictionary with a message confirming that the service is running.
    """
    return JSONResponse(
        {
            "message": "Tesseract OCR Service is running!"
        }
    )
    

@app.post(
        path = "/file-to-base64",
        tags = [
            "Health",
        ],
)
async def file_to_base64(
    file: UploadFile,
):
    """
    Convert an uploaded file to a base64-encoded string.

    This function processes an uploaded file and converts its binary content
    into a base64-encoded string. The resulting string can be used for
    serialization or transmission of the file content in text format.
    If any error occurs during file processing, it raises an appropriate 
    HTTP exception with a message.

    Args:
        file (UploadFile): The file uploaded by the client.

    Returns:
        JSONResponse: A JSON response containing the filename and the 
                      base64-encoded string of the file content.
    """
    try:
        file_data = await file.read()
        base64_string = base64.b64encode(file_data).decode('utf-8')
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process the uploaded file.",
        )

    return JSONResponse(
        content={
            "filename": file.filename,
            "base64_string": base64_string,
        }
    )


@app.post(
        path = "/ocr",
        tags = [
            "Tesseract Functions",
        ],
)

async def perform_ocr(
    file: UploadFile,
    language: Language = Form(Language.Farsi),
):
    """
    Perform OCR on an uploaded image.

    This function handles the OCR process:
    - Accepts an image file and the language to be used for OCR.
    - It reads the uploaded image, performs OCR using Tesseract, 
      and returns the extracted text in the specified language.
    - If any error occurs during image processing or OCR, 
      it raises an appropriate HTTP exception with a message.

    Args:
        file (UploadFile): The image file to perform OCR on.
        language (language): The language to use for OCR.

    Returns:
        JSONResponse: A JSON response with the extracted text and OCR language.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the uploaded file.",
                "error": str(e),
            },
        )
    
    try:
        extracted_text = pytesseract.image_to_string(image, lang=language.value)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error during OCR",
                "error": str(e),
            },
        )
    
    return JSONResponse(
        content={
            "language": language.name,
            "extracted_text": extracted_text,
        }
    )


@app.post(
        path = "/ocr_base64",
        tags = [
            "Tesseract Functions",
        ],
)
async def perform_ocr_base46(
    base64_string: str = Form(...), 
    language: Language = Form(Language.Farsi),
):
    """
    Perform OCR on a base64 encoded image.

    This function handles the OCR process:
    - Accepts a base64 encoded image string and the language to be used for OCR.
    - Decodes the base64 string into image data and converts it to a PIL image.
    - It performs OCR using Tesseract and returns the extracted text in the specified language.
    - If any error occurs during decoding, image processing, or OCR, it raises an 
      appropriate HTTP exception with a message.

    Args:
        base64_string (str): The base64-encoded string representing the image.
        language (language): The language to use for OCR.

    Returns:
        JSONResponse: A JSON response with the extracted text and OCR language.
    """
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the base64 string.",
                "error": str(e),
            },
        )
    
    try:
        extracted_text = pytesseract.image_to_string(image, lang=language.value)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error during OCR",
                "error": str(e),
            },
        )
    
    return JSONResponse(
        content={
            "language": language.name,
            "extracted_text": extracted_text,
        }
    )


@app.post(
        path = "/ocr_base64_json",
        tags = [
            "Tesseract Functions",
        ],
)
async def perform_ocr_base64_json(
    request: OCRJsonRequest
):
    """
    Perform OCR on a base64 encoded image (JSON Input).

    This function handles the OCR process:
    - Accepts a JSON payload with a base64 encoded image string and the language to be used for OCR.
    - Decodes the base64 string into image data and converts it to a PIL image.
    - It performs OCR using Tesseract and returns the extracted text in the specified language.
    - If any error occurs during decoding, image processing, or OCR, it raises an 
      appropriate HTTP exception with a message.

    Args:
        request (OCRJsonRequest): A JSON payload with the base64 string and language.

    Returns:
        JSONResponse: A JSON response with the extracted text and OCR language.
    """
    base64_string = request.base64_string
    ocr_language = request.language

    allowed_langs = {item.value for item in Language}
    try:
        Language(ocr_language)
    except:
        raise HTTPException(
                status_code=422,
                detail={
                    "invalid language entry": ocr_language,
                    "allowed languages": sorted(allowed_langs),
                },
            )

    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the base64 string.",
                "error": str(e),
            },
        )
   
    try:
        extracted_text = pytesseract.image_to_string(image, lang=ocr_language)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error during OCR",
                "error": str(e),
            },
        )

    return JSONResponse(
        content={
            "language": Language(ocr_language).name,
            "extracted_text": extracted_text,
        }
    )


@app.get(
        path = "/get_supported_languages",
        tags = [
            "Tesseract Functions",
        ],
)
async def get_supported_languages():
    """
    Get the list of supported languages for OCR.

    This function returns the available languages that can be used for OCR.
    It extracts the available languages from the 'language' Enum or configuration
    and returns them in a dictionary format.

    Returns:
        dict: A dictionary with the supported languages and their respective codes.
    """
    return JSONResponse(
        {
            "supported_languages": {
                lang.name: lang.value for lang in Language
            }
        }
    )
