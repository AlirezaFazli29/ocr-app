import io
import base64
import asyncio
import tempfile
from PIL import (
    Image,
    UnidentifiedImageError,
)
from fastapi import (
    Form,
    FastAPI,
    UploadFile,
    HTTPException,
)
from .schemas import OCRJsonRequestD
from ..core.engine import DeepSeekOCR
from fastapi.responses import JSONResponse


model = DeepSeekOCR()
model_lock = asyncio.Lock()

app = FastAPI(title="DeepSeek OCR Service")


@app.get(
        path = "/",
        tags = [
            "Health",
        ],
)
async def root():
    """
    Root endpoint for the deepseek OCR service.

    This function serves as a health check for the application.
    When a GET request is made to the root endpoint ("/"), 
    it returns a simple message indicating that the OCR service is running.

    Returns:
        dict: A dictionary with a message confirming that the service is running.
    """
    return JSONResponse(
        {
            "message": "DeepSeek OCR Service is running!"
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
            "DeepSeek Functions",
        ],
)
async def perform_ocr(
    file: UploadFile,
):
    """
    Perform OCR on an uploaded image.

    This function handles the OCR process:
    - Accepts an image file for OCR.
    - It reads the uploaded image, performs OCR using DeepSeek model, 
      and returns the extracted text in the predicted language.
    - If any error occurs during image processing or OCR, 
      it raises an appropriate HTTP exception with a message.

    Args:
        file (UploadFile): The image file to perform OCR on.

    Returns:
        JSONResponse: A JSON response with the extracted text and OCR language.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        suffix = f".{image.format.lower()}" if image.format else ""
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the uploaded file.",
                "error": str(e),
            },
        )
    
    with tempfile.NamedTemporaryFile(
        delete=True, suffix=suffix,
    ) as tmp:
        tmp.write(image_data)
        tmp.flush()

        try:
            async with model_lock:
                res = await asyncio.to_thread(
                    model.infer,
                    tmp.name
                )
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
            "extracted_text": res,
        }
    )



@app.post(
        path = "/ocr_base64",
        tags = [
            "DeepSeek Functions",
        ],
)
async def perform_ocr_base46(
    base64_string: str = Form(...), 
):
    """
    Perform OCR on a base64 encoded image.

    This function handles the OCR process:
    - Accepts a base64 encoded image string for OCR.
    - Decodes the base64 string into image data and converts it to a PIL image.
    - It performs OCR using DeepSeek and returns the extracted text in the predicted language.
    - If any error occurs during decoding, image processing, or OCR, it raises an 
      appropriate HTTP exception with a message.

    Args:
        base64_string (str): The base64-encoded string representing the image.

    Returns:
        JSONResponse: A JSON response with the extracted text.
    """
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        suffix = f".{image.format.lower()}" if image.format else ""
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the uploaded file.",
                "error": str(e),
            },
        )
    
    with tempfile.NamedTemporaryFile(
        delete=True, suffix=suffix,
    ) as tmp:
        tmp.write(image_data)
        tmp.flush()

        try:
            async with model_lock:
                res = await asyncio.to_thread(
                    model.infer,
                    tmp.name
                )
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
            "extracted_text": res,
        }
    )


@app.post(
        path = "/ocr_base64_json",
        tags = [
            "DeepSeek Functions",
        ],
)
async def perform_ocr_base64_json(
    request: OCRJsonRequestD
):
    """
    Perform OCR on a base64 encoded image (JSON Input).

    This function handles the OCR process:
    - Accepts a JSON payload with a base64 encoded image string for OCR.
    - Decodes the base64 string into image data and converts it to a PIL image.
    - It performs OCR using DeepSeek and returns the extracted text in the predicted language.
    - If any error occurs during decoding, image processing, or OCR, it raises an 
      appropriate HTTP exception with a message.

    Args:
        request (OCRJsonRequestD): A JSON payload with the base64 string.

    Returns:
        JSONResponse: A JSON response with the extracted text.
    """
    base64_string = request.base64_string

    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        suffix = f".{image.format.lower()}" if image.format else ""
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not process the uploaded file.",
                "error": str(e),
            },
        )
    
    with tempfile.NamedTemporaryFile(
        delete=True, suffix=suffix,
    ) as tmp:
        tmp.write(image_data)
        tmp.flush()

        try:
            async with model_lock:
                res = await asyncio.to_thread(
                    model.infer,
                    tmp.name
                )
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
            "extracted_text": res,
        }
    )
