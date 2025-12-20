from enum import Enum
from pydantic import BaseModel


class Language(Enum):
    English = "eng"
    Spanish = "spa"
    Arabic = "ara"
    French = "fra"
    German = "deu"
    Farsi = "fas"

class OCRJsonRequest(BaseModel):
    base64_string: str
    language: str
