import json
import pickle
from typing import *
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import HTTPException

# TTS
from TTS.utils.synthesizer import Synthesizer

from pydantic import BaseModel

# IndicTrans
from src.indicTrans.inference.engine import IndicTransModel

# Indic TTS
from src.tts.inference import TextToSpeechEngine
from src.tts.models.request import TTSRequest

# En-Indic
en2indic_model = IndicTransModel(expdir="models/en-indic")

TTS_SUPPORTED_LANGUAGES = {
    'as' : "Assamese - অসমীয়া",
    'bn' : "Bangla - বাংলা",
    'brx': "Boro - बड़ो",
    'en' : "Indian English",
    'gu' : "Gujarati - ગુજરાતી",
    'hi' : "Hindi - हिंदी",
    'kn' : "Kannada - ಕನ್ನಡ",
    'ml' : "Malayalam - മലയാളം",
    'mni': "Manipuri - মিতৈলোন",
    'mr' : "Marathi - मराठी",
    'or' : "Oriya - ଓଡ଼ିଆ",
    'pa' : "Punjabi - ਪੰਜਾਬੀ",
    'raj': "Rajasthani - राजस्थानी",
    'ta' : "Tamil - தமிழ்",
    'te' : "Telugu - తెలుగు",
}

INDIC_TTS_SUPPORTED_LANGUAGES = ["as", "hi", "mr", "ta", "bn", "kn", "or", "te", "gu", "ml", "pa"]

models = {}
for lang in TTS_SUPPORTED_LANGUAGES:
    models[lang]  = Synthesizer(
        tts_checkpoint=f'models/tts/{lang}/fastpitch/best_model.pth',
        tts_config_path=f'models/tts/{lang}/fastpitch/config.json',
        tts_speakers_file=f'models/tts/{lang}/fastpitch/speakers.pth',
        # tts_speakers_file=None,
        tts_languages_file=None,
        vocoder_checkpoint=f'models/tts/{lang}/hifigan/best_model.pth',
        vocoder_config=f'models/tts/{lang}/hifigan/config.json',
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )
    print(f"Synthesizer loaded for {lang}.")
    print("*"*100)

engine = TextToSpeechEngine(models)

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
def homepage():
    return "Y4J System is live"

@api.get("/supported_languages")
def get_supported_languages():
    return TTS_SUPPORTED_LANGUAGES


@api.post("/")
async def batch_tts(request: TTSRequest, response: Response):
    return engine.infer_from_request(request)

# Indic Trans

class requestSchemaTargetTranslate(BaseModel):
    data: str
    targetLan: str
    currLan: str


class requestSchemaAllTranslate(BaseModel):
    data: str


# Target Language Translation
@api.get("/targetTranslate")
async def targetTranslate(request: requestSchemaTargetTranslate):
    request = request.dict()

    data = request["data"]
    data = [data]
    currLan = request["currLan"]
    targetLan = request["targetLan"]
    translationResult = en2indic_model.batch_translate(data, currLan, targetLan)

    result = {
        "success": True,
        "message": "Target Translation from " + currLan + " to " + targetLan,
        "data": translationResult,
    }
    # Return the response
    return result


# All Language Translation at once
@api.get("/allTranslate")
async def allTranslate(request: requestSchemaAllTranslate):
    request = request.dict()

    # Get the data from the request
    data = request["data"]
    data = [data]
    currLan = "en"

    res = {
        "as": "",
        "hi": "",
        "mr": "",
        "ta": "",
        "bn": "",
        "kn": "",
        "or": "",
        "te": "",
        "gu": "",
        "ml": "",
        "pa": "",
    }

    # Translate the data in all the target languages
    for lang in INDIC_TTS_SUPPORTED_LANGUAGES:
        result = en2indic_model.batch_translate(data, currLan, lang)
        res[lang] = result[0]

    # Standard return format
    result = {
        "success": True,
        "message": "All Translation from " + currLan + " to " + str(INDIC_TTS_SUPPORTED_LANGUAGES),
        "data": res,
    }
    # Return the response
    return result

if __name__ == "__main__":
    # uvicorn main:api --host 0.0.0.0 --port 5050 --log-level info
    uvicorn.run("main:api", host="0.0.0.0", port=5050, log_level="info")
