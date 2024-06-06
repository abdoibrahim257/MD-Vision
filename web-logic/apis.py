from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import os
import pandas as pd

from predict import *
from treeTraversal import *
from kengic_main import *
from build_vocab import Vocabulary

from pydantic import BaseModel

app = FastAPI()

# Initialize the models

ngrams_dfs = initialize_kengic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

print("HERE")
# vocab = joblib.load('./Data/vocab.pkl')
answerList = []
mlc = None
visual_extractor = None
sentence_lstm = None
word_lstm = None
vocab = None

class Answer(BaseModel):
    ans: str

class response(BaseModel):
    fileList: list[str]
    

#our start of the chat is at /maven/sypmtom
@app.get("/maven/{symptom}")
def GetfirstQ(symptom: str):
    tree =  warehouse.getTree(symptom.lower())
    Q = tree.get_root().get_question()
    answerList.clear()
    return {"Question": Q}


@app.post("/maven/{symptom}")
def getQ(symptom:str,answer:Answer):
    answerList.append(answer.ans)
    answerList_cpy = answerList.copy()
    return {"Question": traverse_tree2(symptom,warehouse,answerList_cpy),
            "answerList": answerList
            }

@app.get("/maven")
def getFiles():
    files = get_tree_names()
    return {"fileList": files}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = os.path.splitext(file.filename)[1]  # Extract file extension
        file_path = f"./upload/user_image{file_extension}"
        
        with open(file_path, 'wb') as f:
            f.write(contents)

        return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred while uploading the file"})

@app.get("/upload")
def startOrInit(init: int, model: str):
    def initialise(model: str):
        global mlc, visual_extractor, sentence_lstm, word_lstm , vocab
        mlc, visual_extractor, sentence_lstm, word_lstm , vocab = initialize_models(model)
        if all(var is not None for var in [mlc, visual_extractor, sentence_lstm, word_lstm, vocab]):
            return JSONResponse(status_code=200, content={"message": "Models initialized successfully"})
        else:
            return JSONResponse(status_code=500, content={"message": "An error occurred while initializing the models"})
        
    def startPrediction(model : str):
        image_path = "upload/user_image.png"
        image = Image.open(image_path)
        
        #check if any of mlc, visual_extractor, sentence_lstm, word_lstm, vocab is None
        if any(var is None for var in [mlc, visual_extractor, sentence_lstm, word_lstm, vocab]):
            return JSONResponse(status_code=500, content={"message": "Models not initialized"})
        
        if (model == "coAtt"):
            captions = predict(image, mlc, visual_extractor, sentence_lstm, word_lstm, vocab)
            return JSONResponse(status_code=200, content={"message": captions})
        elif(model == "kengic"):
            captions = kengic_main(image, mlc, visual_extractor, ngrams_dfs)
            return JSONResponse(status_code=200, content={"message": captions})
        else:
            return JSONResponse(status_code=500, content={"message": "An error occurred while starting the prediction"})
    
    return startPrediction(model) if init == 1 else initialise(model)
    
# def blabla(init):
#     def init():
#         print("HERE")
#     def predict():
#         print("HERE")
    
#     return init() if init else predict()