from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
from predict import *
from treeTraversal import *
from build_vocab import Vocabulary
from pydantic import BaseModel
from colabcode import ColabCode

app = FastAPI()

# Initialize the models
mlc, visual_extractor, sentence_lstm, word_lstm, vocab = initialize_models()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define routes
@app.get("/maven/{symptom}")
def GetfirstQ(symptom: str):
    tree = warehouse.getTree(symptom.lower())
    Q = tree.get_root().get_question()
    answerList.clear()
    return {"Question": Q}

@app.post("/maven/{symptom}")
def getQ(symptom: str, answer: Answer):
    answerList.append(answer.ans)
    answerList_cpy = answerList.copy()
    return {
        "Question": traverse_tree2(symptom, warehouse, answerList_cpy),
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
def startPrediction():
    image_path = "upload/user_image.png"
    image = Image.open(image_path)
    captions = predict(image, mlc, visual_extractor, sentence_lstm, word_lstm, vocab)
    
    return JSONResponse(status_code=200, content={"message": captions})

if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok

    # Start ngrok tunnel
    public_url = ngrok.connect(port=8000)
    print("Public URL:", public_url)

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
