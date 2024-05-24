from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from treeTraversal import *
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

print("HERE")
answerList = []

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

# @app.get("/maven/{symptom}/{answer}")
# def getQ(symptom:str,answer:str):
#     answerList.append(answer)
#     answerList_cpy = answerList.copy()
#     return {"Question": traverse_tree2(symptom,warehouse,answerList_cpy),
#             "answerList": answerList
#             }

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

#pip install --upgrade typing-extensions always run this (ABOD ONLY)