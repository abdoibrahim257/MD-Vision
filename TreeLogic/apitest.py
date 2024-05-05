from fastapi import FastAPI
from treeTraversal import *
from pydantic import BaseModel

app = FastAPI()

print("HERE")
answerList = []

class Answer(BaseModel):
    ans: str

class response(BaseModel):
    fileList: list[str]

#our start of the chat is at /maven/sypmtom
@app.get("/chatbot/{symptom}")
def GetfirstQ(symptom: str):
    tree =  warehouse.getTree(symptom.lower())
    Q = tree.get_root().get_question()
    answerList.clear()
    return {"Question": Q}

@app.get("/chatbot/{symptom}/{answer}")
def getQ(symptom:str,answer:str):
    answerList.append(answer)
    answerList_cpy = answerList.copy()
    return {"Question": traverse_tree2(symptom,warehouse,answerList_cpy),
            "answerList": answerList
            }

@app.post("/chatbot/{symptom}")
def getQ(symptom:str,answer:Answer):
    answerList.append(answer.ans)
    answerList_cpy = answerList.copy()
    return {"Question": traverse_tree2(symptom,warehouse,answerList_cpy),
            "answerList": answerList
            }

@app.get("/chatbot/files",response_model=response)
def getFiles():
    files = get_tree_names()
    return {"fileList": files}