from fastapi import FastAPI
from treeTraversal import *

app = FastAPI()

print("HERE")
answerList = ["yes", "no", "yes"]

#our start of the chat is at /maven/sypmtom
@app.get("/chatbot/{symptom}")
def GetfirstQ(symptom: str):
    tree =  warehouse.getTree(symptom.lower())
    Q = tree.get_root().get_question()
    # answerList = []
    
    return {"Question": Q}

@app.get("/chatbot/{symptom}/{answer}")
def getYes():
    answerList.append("Yes")
    return {"answerList": answerList}