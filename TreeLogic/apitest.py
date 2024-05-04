from fastapi import FastAPI
from treeTraversal import *

app = FastAPI()

print("HERE")
answerList = []

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