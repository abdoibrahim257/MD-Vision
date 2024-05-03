from fastapi import FastAPI
from treeDS import TreeNode
from treeDS import TreeDS
import json
import os
import re
from treeTraversal import traverse_tree, load_warehouse,treeWarehouse
from pydantic import BaseModel

app = FastAPI()

symptoms={
    1:{
        "name":"S1"
    },
    2:{
        "name":"S2"
    },
    3:{
    "name":"S3"
    }

}
Responseslist = []


class input(BaseModel):
    text:str

class output(BaseModel):
    tree:object

@app.get("/")
def test():
    return {"alo":"bingus"}

@app.get('/symptoms/{symptom}')

def test2(symptom:int):
    return symptoms[symptom]

@app.post("/chatbot",response_model=input)

async def traverse(request:input):
    
    warehouse = treeWarehouse()
    
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    folder = current_directory + '\\Decision Trees'
    folder = folder.replace('\\', '/')
    
    load_warehouse(folder , warehouse)
    try:
        question = request.text
        tree = warehouse.getTree(question.lower())
        return {"text":tree}
        # answer = traverse_tree(question ,warehouse)
    except Exception as e:
        return {"error": str(e)}
        