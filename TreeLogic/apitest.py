from fastapi import FastAPI
from treeTraversal import *

app = FastAPI()

#our start of the chat is at /maven/sypmtom
@app.get("/")
def folderName():
    tree =  warehouse.getTree("abnormally frequent urination")
    return {"First Question": tree.get_root().question_string}
