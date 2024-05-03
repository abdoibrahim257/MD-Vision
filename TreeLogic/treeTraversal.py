from treeDS import TreeNode
from treeDS import TreeDS
import json
import os
import re

class treeWarehouse:
    def __init__(self):
        self.treesDict = {}

    def addTree(self, tree):
        symptom = tree.get_symptom().lower()
        self.treesDict[symptom] = tree

    def getTree(self, symptom):
        return self.treesDict.get(symptom.lower(), None)

    def getTrees(self):
        return self.treesDict

    def getTreeCount(self):
        return len(self.treesDict)

    def clearTrees(self):
        self.treesDict = {}


# function to traverse the tree according to user input (yes or no) and return the diagnosis
def traverse_tree(symptom, warehouse):
    tree = warehouse.getTree(symptom.lower())
    current_node = tree.get_root()
    while current_node:
        #check type of current node class to determine if it is a tree node or another tree node
        if isinstance(current_node, TreeNode):
            if not current_node.get_yes():
                print("Diagnosis: ", current_node.get_question())
                break
            print(current_node.get_question())
            answer = input("Enter Yes or No: ")
            if answer.lower() == 'yes':
                current_node = current_node.get_yes()
            elif answer.lower() == 'no':
                current_node = current_node.get_no()
            else:
                print("Invalid input. Please enter Yes or No.")
        else:
            #get this tree from the warehouse
            tree = warehouse.getTree(current_node.symptom.lower())
            print(f"Symptom: {current_node.symptom}")
            current_node = tree.get_root()
                        

def add_tree_to_warehouse(file_path, symptom, warehouse):
    with open(file_path) as f:
        data = json.load(f)
    tree = TreeDS(data['question'].get('Q'), symptom)
    tree.build_tree(tree.get_root(),data)
    warehouse.addTree(tree)

def load_warehouse(folder, warehouse):
    for file in os.listdir(folder):
        #check if file is a json file and call the function to add it to the warehouse
        if file.endswith('.json'):
            #get file name without extension
            symptom = os.path.splitext(file)[0]
            sypmtomWithSpaces = re.sub(r"(_)"," ",symptom) #replace underscore with space to get symptom name
            if sypmtomWithSpaces not in warehouse.getTrees():
                add_tree_to_warehouse(os.path.join(folder, file), sypmtomWithSpaces, warehouse)
            
warehouse = treeWarehouse()    
#main folder for all trees    
folder= 'D:/Uni/Senior 2/Semester 2/GP/decision tree test/GP-Chatbot/TreeLogic/Decision Trees'
# folder= 'D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/TreeLogic/Decision Trees/'

# #FILL THE WAREHOUSE WITH TREES
# load_warehouse(folder, warehouse)

# traverse_tree("Coughing", warehouse)