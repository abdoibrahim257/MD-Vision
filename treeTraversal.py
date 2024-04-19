from treeDS import TreeNode
from treeDS import TreeDS
import json

class treeWarehouse:
    def __init__(self):
        self.trees = {}

    def addTree(self, tree):
        self.trees[tree.get_symptom().lower()] = tree

    def getTree(self, symptom):
        return self.trees.get(symptom.lower(), None)

    def getTrees(self):
        return self.trees

    def getTreeCount(self):
        return len(self.trees)

    def clearTrees(self):
        self.trees = {}


# function to traverse the tree according to user input (yes or no) and return the diagnosis
def traverseTree(symptom, warehouse):
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
            
warehouse = treeWarehouse()            

def add_tree_to_warehouse(file_path, symptom, warehouse):
    with open(file_path) as f:
        data = json.load(f)
    tree = TreeDS(data['question'].get('Q'), symptom)
    tree.build_tree1(data)
    warehouse.addTree(tree)

#FILL THE WAREHOUSE WITH TREES

# unexplained weight loss tree construction
add_tree_to_warehouse('D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/Decision Trees/unexplained_weight_loss.json', "Unexpected Weight Loss", warehouse)

# Abdominal pain tree construction
add_tree_to_warehouse('D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/Decision Trees/Abdominal_pain.json', "Abdominal Pain", warehouse)

#abnormal looking stools tree construction
add_tree_to_warehouse('D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/Decision Trees/Abnormal_looking_stools.json', "Abnormal Looking Stools", warehouse)

# sore throat tree construction
add_tree_to_warehouse('D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/Decision Trees/Sore_Throat.json', "Sore Throat", warehouse)

# hoarseness tree construction
add_tree_to_warehouse('D:/GAM3A/5-Senior02/GP/1-ChatBot/GP-Chatbot/Decision Trees/Hoarseness_or_Loss_of_Voice.json', "hoarseness or loss of voice", warehouse)

traverseTree("sore throat", warehouse)