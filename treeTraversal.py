from treeDS import TreeNode
from treeDS import TreeDS
import json

class treeWarehouse:
    def __init__(self):
        self.trees = {}

    def addTree(self, tree):
        self.trees[tree.get_symptom()] = tree

    def getTree(self, symptom):
        return self.trees.get(symptom, None)

    def getTrees(self):
        return self.trees

    def getTreeCount(self):
        return len(self.trees)

    def clearTrees(self):
        self.trees = {}


# function to traverse the tree according to user input (yes or no) and return the diagnosis
def traverseTree(tree, warehouse):
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
            tree = warehouse.getTree(current_node.symptom)
            print(f"Symptom: {current_node.symptom}")
            current_node = tree.get_root()
            
warehouse = treeWarehouse()            
           
with open('./unexplained_weight_loss.json') as f:
    data = json.load(f)
unexpectedWeightLoss = TreeDS(data['question'].get('Q'), "Unexpected Weight Loss")
unexpectedWeightLoss.build_tree1(data)

warehouse.addTree(unexpectedWeightLoss)

traverseTree(unexpectedWeightLoss, warehouse)



