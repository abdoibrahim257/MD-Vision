class TreeNode:
    def __init__(self, question_string):
        self.question_string = question_string
        self.children = {}  # dictionary of children nodes key:value are Left:TreeNode and Right: TreeNode

    #setters
    def add_child(self, child, direction): 
        self.children[direction] = child  # child here is a tree node

    def add_Left(self, LeftQuestion):
        self.children['No'] = TreeNode(LeftQuestion)

    def add_Right(self, RightQuestion):
        self.children['Yes'] = TreeNode(RightQuestion)
        
    #getters
    def get_question(self):
        return self.question_string
    
    def get_yes(self):
        return self.children.get('Yes', None)
    
    def get_no(self):
        return self.children.get('No', None)

    def print_node(self):
        print(f"Current Question: {self.question_string}")
        # print children without for loop
        if not self.children:
            return
        print(f"If the answer is No: {self.children.get('No', 'None').question_string}")
        print(f"If the answer is Yes: {self.children.get('Yes', 'None').question_string}")

class anotherTreeNode:
    def __init__(self, root_question, symptom): #not the actual tree but a reference for another tree
        self.root = TreeNode(root_question)
        self.symptom = symptom
        
class TreeDS:
    def __init__(self, root_question, symptom):
        self.root = TreeNode(root_question)
        self.symptom = symptom
        
    #setters    
    def add_node(self, parent_question, newNodeQuestion, direction):
        # find the parent node
        parent_node = self._find_node(parent_question)
        if parent_node:
            # add the child node
            parent_node.add_Left(newNodeQuestion) if direction == 'No' else parent_node.add_Right(newNodeQuestion)
        else:
            print(f"Parent node {parent_question} not found")
            
    def add_tree(self, parent_question, treeRoot, Symptom, direction):
        newTree = anotherTreeNode(treeRoot, Symptom)
        parent_node = self._find_node(parent_question)
        if parent_node:
            parent_node.add_child(newTree, direction)
        else:
            print(f"Parent node {parent_question} not found")
    
    #getters
    def get_root(self):
        return self.root
    
    def get_symptom(self):
        return self.symptom
    
    def _find_node(self, question):
        # search BFS since solution is sparse and tree is not deep
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.question_string == question:
                return node
            for child in node.children.values():
                queue.append(child)
                
                
    def build_tree1(self, jsonObject):
        """
        Recursively build a decision tree from a JSON object.
        """
        if isinstance(jsonObject, dict):
            nodeDetails = jsonObject.get('question', None)
            if nodeDetails:
                parentQ = nodeDetails.get('Q')
                yesNode = nodeDetails.get('Yes')
                noNode = nodeDetails.get('No')
                if yesNode:
                    #get tyoe of next node a question ot a tree
                    if yesNode.get('question', None):
                        self.add_node(parentQ, yesNode['question'].get('Q'), "Yes")
                    else:
                        self.add_tree(parentQ, yesNode['tree'].get('Q'), yesNode['tree'].get('symptom'), "Yes")
                        return
                    self.build_tree1(yesNode)
                else:
                    return
                if noNode:
                    if noNode.get('question', None):
                        self.add_node(parentQ, noNode['question'].get('Q'), "No")
                    else:
                        self.add_tree(parentQ, noNode['tree'].get('Q'), noNode['tree'].get('symptom'), "No")
                        return
                    self.build_tree1(noNode)
                else:
                    return
            else:
                return