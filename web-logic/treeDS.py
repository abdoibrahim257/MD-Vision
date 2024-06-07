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
    def __init__(self, symptom): #not the actual tree but a reference for another tree
        self.symptom = symptom.lower()
        
class TreeDS:
    def __init__(self, root_question, symptom):
        self.root = TreeNode(root_question)
        self.symptom = symptom.lower()
        
    #getters
    def get_root(self):
        return self.root
    
    def get_symptom(self):
        return self.symptom
    
    def build_tree(self,parent,Json):
        yesJson = Json['question'].get('Yes')
        noJson = Json['question'].get('No')
        if yesJson:
            if yesJson.get('question', None):
                yesNode = TreeNode(yesJson['question'].get('Q'))
                parent.add_child(yesNode, 'Yes')
                self.build_tree(yesNode, yesJson)
            else:
                yesNode = anotherTreeNode(yesJson['tree'].get('symptom'))
                parent.add_child(yesNode, 'Yes')
        else: 
            return
        
        if noJson:
            if noJson.get('question', None):
                noNode = TreeNode(noJson['question'].get('Q'))
                parent.add_child(noNode, 'No')
                self.build_tree(noNode, noJson)
            else:
                noNode = anotherTreeNode(noJson['tree'].get('symptom'))
                parent.add_child(noNode, 'No')
        else:
            return
    