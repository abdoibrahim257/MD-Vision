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
        
    #setters    
    def add_node(self, parent_question, newNodeQuestion, direction):
        # find the parent node
        parent_node = self._find_node(parent_question)
        if parent_node:
            # add the child node
            parent_node.add_Left(newNodeQuestion) if direction == 'No' else parent_node.add_Right(newNodeQuestion)
        else:
            print(f"Parent node {parent_question} not found")
            
    #here in this function we need to link the warehouse we just did in the other file
    #another note that we dont need the TreeRoot for anython since the warehouse exists
    #we just need to get the whole tree from the warehouse
    def add_tree(self, parent_question, Symptom, direction):
        newTree = anotherTreeNode(Symptom) #need to get the whole other tree not create a new one 
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
        #lazem ne check lw el node el maskenha de another tree node wla node question
        # w dymn 3ndna el question node how el parent ely bndawra 3aleh
        while queue:
            node = queue.pop(0)
            if node.question_string == question:
                return node
            for child in node.children.values():
                if isinstance(child, TreeNode):
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
                #mmkn ba3d ma n5las build lel trees n3ml el diagnosis w el care 3lashan el chat yb2a more lively
                if yesNode:
                    #get tyoe of next node a question ot a tree
                    if yesNode.get('question', None):
                        self.add_node(parentQ, yesNode['question'].get('Q'), "Yes")
                        self.build_tree1(yesNode)
                    else:
                        self.add_tree(parentQ, yesNode['tree'].get('symptom'), "Yes")
                        # return
                else:
                    return
                if noNode:
                    if noNode.get('question', None):
                        self.add_node(parentQ, noNode['question'].get('Q'), "No")
                        self.build_tree1(noNode)
                    else:
                        self.add_tree(parentQ, noNode['tree'].get('symptom'), "No")
                        # return
                else:
                    return
            else:
                return
            
    def build_tree2(self,parent,Json):

        yesJson = Json['question'].get('Yes')
        noJson = Json['question'].get('No')
        if yesJson:
            if yesJson.get('question', None):
                yesNode = TreeNode(yesJson['question'].get('Q'))
                parent.add_child(yesNode, 'Yes')
                self.build_tree2(yesNode, yesJson)
            else:
                yesNode = anotherTreeNode(yesJson['tree'].get('symptom'))
                parent.add_child(yesNode, 'Yes')
        else: 
            return
        
        if noJson:
            if noJson.get('question', None):
                noNode = TreeNode(noJson['question'].get('Q'))
                parent.add_child(noNode, 'No')
                self.build_tree2(noNode, noJson)
            else:
                noNode = anotherTreeNode(noJson['tree'].get('symptom'))
                parent.add_child(noNode, 'No')
        else:
            return

