""" This module contains the BinaryTree class which is made up of nodes. Each node
    contains its own cargo and a reference to left, right, both or no nodes. Nodes of
    BinaryTree that do not contain references to other nodes are called leaves."""

class BinaryTree():
    def __init__(self, root):
        self.key = root
        self.left = None
        self.right = None

    def getRootVal(self):
        return self.key

    def setRootVal(self, val):
        self.key = val

    def getRightChild(self):
        return self.right

    def getLeftChild(self):
        return self.left

    def insertLeft(self, newNode):
        if self.left:
            newTree = BinaryTree(newNode)
            newTree.left = self.left
            self.left = newTree
        else:
            self.left = BinaryTree(newNode)

    def insertRight(self, newNode):
        if self.right:
            newTree = BinaryTree(newNode)
            newTree.right = self.right
            self.right = newTree
        else:
            self.right = BinaryTree(newNode)


def parseTree(expression):
    expression_list = expression.split()
    tree = BinaryTree('')
    current = tree
    tree_list = []
    tree_list.append(current)

    for item in expression_list:
        if item == '(':
            current.insertLeft('')
            tree_list.append(current)
            current = current.getLeftChild()
        elif item not in ['+', '-', '*', '-', ')', '(']:
            current.setRootVal(int(item))
            current = tree_list.pop()
        elif item in ['+', '-', '*', '-']:
            current.setRootVal(item)
            current.insertRight('')
            tree_list.append(current)
            current = current.getRightChild()
        elif item == ')':
            current = tree_list.pop()
        else:
            raise 'Error'
    return tree

def inorder(tree):
  if tree != None:
      inorder(tree.getLeftChild())
      print(tree.getRootVal())
      inorder(tree.getRightChild())

def evaluate(parsed_tree):
    import operator

    opers = {
        '+': operator.add,
        '-': operator.sub,
        '/': operator.truediv,
        '*': operator.mul
    }

    leftVal = parsed_tree.getLeftChild()
    rightVal = parsed_tree.getRightChild()

    if leftVal and rightVal:
        oper = opers[parsed_tree.getRootVal()]
        return oper(evaluate(leftVal), evaluate(rightVal))
    else:
        return parsed_tree.getRootVal()
            
def print_tree_preorder(tree):
	""" Preorder tree traversal."""
	if tree == None: 
		return

	print tree.cargo
	print_tree(tree.left)
	print_tree(tree.right)

def print_tree_postorder(tree):
	""" Postorder tree traversal."""

	if tree == None: 
		return
	print_tree_postorder(tree.left)
	print_tree_postorder(tree.right)
	print tree.cargo

def print_tree_inorder(tree):
	""" Inorder tree traversal."""

	if tree == None: 
		return
	print_tree_inorder(tree.left)
	print tree.cargo
	print_tree_inorder(tree.right)

""" The following creates a binary tree from a list of lists. 
    Each new node contains the following pattern, [root, [], []]
    The empty arrays on the right of the node represent the 
    left and right subtrees."""

""" Better visual example of binary tree created with array of arrays."""
tree = ['root',
            ['left_sub',
                ['left sub sub', [], []],
                ['left right sub', [], []]],
            ['right_sub',
                ['right left sub', [], []],
                ['right sub sub', [], []]]
        ]

def BinaryTree(root):
    return [root, [], []]

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]


def insertLeft(root, node):
    temp = root.pop(1)
    if len(temp) > 1:
        root.insert(1, [node, temp, [] ])
    else:
        root.insert(1, [node, [], [] ])


def insertRight(root, node):
    temp = root.pop(2)
    if len(temp) > 1:
        root.insert(2, [node, [], temp ])
    else:
        root.insert(2, [node, [], [] ])
