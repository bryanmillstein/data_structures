""" This module contains the BinaryTree class which is made up of nodes. Each node
    contains its own cargo and a reference to left, right, both or no nodes. Nodes of
    BinaryTree that do not contain references to other nodes are called leaves."""

class BinaryTree:
    def __init__(self, cargo, left=None, right=None):
        self.cargo = cargo
        self.left  = left
        self.right = right

    def __str__(self):
    	return str(self.cargo)

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
