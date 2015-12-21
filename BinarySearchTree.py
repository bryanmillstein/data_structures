""" This module contains the BinarySearchTree and TreeNode classes.
	The BinarySearchTree holds a reference to the root of the tree 
	and is initialized as an empty tree."""


class BinarySearchTree():

	def __init__(self):
		self.root = None
		self.size = 0

	def length(self):
		return self.size

	def __iter__(self):
		return self.root.__iter__()

	def put(self, key, val):
		if not self.root:
			self.root = TreeNode(key, val)
		else:
			self._put(key, val, self.root)
		self.size += 1

	def _put(self, key, val, currentNode):
		if key < currentNode.key:
			if currentNode.hasLeftChild():
				self._put(key, val, currentNode.leftChild)
			else:
				newNode = TreeNode(key, val)
				currentNode.leftChild = newNode
				newNode.parent = currentNode
		else:
			if currentNode.hasRightChild():
				self._put(key, val, currentNode.rightChild)
			else:
				newNode = TreeNode(key, val)
				currentNode.rightChild = newNode
				newNode.parent = currentNode

	def get(self, key):
		if self.root:
			if self.root.key == key:
				return self.root.val
			elif key < self.root.key:
				return self._get(key, self.root.leftChild)
			else:
				return self._get(key, self.root.rightChild)
		else:
			return None

	def _get(self, key, currentNode):
		if currentNode == None:
			return None
		elif key == currentNode.key:
			return currentNode.val
		elif key < currentNode.key:
			if currentNode.hasLeftChild():
				return self._get(key, currentNode.leftChild)
		else:
			if currentNode.hasRightChild():
				return self._get(key, currentNode.rightChild)

	def __contains__(self, key):
		if self.get(key):
			return True
		else:
			return False


class TreeNode():

	def __init__(self, key, val, leftChild=None, rightChild=None, parent=None):
		self.key = key
		self.val = val
		self.leftChild = leftChild
		self.rightChild = rightChild
		self.parent = parent

	def isLeftChild(self):
		return self.parent and self.parent.leftChild == self

	def isRightChild(self):
		return self.parent and self.parent.rightChild == self

	def hasLeftChild(self):
		return self.leftChild

	def hasRightChild(self):
		return self.rightChild

	def hasAnyChildren(self):
		return self.leftChild and self.rightChild

	def hasOneChild(self):
		return self.leftChild or self.rightChild



myTree = BinarySearchTree()
myTree.put(99, 'Robert')
myTree.put(33, 'Rachel')
myTree.put(78, 'Bobby')
myTree.put(45, 'Jonah')
print 99 in myTree
