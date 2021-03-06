""" This module contains the Node and LinkedList class."""

class Node(object):

    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next

class LinkedList:

    def __init__(self, head=None):
        self.head = head

    def insert(data):
        new_node = Node(data)
        new_node.set_next(self.head)
        self.head = new_node

    def size():
        list_size = 0
        current_node = self.head

        while current_node:
            list_size += 1
            current_node = current_node.get_next()

        return list_size

    def search(target_data):
        current_node = self.head

        while current_node:
            if current_node.data == target_data:
                return current_node
            else:
                current_node = current_node.get_next()

        raise ValueError('Data is not in the list. Sorry.')
        return None

    def delete(target_data):
        current_node = self.head
        previous_node = None

        while current_node:
            if current_node.data == target_data:
                if previous_node == None:
                    self.head = current_node.get_next()
                else:
                    previous_node.set_next(current_node.get_next())
                return
            else:
                previous_node = current_node
                current_node = current_node.get_next()

        raise ValueError('Data is not in the list. Sorry.')
        return None


def longestPalindrome(string):
    index = len(string)
    while index > 0:
        startPoint = 0
        while startPoint + index <= len(string):
            if isPalindrome(string[startPoint:startPoint + index]):
                return string[startPoint:startPoint + index]
            startPoint += 1
        index -= 1
    return None

def isPalindrome(string):
    print string
    for i in range(len(string)/2):
        if string[i] != string[len(string) - i - 1]:
            return False
    return True

print longestPalindrome('bbbbaa')
