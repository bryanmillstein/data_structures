""" This module contains the Stack class and its accompanying methods. We will
    use Python's primitive data type, List, to implement the Stack. Note that this
    implementation assumes the top of the Stack is the end of the list."""

class Stack():

    def __init__(self):
        self.items = []

    def is_empty(self):
        return not self.items

    def size(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)

    def remove(self):
        return self.items.pop()

    def rebuild(self, elements):
        for element in elements:
            self.items.append(element)

    def search(self, target):
        """ Returns the index of the target if present, else
        returns None. Adheres to only being able to add and
        remove one element at a time."""

        length = len(self.items)
        current_element = self.remove()
        removed_elements = [current_element]
        found = False

        while self.items and not found:
            if current_element == target:
                found = length - len(removed_elements)
            else:
                current_element = self.remove()
                removed_elements = [current_element] + removed_elements

        self.rebuild(removed_elements)
        return found
