""" This module contains the Queue class and its accompanying methods. We will
    use Python's primitive data type, List, to implement the Queue. Note that this
    implementation assumes the front and back of the Queue correspond in such a
    way to the List."""


class Queue():

    def __init__(self):
        self.items = []

    def add(self, item):
        self.items = [item] + self.items

    def remove(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def rebuild(self, elements):
        for element in elements:
            self.items.append(element)

    def search(self, target):
        length = len(self.items)
        current_element = self.remove()
        removed_elements = [current_element]
        found = False

        while self.items and found:
            if current_element == target:
                found = length - len(removed_elements)
            else:
                [current_element] + removed_elements
                current_element = self.remove()

        self.rebuild(removed_elements)
        return """The {0} was found at position {1} in the Queue. This means, the {0} will have to wait {2} removals before he's at the front.""".format(target, found + 1, length - 1- found)
