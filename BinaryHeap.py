""" This module contains the BinaryHeap class."""

class BinaryHeap():
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def insert(self, node):
        self.heapList.append(node)
        self.currentSize += 1
        self.percUp(self.currentSize)

    def percUp(self, index):
        done = False
        while index > 1 and not done:
            parent_index = index / 2
            parent = self.heapList[parent_index]

            if self.heapList[index] < parent:
                self.heapList[index], parent = parent, self.heapList[index]
                index = index / 2
            else:
                done = True

    def deleteMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.heapList.pop()
        self.currentSize -= 1

        self.percDown(1)
        return retval

    def percDown(self):
        index = 1
        done = False
        while not done and index*2 <= self.currentSize:
            minChildIndex = self.minChild(index)

            if self.heapList[index] > self.heapList[minChildIndex]:
                self.heapList[index], self.heapList[minChildIndex] = self.heapList[minChildIndex], self.heapList[index]
                index = minChildIndex
            else:
                done = True


    def minChild(self, index):
        if self.heapList[(index*2) + 1] > self.currentSize:
            return index*2

        if self.heapList[index*2] < self.heapList[(index*2) + 1]:
            return index*2
        else:
            return (index*2) + 1

""" The following builds a binary heap from an unordered array."""
def buildHeap(array):
    middleIndex = len(array)/2
    heap = BinaryHeap()
    heap.heapList += array[:]
    heap.currentSize = len(array)

    while middleIndex > 0:
        heap.percDown(i)
        i -= 1

