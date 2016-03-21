import math

def one_away(string1, string2):
    if len(string1) > len(string2):
        check_string = string1
        short_string = string2
    else:
        check_string = string2
        short_string = string1

    for index, letter in enumerate(check_string):
        check = check_string[:index] + check_string[(index + 1):]
        if len(check_string) == len(short_string):
            temp_check = short_string[:index] + short_string[(index + 1):]
            if check == temp_check:
                return True
        elif check == short_string:
            return True
    return False

def reverse_int(integer):
    is_negative = integer < 0

    result = 0
    abs_integer = abs(integer)
    while abs_integer > 9:
        result = (result * 10) + (abs_integer % 10)
        abs_integer /= 10
    result = result * 10 + abs_integer % 10

    if is_negative:
        return result * -1
    else:
        return result

def is_palindrome(integer):
    if integer < 0:
        return False

    num_digits = math.floor(math.log(integer, 10)) + 1

    for i in xrange(1, int(math.ceil(num_digits/2))):
        first = math.floor(integer/(10**(num_digits-i))) % 10
        last = (integer % (10**i))/10**(i-1)
        if not first == last:
            return False
    return True

def intersection(rect1, rect2):
    rect1_minx = None
    rect1_maxx = None
    rect1_miny = None
    rect1_maxy = None

    rect2_minx = None
    rect2_maxx = None
    rect2_miny = None
    rect2_maxy = None

    int_minx = None
    int_maxx = None
    int_miny = None
    int_maxy = None

    for coordinate in rect1:
        if rect1_minx == None or coordinate[0] < rect1_minx:
            rect1_minx = coordinate[0]
        elif rect1_maxx == None or coordinate[0] > rect1_maxx:
            rect1_maxx = coordinate[0]

        if rect1_miny == None or coordinate[1] < rect1_miny:
            rect1_miny = coordinate[1]
        elif rect1_maxy == None or coordinate[1] > rect1_maxy:
            rect1_maxy = coordinate[1]

    for coordinate in rect2:
        if rect2_minx == None or coordinate[0] < rect2_minx:
            rect2_minx = coordinate[0]
        elif rect2_maxx == None or coordinate[0] > rect2_maxx:
            rect2_maxx = coordinate[0]

        if rect2_miny == None or coordinate[1] < rect2_miny:
            rect2_miny = coordinate[1]
        elif rect2_maxy == None or coordinate[1] > rect2_maxy:
            rect2_maxy = coordinate[1]

    # Set the minimum x value if intersection else return false.
    if rect1_minx <= rect2_maxx and rect1_minx >= rect2_minx:
        int_minx = rect1_minx
    elif rect2_minx <= rect1_maxx and rect2_minx >= rect1_minx:
        int_minx = rect2_minx
    else:
        return False

    # Set the maximum x value if intersection else return false.
    if rect1_maxx <= rect2_maxx and rect1_maxx >= rect2_minx:
        int_maxx = rect1_maxx
    elif rect2_maxx <= rect1_maxx and rect2_maxx >= rect1_minx:
        int_maxx = rect2_maxx
    else:
        return False

    # Set the minimum y value if intersection else return false.
    if rect1_miny <= rect2_maxy and rect1_miny >= rect2_miny:
        int_miny = rect1_miny
    elif rect2_miny <= rect1_maxy and rect2_miny >= rect1_miny:
        int_miny = rect2_miny
    else:
        return False

    # Set the maximum y value if intersection else return false.
    if rect1_maxy <= rect2_maxy and rect1_maxy >= rect2_miny:
        int_maxy = rect1_maxy
    elif rect2_maxy <= rect1_maxy and rect2_maxy >= rect1_miny:
        int_maxy = rect2_maxy
    else:
        return False

    return [[int_minx,int_miny],[int_maxx,int_maxy],[int_maxx,int_miny],[int_minx,rect1_maxy]]

def quick_sort(array):
    quick_sort_helper(array, 0, len(array)-1)

def quick_sort_helper(array, start, end):
    if start < end:
        split_point = partition(array, start, end)

        quick_sort_helper(array, start, split_point-1)
        quick_sort_helper(array, split_point+1, end)

def partition(array, start, end):
    # Returns an index.
    pivot_value = array[start]

    left_index = start + 1
    right_index = end

    done = False
    while not done:

        while left_index <= right_index and array[left_index] <= pivot_value:
            left_index += 1

        while right_index >= left_index and array[right_index] >= pivot_value:
            right_index -= 1

        if right_index < left_index:
            done = True
        else:
            array[left_index], array[right_index] = array[right_index], array[left_index]

    array[start], array[right_index] = array[right_index], array[start]
    return right_index

def dutch_sort(array):
    high_index = len(array)-1
    low_index = 0
    placeholder_index = 0


    while low_index <= high_index:
        if array[low_index] == 0:
            array[low_index], array[placeholder_index] = array[placeholder_index], array[low_index]
            low_index += 1
            placeholder_index += 1
        elif array[low_index] == 2:
            array[low_index], array[high_index] = array[high_index], array[low_index]
            high_index -= 1
        else:
            low_index += 1

    return array

def reverse_words(string):
    length = len(string)
    addedCount = 0
    currentWord = ""

    for letter in string:
      if letter != " ":
        currentWord += letter
      else:
        first = string[len(currentWord)+1:length-addedCount]
        last = string[length-addedCount:]
        string = first + " " + currentWord + " " + last
        addedCount += len(currentWord)
        currentWord = ""
    return string

def rotatedSearch(array, start, end, target):
    if end < start:
        return False

    middleIndex = (start+end)/2

    if array[middleIndex] == target:
        return middleIndex
    elif array[middleIndex] > target and array[start] <= target:
        return rotatedSearch(array, 0, middleIndex, target)
    else:
        temp = rotatedSearch(array, middleIndex + 1, end, target)
        if temp:
            return temp
        else:
            return False

""" The following is a brief illustration of using python's
    built in unit testing 'unittest'."""
class VendingMachine():
    def __init__(self):
        self.count = 0
        self.items = {}

    def addItem(self, foodType):
        self.count += 1
        newItem = Item(foodType)
        self.items['A'+ str(self.count)] = newItem.name
        return newItem


class Item():

    def __init__(self, foodType):
        self.name = foodType

""" The following is an example of dynamic programming. It is
    the 0/1 knapsack problem."""

def bestValue(items, weight, i):
    if i < 0:
        return 0

    if items[i][0] > weight:
        return bestValue(items, weight, i - 1)
    else:
        return max(items[i][1] + bestValue(items, weight - items[i][0], i - 1), bestValue(items, weight, i - 1))



def knapsack(items, totalWeight):
    """
    input:
        items: [[W1, V1], [W2, V2], [W3, V3]]
        totalWeight: W
    """

    itemsUsed = []
    for i in range(len(items) - 1, -1, -1):
        if bestValue(items, totalWeight, i) > bestValue(items, totalWeight, i - 1):
            itemsUsed.append(items[i])
            totalWeight -= items[i][0]

    return itemsUsed


""" Print multiplication table."""

def multiTable(tableSize):
    result = []
    for i in range(1, tableSize+1):
        currentRow = []
        for j in range(1, tableSize+1):
            currentRow.append(i*j)
        result.append(currentRow)

    return result

class Stack:

	def __init__(self):
		self.items = []

	def push(self, item):
		self.items.append(item)

	def pop(self):
		return self.items.pop()

	def peek(self):
		return self.items[len(self.items) - 1]

	def is_empty(self):
		return not self.items

	def size(self):
		return len(self.items)


def revstring(mystr):
    stack = Stack()

    for letter in mystr:
        stack.push(letter)

    new_str = []
    while not stack.is_empty():
        new_str.append(stack.pop())

    return "".join(new_str)

class Queue:

    def __init__(self):
        self.items = []

    def enqeue(self, item):
        self.items.append(item)

    def deqeue(self):
        if not self.empty():
            return self.items.pop(0)

    def empty(self):
        return not self.items

    def size(self):
        return len(self.items)

    def peek(self):
        return self.items[0]

def hot_potato(names, n):
    queue = Queue()
    for name in names:
        queue.enqeue(name)


    while queue.size() > 1:
        for i in range(n):
            first = queue.deqeue()
            queue.enqeue(first)
        queue.deqeue()
    return queue.peek()

def is_palindrome(string):
    for i in range(len(string)/2):
        if string[i] != string[len(string) - 1 - i]:
            return False
    return True

def pairs(K, alist):
    all_nums = {}
    for num in alist:
        all_nums[str(num)] = True

    count = 0
    for num in alist:
        try:
            num = all_nums[str(num - K)]
            if num:
                count += 1
        except:
            pass

    return count

# print pairs(2, [1,5,3,4,2]) -> 3
# print pairs(3, [1,5,3,4,2]) -> 2

class Graph:

    def __init__(self):
        self.vertexList = {}

class Vertex:

    def __init__(self, value):
        self.value = value
        self.color = 'white'
        self.pred = None
        self.distance = 1000
        self.connectedTo = []

    def addEdge(self, connectedEdge):
        self.connectedTo.append(connectedEdge)

def buildGraph(vertices):
    graph= Graph()

    for value in vertices:
        newVertex = Vertex(value)
        graph.vertexList[value] = newVertex

    graph.vertexList['A'].addEdge(graph.vertexList['H'], 2)
    graph.vertexList['A'].addEdge(graph.vertexList['B'], 1)
    graph.vertexList['A'].addEdge(graph.vertexList['I'], 50)
    graph.vertexList['B'].addEdge(graph.vertexList['A'], 1)
    graph.vertexList['B'].addEdge(graph.vertexList['C'], 1)
    graph.vertexList['C'].addEdge(graph.vertexList['B'], 1)
    graph.vertexList['C'].addEdge(graph.vertexList['D'], 1)
    graph.vertexList['D'].addEdge(graph.vertexList['C'], 1)
    graph.vertexList['D'].addEdge(graph.vertexList['E'], 10)
    graph.vertexList['D'].addEdge(graph.vertexList['I'], 1)
    graph.vertexList['E'].addEdge(graph.vertexList['F'], 2)
    graph.vertexList['E'].addEdge(graph.vertexList['I'], 1)
    graph.vertexList['E'].addEdge(graph.vertexList['D'], 10)
    graph.vertexList['F'].addEdge(graph.vertexList['I'], 2)
    graph.vertexList['F'].addEdge(graph.vertexList['E'], 2)
    graph.vertexList['F'].addEdge(graph.vertexList['G'], 2)
    graph.vertexList['G'].addEdge(graph.vertexList['H'], 2)
    graph.vertexList['G'].addEdge(graph.vertexList['F'], 2)
    graph.vertexList['H'].addEdge(graph.vertexList['A'], 2)
    graph.vertexList['H'].addEdge(graph.vertexList['G'], 2)
    graph.vertexList['I'].addEdge(graph.vertexList['A'], 50)
    graph.vertexList['I'].addEdge(graph.vertexList['E'], 1)
    graph.vertexList['I'].addEdge(graph.vertexList['F'], 2)
    graph.vertexList['I'].addEdge(graph.vertexList['D'], 1)

    return graph

def setPredsWeighted(start, end, G):
    startVertex = G.vertexList[start]
    searchQueue = [startVertex]

    startVertex.distance = 0
    done = False
    while len(searchQueue) > 0 and not done:
        currentVertex = searchQueue.pop(0)
        for nbr in currentVertex.connectedTo:
            weight = currentVertex.connectedTo[nbr]
            if nbr.distance > (currentVertex.distance + weight):
                nbr.pred = currentVertex
                nbr.distance = currentVertex.distance + weight
                # print("{0}, {1}".format(nbr.value, nbr.pred.value))
                searchQueue.append(nbr)

    return G

def setPredsUnweighted(start, end, G):
    startVertex = G.vertexList[start]
    searchQueue = [startVertex]

    startVertex.distance = 0
    done = False
    while len(searchQueue) > 0 and not done:
        currentVertex = searchQueue.pop(0)
        currentVertex.color = 'gray'
        for nbr in currentVertex.connectedTo.keys():
            if nbr.color == 'white':
                nbr.pred = currentVertex
                # print("{0}, {1}".format(nbr.value, nbr.pred.value))
                searchQueue.append(nbr)

    return G

def shortestPath(start, end, G):
    x = end

    while x != start:
        vertex = G.vertexList[x]
        print vertex.pred.value
        x = vertex.pred.value

# graph = buildGraph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
#
# graph_paths = setPredsWeighted('A', 'E', graph)
#
# shortestPath('A', 'E', graph_paths)


def buildBoggleGraph(vertices):
    graph= Graph()

    for value in vertices:
        newVertex = Vertex(value)
        graph.vertexList[value] = newVertex

    graph.vertexList['C'].addEdge(graph.vertexList['A'])
    graph.vertexList['C'].addEdge(graph.vertexList['O'])
    graph.vertexList['C'].addEdge(graph.vertexList['D'])

    graph.vertexList['A'].addEdge(graph.vertexList['C'])
    graph.vertexList['A'].addEdge(graph.vertexList['T'])
    graph.vertexList['A'].addEdge(graph.vertexList['D'])
    graph.vertexList['A'].addEdge(graph.vertexList['O'])
    graph.vertexList['A'].addEdge(graph.vertexList['G'])

    graph.vertexList['T'].addEdge(graph.vertexList['A'])
    graph.vertexList['T'].addEdge(graph.vertexList['O'])
    graph.vertexList['T'].addEdge(graph.vertexList['G'])

    graph.vertexList['D'].addEdge(graph.vertexList['C'])
    graph.vertexList['D'].addEdge(graph.vertexList['A'])
    graph.vertexList['D'].addEdge(graph.vertexList['O'])
    graph.vertexList['D'].addEdge(graph.vertexList['L'])
    graph.vertexList['D'].addEdge(graph.vertexList['R'])

    graph.vertexList['O'].addEdge(graph.vertexList['C'])
    graph.vertexList['O'].addEdge(graph.vertexList['A'])
    graph.vertexList['O'].addEdge(graph.vertexList['T'])
    graph.vertexList['O'].addEdge(graph.vertexList['D'])
    graph.vertexList['O'].addEdge(graph.vertexList['G'])
    graph.vertexList['O'].addEdge(graph.vertexList['R'])
    graph.vertexList['O'].addEdge(graph.vertexList['L'])
    graph.vertexList['O'].addEdge(graph.vertexList['P'])

    graph.vertexList['G'].addEdge(graph.vertexList['T'])
    graph.vertexList['G'].addEdge(graph.vertexList['A'])
    graph.vertexList['G'].addEdge(graph.vertexList['O'])
    graph.vertexList['G'].addEdge(graph.vertexList['R'])
    graph.vertexList['G'].addEdge(graph.vertexList['P'])

    graph.vertexList['L'].addEdge(graph.vertexList['D'])
    graph.vertexList['L'].addEdge(graph.vertexList['O'])
    graph.vertexList['L'].addEdge(graph.vertexList['O'])

    graph.vertexList['R'].addEdge(graph.vertexList['T'])
    graph.vertexList['R'].addEdge(graph.vertexList['D'])
    graph.vertexList['R'].addEdge(graph.vertexList['O'])
    graph.vertexList['R'].addEdge(graph.vertexList['G'])
    graph.vertexList['R'].addEdge(graph.vertexList['P'])

    graph.vertexList['P'].addEdge(graph.vertexList['G'])
    graph.vertexList['P'].addEdge(graph.vertexList['O'])
    graph.vertexList['P'].addEdge(graph.vertexList['G'])

    return graph

def find_words(board, all_words, current_word="", words=[], start_vertex=None):
    if start_vertex == None:
        start_vertex = board.vertexList[board.vertexList.keys()[0]]
    queue = [start_vertex]

    while len(queue) > 0:
        current_vertex = queue.pop(0)
        current_vertex.color = 'gray'
        if current_word == None:
            current_word = current_vertex.value
        # print current_word
        try:
            word_try = all_words[current_word]
            if word_try:
                print current_word
                words.append(current_word)
        except:
            pass

        for nbr in current_vertex.connectedTo:
            if nbr.color == 'white':
                current_word += nbr.value
                next_words = find_words(board, all_words, current_word, words, nbr)
                words = words + next_words
                current_word=""
        current_vertex.color = 'white'

    return words

graph = buildBoggleGraph(['A', 'C', 'T', 'D', 'O', 'G', 'L', 'R', 'P'])

all_words = {}
lines = open('boggle_words.py', 'r')

for line in lines:
    all_words[line[1:-2].upper()] = True

# print all_words
print find_words(graph, all_words)
