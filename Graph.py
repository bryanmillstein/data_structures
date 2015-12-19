""" This module contains the Graph class."""

class EdgeGraph:

	def __init__(self, edges):
		""" Sets self.__graph_dict based on an input of vertices (list of lists)."""
		self.__graph_dict = {}
		for pair in edges:
			vertices = self.__graph_dict.setdefault(pair[0], [])
			vertices.append(pair[1])

	def vertices(self):
		""" Returns the vertices of the Graph."""
		return self.__graph_dict.keys()

	def edges(self):
		""" Returns the edges of the Graph."""
		edges = []
		for vertex in self.vertices():
			for neighbor in self.__graph_dict[vertex]:
				if [vertex, neighbor] not in edges:
					edges.append([vertex, neighbor])
		return edges

	def add_vertex(self, vertex):
		self.__graph_dict.setdefault(vertex, [])

	def add_edge(self, start, end):
		neighbors = self.__graph_dict.setdefault(start, [])
		if end not in neighbors:
			neighbors.append(end)

	def find_shortest_path(self, start_vertex, end_vertex, path=[]):
		graph = self.__graph_dict
		path = path + [start_vertex]
		shortest_path = None
		if start_vertex not in graph:
			return None
		elif start_vertex == end_vertex:
			return path
		else:
			for vertex in graph[start_vertex]:
				if vertex not in path:
					extended_path = self.find_shortest_path(vertex, end_vertex, path)

					if extended_path:
						if not shortest_path or len(extended_path) < len(shortest_path):
							shortest_path = extended_path
			return shortest_path

	def find_all_paths(self, start_vertex, end_vertex, path=[], paths=[]):
		graph = self.__graph_dict
		path = path + [start_vertex]
		if start_vertex not in graph:
			return None
		elif start_vertex == end_vertex:
			return paths.append(path)
		else:
			for vertex in graph[start_vertex]:
				if vertex not in path:
					extended_path = self.find_all_paths(vertex, end_vertex, path)

					if extended_path:
						paths.append(extended_path)
			return paths
#
# edges = [['A', 1], ['B', 2], ['A', 3], ['C', 4], ['B', 5], ['A', 'B'], ['A', 'C'], ['B', 'C']]
# graph = Graph(edges)
# print graph.find_all_paths('A', 'C')

# Adjacency list implementation.
class Graph():
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices += 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, vert):
        if vert in self.vertList:
            return self.vertList[vert]
        else:
            return None

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


class Vertex():
    def __init__(self, key, distance=0, pred=None):
		self.id = key
		self.connectedTo = {}
		self.color = 'white'
		self.distance = distance
		self.pred = pred

    def addNeighbor(self, neighbor, weight=0):
        self.connectedTo[neighbor] = weight

	def getDistance(self):
		return self.distance

	def setDistance(self, distance):
		self.distance = distance

	def getPred(self):
		return self.pred

	def setPred(self, pred):
		self.pred = pred

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, neighbor):
        return self.connectedTo[neighbor]


def buildGraph():
	d = {}
	graph = Graph()
	lines = open('words.py', 'r')
	for line in lines:
		word = line[1:-2]
		for i in range(len(word)):
			bucket = word[:i] + '_' + word[i+1:]
			if bucket not in d:
				d[bucket] = [word]
			else:
				d[bucket].append(word)

	for bucket in d.keys():
		for word1 in d[bucket]:
			for word2 in d[bucket]:
				if word1 != word2:
					graph.addEdge(word1, word2)
	return graph

def wordLadder(start):
	graph = buildGraph()
	startPos = graph.vertList[start]

	queue = [startPos]
	while len(queue) > 0:
		currentVertex = queue.pop(0)
		for nbr in currentVertex.connectedTo.keys():
			if nbr.color == 'white':
				nbr.color = 'gray'
				nbr.pred = currentVertex
				nbr.distance = currentVertex.distance + 1
				queue.append(nbr)
		currentVertex.color = 'black'
	return graph

def traverse(start, target):
	graph = wordLadder(start)
	x = graph.vertList[target]
	path = []
	while (x):
		path.insert(0,x)
		x = x.pred

	for word in path:
		print word.id

# traverse('fool', 'sage')

def generateMoves(x, y, boardSize):
    jumps = [[1,2], [1,-2], [-1,2], [-1,-2],
            [2,1], [2,-1], [-2,1], [-2,-1]]
    moves = []
    for jump in jumps:
        newX = x + jump[0]
        newY = y + jump[1]

        if (newY >= 0 and newY <= boardSize-1) and (newX >= 0 and newX <= boardSize-1):
            moves.append([newX, newY])
    return moves

def buildKnightsGraph(boardSize):
    knightGraph = Graph()
    for i in range(boardSize*boardSize):
		x = i % 5
		y = i / 5
		moves = generateMoves(x, y, boardSize)
		for move in moves:
			convertedSquare = move[1] + (move[0] * 5)
			knightGraph.addEdge(i, convertedSquare)
    return knightGraph

def knightsTour(vertex, path, numSquares):
	vertex.color = 'gray'
	path.append(vertex)

	if len(path) < numSquares:
		nbrs = vertex.getConnections()
		fullyExplored = False
		index = 0
		while index < len(nbrs) and not fullyExplored:
			if nbrs[index].color == 'white':
				fullyExplored = knightsTour(nbrs[index], path, numSquares)
			index += 1
		if not fullyExplored:
			path.pop()
			vertex.color = 'white'
	else:
		pathIds = [vertex.id for vertex in path]
		return pathIds
	return fullyExplored

knightsGraph = buildKnightsGraph(5)
path = knightsTour(knightsGraph.vertList[0],[],24)
print path
# for vertex in path:
# 	print vertex.id
