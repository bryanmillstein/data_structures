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
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, neighbor, weight=0):
        self.connectedTo[neighbor] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, neighbor):
        return self.connectedTo[neighbor]


def buildGraph():
    d = {}
    graph = Graph()

    lines = open('words.py')
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
    print d
buildGraph()
