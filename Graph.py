""" This module contains the Graph class."""

class Graph:

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
					extended_path = self.find_path(vertex, end_vertex, path)

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

edges = [['A', 1], ['B', 2], ['A', 3], ['C', 4], ['B', 5], ['A', 'B'], ['A', 'C'], ['B', 'C']]
graph = Graph(edges)
print graph.find_all_paths('A', 'C')