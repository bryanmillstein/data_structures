""" This module contains the Trie class."""

class Trie():

	def __init__(self, *words):
		""" The initialize builds the Trie with nested dictionaries 
		    from an argument of words."""

		root = dict()
		for word in words:
			current_dict = root
			for letter in word:
				current_dict = current_dict.setdefault(letter, {})
			current_dict['_end_'] = '_end_'
		self.trie = root

	def search(self, word):
		current_dict = self.trie
		for letter in word:
			if letter in current_dict:
				current_dict = current_dict[letter]
			else:
				return False
		else:
			if '_end_' in current_dict:
				return True
			else:
				return False

	def insert(self, word):
		current_dict = self.trie
		for letter in word:
			current_dict = current_dict.setdefault(letter, {})
		current_dict['_end_'] = '_end_'

	def remove(self, word, current_dict={}):

		if current_dict == {}:
			if not self.search(word): return False
			current_dict = self.trie

		letter = word[0]
		if len(word) == 1:
			if '_end_' in current_dict[letter]:
				del(current_dict[letter]['_end_'])
			return
		if current_dict[letter] == {}:
			del(current_dict[letter])
			return
		else:
			self.remove(word[1:], current_dict[letter])
