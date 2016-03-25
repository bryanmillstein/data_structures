class Trie:

    def __init__(self, words):
        # Iterate over words array and add each to the trie.

        root = {}
        for word in words:
            current_letter = root
            for letter in word[1:-2]:
                current_letter = current_letter.setdefault(letter, {})
            current_letter['word'] = word[1:-2]
        self.trie = root

def boggle_words(board, dictionary):
    # Iterates across board's letters and checks each combincation
    # of letters to match words in dictionary.
    # Returns array of dictionary words in boogle board.

    root = dictionary.trie
    words = []
    queue = []
    for row in range(len(board)):
        for col in range(len(board)):
            current_letter = board[row][col]
            try:
                current_node = root[current_letter]
                if current_node:
                    queue.append([current_node, row, col, None, None])
            except:
                pass

    while len(queue) > 0:
        current_node, row, col, prow, pcol = queue.pop(0)
        # print current_node
        moves = [
            [1,1],
            [1,-1],
            [-1, 1],
            [-1, -1],
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ]

        for move in moves:
            next_row, next_col = row + move[0], col + move[1]
            if (next_row != prow or next_col != pcol) and (0 <= next_row < len(board) and 0 <= next_col < len(board)):
                next_letter = board[next_row][next_col]
                try:
                    next_node = current_node[next_letter]
                    if next_node:
                        if 'word' in next_node and next_node['word'] not in words:
                            words.append(next_node['word'])
                        queue.append([next_node, next_row, next_col, row, col])
                except:
                    pass
    return words

dictionary = Trie(open('boggle_words.py'))
board = [
	'cats',
	'dogs',
	'cops',
	'toad'
]

# print boggle_words(board, dictionary)
# print len(boggle_words(board, dictionary))

# 1010 -> 10

def binary_to_integer(binary_num):
    num = 0
    for i in range(len(binary_num)-1, -1, -1):
        val = (int(binary_num[i]) * (2**(len(binary_num)-i-1)))
        num += val
    return num

def decimal_to_binary(decimal):

    binary_nums = []
    while decimal > 0:
        binary_nums.insert(0, str(decimal % 2))
        decimal = decimal / 2

    binary_str = "".join(binary_nums)
    return binary_str

def decimal_to_hex(decimal):
    hex_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']

    hex_nums = []
    while decimal > 0:
        print decimal
        hex_nums.insert(0, str(hex_values[decimal % 16]))
        decimal = decimal / 16

    hex_str = "".join(hex_nums)
    return "#" + hex_str


# print binary_to_integer('10000000000000000')
# print decimal_to_hex(2**16)
