class Node:
    def __init__(self, ch, freq, left=None, right=None):
        self.ch = ch    # char that is displayed
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

class BinaryNode:
    def __init__(self, ch, num, begin=0, end=0, parent=None, left=None, right=None):
        self.ch = ch    # char that is displayed
        self.begin = begin  # lower interval limit
        self.end = end    # upper interval limit
        self.num = num
        self.parent = parent
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.ch < other.ch