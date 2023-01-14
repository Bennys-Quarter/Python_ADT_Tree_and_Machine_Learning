class ConsoleTree:
    def __init__(self, value):
        self.val = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __lt__(self, other):
        return self.val < other.val