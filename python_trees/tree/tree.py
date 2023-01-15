from heapq import heapify, heappop, heappush
from PrettyPrint import PrettyPrintTree
from .node import Node
from .console_tree import  ConsoleTree
# Source: https://youtu.be/B3y0RsVCyrw


class HuffmanTree:
    def __init__(self, text: str):
        self.text = text
        self.tree = self.get_huffman_tree()

    def get_huffman_tree(self, input_text: str = ''):
        """
        Calculate the frequency of each character in the text
        Huffman encoding is then performed using a priority queue
        :input_text: optional; calculates the hoffman tree of arbitrary string
        :return: pq; priority queue of huffman encoded nodes forming the optimal encoding tree
        """
        if len(self.text) == 0:
            return
        if len(input_text) != 0:
            self.text = input_text

        freq = {ch: self.text.count(ch) for ch in set(self.text)}
        pq = [Node(k, v) for k, v in freq.items()]
        heapify(pq)
        while len(pq) > 1:
            left, right = heappop(pq), heappop(pq)
            new_freq = left.freq + right.freq
            heappush(pq, Node(None, new_freq, left, right))

        return pq

    def print_tree(self):
        #Todo: print a tree in a console application
        pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val)
        pt_tree = ConsoleTree("f=" + str(self.tree[0].freq))
        parent = self.tree[0]

        children = []
        stack = []
        stack_counter = []
        base_node = pt_tree

        go_right = False
        node_count = 0
        while True:

            child_left = parent.left
            child_right = parent.right
            char = ''

            if child_right:
                if child_right.ch:
                    char = child_right.ch
                children.append(base_node.add_child(ConsoleTree(char + " f=" + str(child_right.freq))))
                node_count += 1
                stack.append(child_right)
                stack_counter.append(node_count)

            char = ''
            if child_left:
                if child_left.ch:
                    char = child_left.ch
                children.append(base_node.add_child(ConsoleTree(char + " f=" + str(child_left.freq))))
                parent = child_left
                node_count += 1
                go_right = False
            elif not stack == []:
                parent = stack.pop()

                go_right = True
            else:
                break

            if go_right:
                base_node = children[stack_counter.pop()-1]
            else:
                if child_right.freq < child_left.freq:
                    base_node = children[node_count-2]
                else:
                    base_node = children[node_count-1]

        pt(pt_tree)
