from heapq import heapify, heappop, heappush
from PrettyPrint import PrettyPrintTree
from .node import Node, BinaryNode
from .console_tree import  ConsoleTree

#TODO: Seperate this file from huffman tree

class BinaryTree:
    def __init__(self, num_list:list):
        self.num_list = num_list
        self.root: BinaryNode = self.get_binary_tree()
        self.node_number = len(num_list)

    def get_binary_tree(self, input_list:list = []):
        """
        Calculate the binary tree all numbers in a list
        :input_text: optional; calculates the binary tree of arbitrary numeric list
        :return: pq; priority queue of binary encoded nodes forming the binary search tree
        """
        for item in self.num_list:
            if type(item) != int:
                print("List contains none numerical entries!")
                return
        if len(self.num_list) == 0:
            return
        if input_list != []:
            self.num_list = input_list

        q = []
        self.num_list = sorted(self.num_list)

        new_number = int(len(self.num_list)/2)
        start = self.num_list[new_number]
        root = BinaryNode(str(start), new_number, begin=0, end=len(self.num_list), parent=None, left=None, right=None)
        left_ind = int((root.begin + root.num)/2)
        right_ind = int((root.num + root.end)/2)
        if right_ind != root.end:
            right_num = self.num_list[right_ind]
            root.right = BinaryNode(str(right_num), right_ind, begin=new_number, end=root.end, parent=root, left=None, right=None)
            heappush(q, root.right)
        if left_ind != root.begin:
            left_num = self.num_list[left_ind]
            root.left = BinaryNode(str(left_num), left_ind, begin=root.begin, end=new_number, parent=root, left=None, right=None)
            heappush(q, root.left)
        print(start)
        while len(q) > 0:
            current_node: BinaryNode = heappop(q)
            left_ind = int((current_node.begin + current_node.num) / 2)
            right_ind = int((current_node.num + current_node.end) / 2)
            if right_ind != current_node.num:
                right_num = self.num_list[right_ind]
                current_node.right = BinaryNode(str(right_num), right_ind, begin=current_node.num, end=current_node.end,
                                                parent=current_node, left=None, right=None)
                heappush(q, current_node.right)
            if (left_ind != current_node.num) and (left_ind > current_node.begin):
                left_num = self.num_list[left_ind]
                current_node.left = BinaryNode(str(left_num), left_ind, begin=current_node.begin, end=current_node.num,
                                               parent=current_node, left=None, right=None)
                heappush(q, current_node.left)

            if (current_node.num == 1):
                num = self.num_list[0]
                current_node.left = BinaryNode(str(num), 0, begin=current_node.begin, end=0,
                                               parent=current_node, left=None, right=None)

        return root

    def search_tree(self, key_num: int):
        node = self.root
        key = BinaryNode(str(key_num), key_num)
        while True:
            if str(key_num) == node.ch:
                print(f"Value {key_num} found")
                return
            else:
                if key < node:
                    if node.left:
                        node = node.left
                    else:
                        print(f"Value {key_num} not found")
                        return
                else:
                    if node.right:
                        node = node.right
                    else:
                        print(f"Value {key_num} not found")
                        return

    def print_btree(self):
        if not self.root: return
        pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val)
        pt_tree = ConsoleTree(self.root.ch)
        parent = self.root

        children = []
        stack = []
        stack_counter = []
        base_node = pt_tree

        node_count = 0
        for i in range(self.node_number - 1):

            child_left = parent.left
            child_right = parent.right

            char = ''
            if child_left:
                if child_left.ch:
                    char = child_left.ch
                children.append(base_node.add_child(ConsoleTree(char)))
                heappush(stack, child_left)
                node_count += 1
                child_l = children[-1:][0]
                stack_counter.append(child_l)

            char = ''
            if child_right:
                if child_right.ch:
                    char = child_right.ch
                children.append(base_node.add_child(ConsoleTree(char)))
                heappush(stack, child_right)
                node_count += 0
                child_r = children[-1:][0]
                stack_counter.append(child_r)

            parent: BinaryNode = heappop(stack)
            for item in stack_counter:
                if item.val == parent.ch:
                    base_node: ConsoleTree = item
            #print("Base_Node: ", base_node.val, "\tNode_Num: ", parent.num)

        pt(pt_tree)


    def add(self, new_number:int):
        pass

    def get_huffman_tree(self, input_text:str = ''):
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
                base_node = children[stack_counter.pop() - 1]
            else:
                if child_right.freq < child_left.freq:
                    base_node = children[node_count - 2]
                else:
                    base_node = children[node_count - 1]

        pt(pt_tree)
