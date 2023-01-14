from tree.binary_tree import BinaryTree

if __name__ == "__main__":
    bin_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    #bin_list = [1,12,4,55,66,88,54,11,100,200,300,400,13,14,15,16,17,18,19,20,21,22,23,24,666,42,80,99,999,43,44]
    print(bin_list)
    b1 = BinaryTree(bin_list)
    b1.print_btree()
    b1.search_tree(0)
    b1.search_tree(16)
    b1.search_tree(30)
    b1.search_tree(31)
    b1.search_tree(-1)

