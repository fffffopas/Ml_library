class Node():
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.tresh = None
        self.index_feature = None
        self.val = val

    def _is_leaf(self):
        return self.val is not None