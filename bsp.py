import random


def _bsp(x_mn, y_mn, x_mx, y_mx, vertical=False, r_v=0.25, r_h=0.25):
    if vertical:
        w = x_mx - x_mn
        x0 = random.randint(int(x_mn + w * r_v), int(x_mx - w * r_v))
        return (x_mn, y_mn, x0, y_mx), (x0, y_mn, x_mx, y_mx)
    else:
        h = y_mx - y_mn
        y0 = random.randint(int(y_mn + h * r_h), int(y_mx - h * r_h))
        return (x_mn, y_mn, x_mx, y0), (x_mn, y0, x_mx, y_mx)


class TreeNode:

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = None

        self.parent = None
        self.left = None
        self.right = None
        self.depth = 0

    @property
    def leaf(self):
        return self.left is None and self.right is None

    @property
    def root(self):
        return self.parent is None


class BSPTree:
    """ Problem specific BPS """

    def __init__(self, x: int, y: int, w: int, h: int):
        self.root = TreeNode(x, y, w, h)

    def grow(self, depth: int):
        del self.root.left
        del self.root.right
        self.root.left = None
        self.root.right = None
        leaves = [self.root]

        for depth in range(depth):
            new_leaves = []
            # breadth-first generation
            while len(leaves) > 0:
                leaf = leaves.pop()
                # handle disproportions of ratio between width and height
                ratio = leaf.w / leaf.h
                vertical = False
                if ratio > (random.randint(75, 150) / 100):
                    vertical = True
                ###########################################################
                a, b = _bsp(leaf.x, leaf.y, leaf.x + leaf.w, leaf.y + leaf.h, vertical, 0.3, 0.3)
                leaf.left = TreeNode(a[0], a[1], a[2] - a[0], a[3] - a[1])
                leaf.left.parent = leaf
                leaf.left.depth = depth
                leaf.right = TreeNode(b[0], b[1], b[2] - b[0], b[3] - b[1])
                leaf.right.parent = leaf
                leaf.right.depth = depth
                new_leaves.append(leaf.left)
                new_leaves.append(leaf.right)
            leaves = new_leaves

    def __iter__(self):
        leaves = [self.root]
        while len(leaves) > 0:
            leaf = leaves.pop()
            yield leaf
            if leaf.left:
                leaves.append(leaf.left)
            if leaf.right:
                leaves.append(leaf.right)
