import random
from abc import abstractmethod, ABC


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
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._type = None

        self._left = None
        self._right = None
        self._depth = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def leaf(self):
        return self._left is None and self._right is None

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value


class TreeVisitor(ABC):
    @abstractmethod
    def visit(self, node: TreeNode):
        pass


class BSPTree:
    """ Problem specific BPS """

    def __init__(self, x: int, y: int, w: int, h: int):
        self._root = TreeNode(x, y, w, h)

    def grow(self, depth: int):
        self._root.left = None
        self._root.right = None
        leaves = [self._root]

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
                leaf.left.depth = depth
                leaf.right = TreeNode(b[0], b[1], b[2] - b[0], b[3] - b[1])
                leaf.right.depth = depth
                new_leaves.append(leaf.left)
                new_leaves.append(leaf.right)
            leaves = new_leaves

    @property
    def root(self):
        return self._root

    def traverse(self, visitor: TreeVisitor):
        leaves = [self.root]
        while len(leaves) > 0:
            leaf = leaves.pop()
            visitor.visit(leaf)
            if leaf.left:
                leaves.append(leaf.left)
            if leaf.right:
                leaves.append(leaf.right)
