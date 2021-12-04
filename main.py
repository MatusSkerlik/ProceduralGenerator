import math
import random
import string
import time
from abc import abstractmethod, ABC
from enum import Enum, auto
from functools import partial
from typing import Tuple, List, Dict

from bsp import TreeVisitor, TreeNode, BSPTree


def get_bounding_rect(pixels):
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    initialized = False

    for x, y in pixels:
        if not initialized:
            min_x = x
            min_y = y
            max_x = x
            max_y = y
            initialized = True
        else:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    x = min_x
    y = min_y
    w = max_x - min_x + 1
    h = max_y - min_y + 1

    return x, y, w, h


WIDTH = 1920
HEIGHT = 1080
HILLS_LEVEL1_COUNT = 1
HILLS_LEVEL2_COUNT = 3
HILLS_LEVEL3_COUNT = 2
LOWLAND_LEVEL0_COUNT = 3
LOWLAND_LEVEL1_COUNT = 2
LOWLAND_LEVEL2_COUNT = 1
LOWLAND_LEVEL3_COUNT = 1
DEBUG = False


class Material(Enum):
    """ Abstract class for indirect mapping of colors """
    BASE = auto()
    SECONDARY = auto()
    TERTIARY = auto()
    MATERIAL3 = auto()
    MATERIAL4 = auto()
    MATERIAL5 = auto()
    MATERIAL6 = auto()
    MATERIAL7 = auto()
    MATERIAL8 = auto()
    MATERIAL9 = auto()
    BACKGROUND = auto()
    CAVE_BACKGROUND = auto()


class Colors:
    """ Colors """

    BACKGROUND_SPACE = (8, 0, 60)
    BACKGROUND_SURFACE = (155, 209, 254)
    BACKGROUND_UNDERGROUND = (150, 107, 76)
    BACKGROUND_CAVERN = (127, 127, 127)
    BACKGROUND_UNDERWORLD = (0, 0, 0)

    GRASS = (60, 189, 90)
    STONE = (53, 53, 62)
    DIRT = (88, 63, 50)
    MUD = (94, 68, 71)

    COPPER = (148, 68, 28)
    IRON = (127, 127, 127)  # todo
    SILVER = (215, 222, 222)
    GOLD = (183, 162, 29)


Color = Tuple[int, int, int]
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)


class Rectangle:

    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def get_rect(self):
        return self.x, self.y, self.w, self.h

    def get_points(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def get_center(self):
        return int(self.x + (self.w / 2)), int(self.y + (self.h / 2))

    def enclosed_in(self, rect: 'Rectangle'):
        x, y, w, h = rect.get_rect()
        return self.x < x and self.x + self.w > x + w and self.y < y and self.y + self.h > y + h

    def is_inside(self, x: int, y: int):
        return self.x <= x <= (self.x + self.w) and self.y <= y <= (self.y + self.h)

    def __repr__(self):
        return "%d %d %d %d \n" % (self.x, self.y, self.w, self.h)


class Grid:

    def __init__(self, x: int, y: int, w: int, h: int, default_state: int = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self._array = [default_state for _ in range(w * h)]

    def __setitem__(self, key, value):
        x, y = key
        self._array[(y - self.y) * self.w + (x - self.x)] = value

    def __getitem__(self, item):
        x, y = item
        return self._array[(y - self.y) * self.w + (x - self.x)]


def flood_fill(x: int, y: int, state: int, grid: Grid):
    queue: List[Tuple[int, int]] = [(x, y)]
    visited: Dict[Tuple[int, int], bool] = dict()
    cells: List[Tuple[int, int]] = [(x, y)]

    while len(queue) > 0:
        coords = queue.pop()
        if coords not in visited:
            x0, y0 = coords
            cell_state = grid[x0, y0]
            visited[coords] = True
            if cell_state == state:
                cells.append((x0, y0))
                if x0 > grid.x:
                    queue.append((x0 - 1, y0))
                if x0 < grid.x + grid.w - 1:
                    queue.append((x0 + 1, y0))
                if y0 > grid.y:
                    queue.append((x0, y0 - 1))
                if y0 < grid.y + grid.h - 1:
                    queue.append((x0, y0 + 1))
    return cells


def nbs_neumann(x: int, y: int, grid: Grid):
    # top right bottom left
    nbs = [None, None, None, None]

    if y > grid.y:
        nbs[0] = grid[x, y - 1]
    if x < grid.x + grid.w - 1:
        nbs[1] = grid[x + 1, y]
    if y < grid.y + grid.h - 1:
        nbs[2] = grid[x, y + 1]
    if x > grid.x:
        nbs[3] = grid[x - 1, y]

    return nbs


def nbs_moore(x: int, y: int, grid: Grid):
    # top right bottom left top_right bottom_right bottom_left top_left
    nbs = nbs_neumann(x, y, grid) + [None, None, None, None]

    if x < grid.x + grid.w - 1 and y > grid.y:
        nbs[4] = grid[x + 1, y - 1]
    if x < grid.x + grid.w - 1 and y < grid.y + grid.h - 1:
        nbs[5] = grid[x + 1, y + 1]
    if x > grid.x and y < grid.y + grid.h - 1:
        nbs[6] = grid[x - 1, y + 1]
    if x > grid.x and y > grid.y:
        nbs[7] = grid[x - 1, y - 1]

    return nbs


def create_cave(rect: Rectangle, config_seq: Tuple[Tuple[int, int], ...], birth_chance: float):
    grid = Grid(rect.x, rect.y, rect.w, rect.h)

    for x in range(grid.x, grid.x + grid.w):
        for y in range(grid.y, grid.y + grid.h):
            grid[x, y] = 1 if random.random() > birth_chance else 0

    for step in range(len(config_seq)):
        death_limit, birth_limit = config_seq[step]
        updated_grid = Grid(rect.x, rect.y, rect.w, rect.h)
        for x in range(grid.x, grid.x + grid.w):
            for y in range(grid.y, grid.y + grid.h):
                alive = len([True for _ in nbs_moore(x, y, grid) if _ == 1])

                if grid[x, y] == 1:
                    if alive < death_limit:
                        updated_grid[x, y] = 0
                    else:
                        updated_grid[x, y] = 1
                else:
                    if alive > birth_limit:
                        updated_grid[x, y] = 1
                    else:
                        updated_grid[x, y] = 0
        grid = updated_grid

    return [(x, y) for x in range(grid.x, grid.x + grid.w) for y in range(grid.y, grid.y + grid.h) if grid[x, y] == 0]


def cerp(v0: float, v1: float, t: float):
    mu2 = (1 - math.cos(t * math.pi)) / 2
    return v0 * (1 - mu2) + v1 * mu2


def lerp(v0: float, v1: float, t: float):
    return v0 + t * (v1 - v0)


def create_surface(rect: Rectangle, l1: int, l2: int, l3: int, b: int, h1: int, h2: int, h3: int, fade_width: float):
    num_of_levels = l1 + l2 + l3 + b + h1 + h2 + h3

    level_order = [-1] * l1 + [-2] * l2 + [-3] * l3 + [0] * b + [1] * h1 + [2] * h2 + [3] * h3
    random.shuffle(level_order)

    mean_width = rect.w / num_of_levels
    min_level_width = int(mean_width * 0.5)
    max_level_width = int(mean_width * 1.5)
    level_width = [random.randint(min_level_width, max_level_width) for _ in range(num_of_levels)]
    level_fade = [random.randint(int(lw * fade_width * 0.75), int(lw * fade_width * 1)) for lw in level_width]

    # corrections due to float point rounding
    while sum(level_width) > rect.w:
        i = random.randint(0, num_of_levels - 1)
        level_width[i] -= 1
    while sum(level_width) < rect.w:
        i = random.randint(0, num_of_levels - 1)
        level_width[i] += 1

    noise = []
    cw = 0
    while num_of_levels > 1:
        num_of_levels -= 1
        l = level_order.pop()
        ln = level_order[num_of_levels - 1]
        w = level_width.pop()
        fw = level_fade.pop()
        noise += [(l + 3) * 1 / 6] * (w - fw)
        for i in range(fw):
            noise += [cerp((l + 3) * 1 / 6, (ln + 3) * 1 / 6, i / fw)]
        #        assert len(noise) - cw == w
        cw += w
    noise += [(level_order.pop() + 3) * 1 / 6] * level_width.pop()

    # map to pixels
    pixels = []
    x = 0
    for y in noise:
        for y0 in range(int(rect.y + rect.h - rect.h * y), rect.y + rect.h):
            pixels.append((x, y0))
        x += 1
    return pixels


Space = Rectangle(0, 0, WIDTH, 1 * HEIGHT / 10)
Surface = Rectangle(0, Space.y + Space.h, WIDTH, 2 * HEIGHT / 10)
Underground = Rectangle(0, Surface.y + Surface.h, WIDTH, 2 * HEIGHT / 10)
Cavern = Rectangle(0, Underground.y + Underground.h, WIDTH, 4 * HEIGHT / 10)
Underworld = Rectangle(0, Cavern.y + Cavern.h, WIDTH, 1 * HEIGHT / 10)


class PixelMaterialColorMap:
    """ Class which maps colors of materials into individual pixel """
    _map: Dict[Tuple[int, int], Dict[Material, Color]] = {}
    _default_mapping = {}

    @classmethod
    def add_rect(cls, rect: Rectangle, mapping: Dict[Material, Color]) -> None:
        """ Add rectangle with specific mapping """
        for x in range(rect.x, rect.x + rect.w):
            for y in range(rect.y, rect.y + rect.h):
                cls.add_pixel(x, y, mapping)

    @classmethod
    def add_pixel(cls, x, y, mapping: Dict[Material, Color]) -> None:
        """ Add pixel with specific mapping """
        if (x, y) not in cls._map:
            cls._map[(x, y)] = dict(cls._default_mapping)
        cls._map[(x, y)].update(mapping)

    @classmethod
    def get_color(cls, x, y, material: Material) -> Color:
        return cls._map[(x, y)][material]


class RegionType(Enum):
    CAVE = auto()
    ORE = auto()


class PygameDrawer:
    """ Drawing interface """

    def __init__(self):
        self.canvas = None
        self.running = False
        self._font = None

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
                pygame.image.save(self.canvas, "img/" + name + ".png")
                pygame.quit()
                pygame.font.quit()
                signal.raise_signal(signal.SIGINT)
                break

    def init(self):
        global WIDTH, HEIGHT
        pygame.init()
        pygame.font.init()
        self.canvas = pygame.display.set_mode([WIDTH, HEIGHT])
        self._font = pygame.font.Font("font.ttf", 12)

    def free(self):
        while True:
            time.sleep(1 / 60)
            self._handle_events()

    def draw_rect(self, rect: Rectangle, color: Color = None):
        x, y, x1, y1 = rect.get_points()
        if color is not None:
            pygame.draw.rect(self.canvas, color, (x, y, x1, y1))
        else:
            for x0 in range(x, x1):
                for y0 in range(y, y1):
                    color = PixelMaterialColorMap.get_color(x0, y0, Material.BACKGROUND)
                    pygame.draw.rect(self.canvas, color, (x0, y0, 1, 1))
        pygame.display.update((x, y, x1, y1))
        self._handle_events()

    def draw_outline(self, rect: Rectangle, color: Color):
        x, y, x1, y1 = rect.get_rect()
        pygame.draw.rect(self.canvas, color, (x, y, x1, y1), 1)
        pygame.display.update((x, y, x1, y1))
        self._handle_events()

    def draw_pixels(self, pixels, color: Color = None, material: Material = None):
        for x, y in pixels:
            if color is None:
                if material is None:
                    c = PixelMaterialColorMap.get_color(x, y, Material.BACKGROUND)
                else:
                    c = PixelMaterialColorMap.get_color(x, y, material)
                pygame.draw.rect(self.canvas, c, (x, y, 1, 1))
            else:
                pygame.draw.rect(self.canvas, color, (x, y, 1, 1))
        pygame.display.update(get_bounding_rect(pixels))
        self._handle_events()

    def draw_polygon(self, points: Tuple[Tuple[int, int], ...], color: Tuple[int, int, int]):
        pygame.draw.polygon(self.canvas, color, points)

        sx = None
        sy = None
        bx = None
        by = None
        for point in points:
            x, y = point
            if sx is None:
                sx = x
            elif sx > x:
                sx = x

            if sy is None:
                sy = y
            elif sy > y:
                sy = y

            if bx is None:
                bx = x
            elif bx < x:
                bx = x

            if by is None:
                by = y
            elif by < y:
                by = y

        pygame.display.update((sx, sy, bx - sx, by - sy))
        self._handle_events()

    def draw_progress(self, text: string):
        global WIDTH, HEIGHT
        pygame.draw.rect(self.canvas, (0, 128, 128), (0, HEIGHT, WIDTH, HEIGHT + 64))
        font = self._font.render(text, False, (255, 255, 255))
        self.canvas.blit(font, (4, HEIGHT + 4))
        pygame.display.update((0, HEIGHT, WIDTH, HEIGHT))
        self._handle_events()


class CreateCaveTreeVisitor(TreeVisitor):

    def __init__(self, drawer: PygameDrawer):
        self.drawer = drawer

    def visit(self, node: TreeNode):
        if node.leaf and node.type is None:
            func = partial(create_cave, Rectangle(node.x, node.y, node.w, node.h), ((3, 4),) * 5 + ((4, 4),) * 2, .525)

            def success(pixels):
                self.drawer.draw_pixels(pixels, material=Material.CAVE_BACKGROUND)

            def error(err):
                print(err)

            parallel.to_process(func, success, error)
            node.type = RegionType.CAVE


class DrawTreeVisitor(TreeVisitor):
    """ Traverse tree and draw leaf nodes """

    def __init__(self, drawer: PygameDrawer):
        self._drawer = drawer

    def visit(self, node: TreeNode):
        if node.leaf:
            self._drawer.draw_outline(Rectangle(node.x, node.y, node.w, node.h), RED)


class Scene(ABC):
    """ Generation scene """
    drawer: PygameDrawer

    @abstractmethod
    def run(self):
        pass


class MainScene(Scene):

    def __init__(self):
        self.drawer = PygameDrawer()

    def run(self):
        global WIDTH, HEIGHT, DEBUG

        tree = BSPTree(0, Underground.y, WIDTH, Underground.h + Cavern.h)
        tree.grow(8)

        PixelMaterialColorMap.add_rect(Space, {
            Material.BACKGROUND: Colors.BACKGROUND_SPACE,
        })

        PixelMaterialColorMap.add_rect(Surface, {
            Material.BACKGROUND: Colors.BACKGROUND_SURFACE,
            Material.CAVE_BACKGROUND: Colors.DIRT
        })

        PixelMaterialColorMap.add_rect(Underground, {
            Material.BACKGROUND: Colors.BACKGROUND_UNDERGROUND,
            Material.CAVE_BACKGROUND: Colors.DIRT
        })

        PixelMaterialColorMap.add_rect(Cavern, {
            Material.BACKGROUND: Colors.BACKGROUND_CAVERN,
            Material.CAVE_BACKGROUND: Colors.STONE
        })

        PixelMaterialColorMap.add_rect(Underworld, {
            Material.BACKGROUND: Colors.BACKGROUND_UNDERWORLD,
            Material.CAVE_BACKGROUND: Colors.STONE
        })

        self.drawer.init()
        self.drawer.draw_progress("Init ...")
        self.drawer.draw_rect(Space, Colors.BACKGROUND_SURFACE)
        self.drawer.draw_rect(Surface, Colors.BACKGROUND_SURFACE)
        self.drawer.draw_rect(Underground, Colors.BACKGROUND_UNDERGROUND)
        self.drawer.draw_rect(Cavern, Colors.BACKGROUND_CAVERN)
        self.drawer.draw_rect(Underworld, Colors.BACKGROUND_UNDERWORLD)

        if DEBUG:
            self.drawer.draw_outline(Space, MAGENTA)
            self.drawer.draw_outline(Surface, MAGENTA)
            self.drawer.draw_outline(Underground, MAGENTA)
            self.drawer.draw_outline(Cavern, MAGENTA)
            self.drawer.draw_outline(Underworld, MAGENTA)
            tree.traverse(DrawTreeVisitor(self.drawer))

        self.drawer.draw_pixels(create_surface(Surface, 1, 2, 3, 1, 3, 2, 1, 0.8), Colors.BACKGROUND_UNDERGROUND)

        cave_tree_visitor = CreateCaveTreeVisitor(self.drawer)
        tree.traverse(cave_tree_visitor)

        self.drawer.draw_progress("Done ...")
        self.drawer.free()


if __name__ == '__main__':
    import pygame
    import parallel
    import signal

    scene = MainScene()
    scene.run()
