import itertools
import math
import random
import string
import threading
from enum import Enum, auto
from functools import partial
from typing import Tuple, List, Dict, Union

from noise import pnoise2

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

# dynamic globals
_dynamic_lock = threading.RLock()


class DynamicInt(int):
    def __add__(self, other):
        with _dynamic_lock:
            return DynamicInt(super().__add__(other))

    def __sub__(self, other):
        with _dynamic_lock:
            return DynamicInt(super().__sub__(other))

    def __int__(self):
        with _dynamic_lock:
            return self.real

    def __repr__(self):
        with _dynamic_lock:
            return super().__repr__()


CAVE = DynamicInt(0)
WATER = DynamicInt(0)
LAVA = DynamicInt(0)
GRASS = DynamicInt(0)
SAND = DynamicInt(0)
DIRT = DynamicInt(0)
MUD = DynamicInt(0)
STONE = DynamicInt(0)
COPPER = DynamicInt(0)
GOLD = DynamicInt(0)


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
    WATER = auto()
    LAVA = auto()
    SAND = auto()


class Colors:
    """ Colors """

    BACKGROUND_SPACE = (8, 0, 60)
    BACKGROUND_SURFACE = (155, 209, 254)
    BACKGROUND_UNDERGROUND = (150, 107, 76)
    BACKGROUND_CAVERN = (127, 127, 127)
    BACKGROUND_UNDERWORLD = (0, 0, 0)

    STONE = (53, 53, 62)
    DIRT = (88, 63, 50)
    MUD = (94, 68, 71)

    COPPER = (148, 68, 28)
    IRON = (127, 127, 127)  # todo
    SILVER = (215, 222, 222)
    GOLD = (183, 162, 29)

    GRASS = (33, 214, 94)
    JUNGLE_GRASS = (136, 204, 33)
    SAND = (255, 218, 56)
    WATER = (14, 59, 192)
    LAVA = (251, 31, 8)


Color = Tuple[int, int, int]
PixelArray = List[Tuple[int, int]]

RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


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

    def to_pixel_array(self):
        pixels = []
        for x in range(self.x, self.x + self.w):
            for y in range(self.y, self.y + self.h):
                pixels.append((x, y))
        return pixels

    def __add__(self, other):
        return Rectangle(self.x, self.y, self.w, self.h + other.h)

    def __iter__(self):
        return itertools.product(range(self.x, self.x + self.w), range(self.y, self.y + self.h))

    def __repr__(self):
        return "%d %d %d %d \n" % (self.x, self.y, self.w, self.h)


class Grid:

    def __init__(self, x: int, y: int, w: int, h: int, default_state: int = 0, locked_state: int = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self._array = [default_state for _ in range(w * h)]
        self._locked = {}
        self._locked_state = locked_state

    def __setitem__(self, key, value):
        x, y = key
        if (x, y) in self._locked:
            pass
        else:
            self._array[(y - self.y) * self.w + (x - self.x)] = value

    def __getitem__(self, item):
        x, y = item
        if (x, y) in self._locked:
            return self._locked_state
        else:
            return self._array[(y - self.y) * self.w + (x - self.x)]

    def __iter__(self):
        return itertools.product(range(self.x, self.x + self.w), range(self.y, self.y + self.h))

    def lock(self, state: int):
        for x, y in self:
            if self[x, y] == state:
                self._locked[(x, y)] = True

    def unlock(self, state):
        for x, y in self:
            if self[x, y] == state:
                del self._locked[(x, y)]

    def lock_all(self):
        self._locked = {(x, y): True for x, y in self}

    def unlock_all(self):
        self._locked = {}

    def is_locked(self, x: int, y: int):
        return (x, y) in self._locked

    def extract(self, state: int):
        coords = []
        for x, y in self:
            if self[x, y] == state:
                coords.append((x, y))
        return coords

    def locked(self):
        return [(x, y) for x, y in self if self.is_locked(x, y)]

    def unlocked(self):
        return [(x, y) for x, y in self if not self.is_locked(x, y)]

    @staticmethod
    def from_rect(rect: Rectangle, default_state: int = 0):
        return Grid(rect.x, rect.y, rect.w, rect.h, default_state)

    @staticmethod
    def from_pixels(rect: Rectangle, pixels, state: int = 1, lock=False, default_state: int = 0):
        grid = Grid.from_rect(rect, default_state)
        for x, y in pixels:
            grid[x, y] = state
        if lock:
            grid.lock(state)
        return grid


def get_bounding_rect(pixels) -> Rectangle:
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

    return Rectangle(x, y, w, h)


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

    return cells, visited


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


def make_grid(rect: Rectangle, surface, mapping: Dict[Material, int], default_state: int = 0) -> Grid:
    grid = Grid.from_rect(rect)
    pixel_buffer = pygame.surfarray.pixels3d(surface)

    for x, y in grid:
        material_to_color = PixelMaterialColorMap.get_mapping(x, y)
        color_to_material = dict((v, k) for k, v in material_to_color.items())
        color = tuple(pixel_buffer[x, y])
        if color in color_to_material:
            material = color_to_material[color]
            if material in mapping:
                grid[x, y] = mapping[material]
            else:
                grid[x, y] = default_state
        else:
            grid[x, y] = default_state
    del pixel_buffer
    return grid


def create_grass(rect: Rectangle, grid: Grid, air_state: int, wall_state: int) -> PixelArray:
    grass_pixels = []
    for x, y in rect:
        nbs = nbs_neumann(x, y, grid)
        top, right, bottom, left = nbs

        if len([1 for c in (right, bottom, left) if c == air_state]) and len([1 for c in nbs if c == wall_state]):
            grass_pixels.append((x, y))

    return grass_pixels


def create_cave(rect: Union[Rectangle, Grid], config_seq: Tuple[Tuple[int, int], ...], birth_chance: float,
                min_size: int = 75, max_size: int = 10000) -> PixelArray:
    if isinstance(rect, Rectangle):
        grid = Grid.from_rect(rect)
    else:
        grid = rect

    # optimization to not traverse coords that are locked
    if isinstance(rect, Grid):
        unlocked = grid.unlocked()
    else:
        unlocked = grid

    for x, y in unlocked:
        grid[x, y] = 1 if random.random() > birth_chance else 0

    for step in range(len(config_seq)):
        death_limit, birth_limit = config_seq[step]
        updated_grid = Grid.from_rect(rect)
        for x, y in unlocked:
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

    if min_size > 0:
        visited = {}
        for x, y in unlocked:
            if (x, y) not in visited:
                cells, newly_visited = flood_fill(x, y, 1, grid)
                visited.update(newly_visited)
                if max_size > len(cells) < min_size:
                    for x0, y0 in cells:
                        grid[x0, y0] = 0

    return grid.extract(1)


def create_water(rect: Rectangle, mask: PixelArray, p: float) -> PixelArray:
    pixels = dict.fromkeys(mask, True)
    water_pixels = []
    bounding_rect = get_bounding_rect(pixels.keys())
    for y in range(rect.y + rect.h - int(p * bounding_rect.h), rect.y + rect.h):
        for x in range(rect.x, rect.x + rect.w):
            if (x, y) in pixels:
                water_pixels.append((x, y))
    return water_pixels


def create_ocean(rect: Rectangle, left=True, descent: int = 10) -> (PixelArray, PixelArray):
    """
    Will create ocean into rectangle with log base as function of depth (bigger -> deeper)
    """
    assert descent > 1

    sand_pixels = []
    water_pixels = []

    for y in range(rect.h):
        y0 = (y / (rect.h + 1)) + (1 / descent)
        if y0 < 1:  # correction
            for x0 in range(int(abs(math.log(y0, descent)) * rect.w)):
                if left:
                    sand_pixels.append((rect.x - 1 + rect.w - x0, rect.y + rect.h - int(y0 * rect.h)))
                else:
                    sand_pixels.append((rect.x + x0, rect.y + rect.h - int(y0 * rect.h)))

        # bottom correction
        for x in range(rect.x, rect.x + rect.w):
            for y in range(rect.y + rect.h - int((1 / descent) * rect.h), rect.y + rect.h):
                sand_pixels.append((x, y))

    for y in range(rect.h):
        y0 = (y / (rect.h + 1)) + (1 / descent)
        if y0 < 1:  # correction
            for x0 in range(int(abs(math.log(y0, descent)) * rect.w), rect.w):
                if left:
                    water_pixels.append((rect.x - 1 + rect.w - (rect.x + x0), rect.y + rect.h - int(y0 * rect.h)))
                else:
                    water_pixels.append((rect.x + x0, rect.y + rect.h - int(y0 * rect.h)))

    return sand_pixels, water_pixels


def cerp(v0: float, v1: float, t: float):
    mu2 = (1 - math.cos(t * math.pi)) / 2
    return v0 * (1 - mu2) + v1 * mu2


def lerp(v0: float, v1: float, t: float):
    return v0 + t * (v1 - v0)


def create_surface(rect: Rectangle, l1: int, l2: int, l3: int, b: int, h1: int, h2: int, h3: int, fade_width: float,
                   octaves: int = 0, persistence: float = 0.5) -> (PixelArray, int, int):
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
        cw += w
    noise += [(level_order.pop() + 3) * 1 / 6] * level_width.pop()

    o = octaves
    while o > 0:
        o -= 1
        octave = octaves - o
        for n in range(len(noise)):
            noise[n] += noise[(n * octave ** 2) % len(noise)] / (octave ** 1 / persistence)

    # normalise
    _max = max(noise)
    for n in range(len(noise)):
        noise[n] /= _max

    pixels = []
    x = rect.x
    for y in noise:
        for y0 in range(int(rect.y + rect.h - rect.h * y), rect.y + rect.h):
            pixels.append((x, y0))
        x += 1

    start_y = int(rect.y + rect.h - rect.h * noise[0])
    end_y = int(rect.y + rect.h - rect.h * noise[len(noise) - 1])

    return pixels, start_y, end_y


def perlin_fusion(rect: Rectangle, mask: PixelArray, h_start_prob: float, h_end_prob: float, fq: int,
                  octaves: int) -> PixelArray:
    assert h_start_prob >= h_end_prob

    pixels = []
    values = {}
    masked_pixels = dict.fromkeys(mask, True)
    for x, y in rect:
        if (x, y) not in masked_pixels:
            for octave in range(1, octaves + 1):
                if octave > 1:
                    values[(x, y)] += pnoise2((x - rect.x) / rect.w * fq * octave, (y - rect.y) / rect.w * fq * octave)
                else:
                    values[(x, y)] = pnoise2((x - rect.x) / rect.w * fq, (y - rect.y) / rect.w * fq)

    minimum = min(values.values())
    maximum = max(values.values())
    interval = maximum - minimum
    prob_decrease_per_y = (h_start_prob - h_end_prob) / rect.h
    for x, y in rect:
        y_prob = h_start_prob - ((y - rect.y) * prob_decrease_per_y)
        if minimum <= values[(x, y)] <= minimum + y_prob * interval:
            pixels.append((x, y))
    return pixels


# TODO
def ore_feasibility_check(rect: Rectangle, mask: PixelArray, count: int):
    return True


def create_ore(rect: Rectangle, mask: PixelArray, min_size: int, max_size: int, count: int, iterations: int = 3):
    grid = Grid.from_pixels(rect, mask, 1)
    grid.lock(1)

    pixels = []
    while len(pixels) < count and ore_feasibility_check(rect, mask + pixels, count) and iterations > 0:
        pixels += create_cave(grid, ((5, 1),) * 3 + ((5, 8),) * 3, 0.75, min_size, max_size)
        for x, y in pixels:
            grid[x, y] = 1
            grid.lock(1)
        iterations -= 1

    if iterations == 0 or not ore_feasibility_check(rect, mask + pixels, count):
        raise NameError("Ore could not be created.")
    else:
        return pixels


Space = Rectangle(0, 0, WIDTH, 1 * HEIGHT / 10)
Surface = Rectangle(0, Space.y + Space.h, WIDTH, 2 * HEIGHT / 10)
Underground = Rectangle(0, Surface.y + Surface.h, WIDTH, 2 * HEIGHT / 10)
Cavern = Rectangle(0, Underground.y + Underground.h, WIDTH, 4 * HEIGHT / 10)
Underworld = Rectangle(0, Cavern.y + Cavern.h, WIDTH, 1 * HEIGHT / 10)


class PixelMaterialColorMap:
    """ Class which maps colors of materials into individual pixel """
    _map: Dict[Tuple[int, int], Dict[Material, Color]] = {}
    _default_mapping = {
        Material.WATER: Colors.WATER,
        Material.LAVA: Colors.LAVA,
        Material.SAND: Colors.SAND
    }

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

    @classmethod
    def get_mapping(cls, x, y) -> Dict[Material, Color]:
        return cls._map[(x, y)]


if __name__ == '__main__':
    import parallel
    import pygame
    import time
    from bsp import TreeVisitor, TreeNode, BSPTree


    class Draw:
        """ Drawing interface """
        pygame.init()
        pygame.font.init()
        surface = pygame.display.set_mode([WIDTH, HEIGHT])
        font = pygame.font.SysFont(None, 28)

        def __init__(self):
            raise NotImplemented

        @staticmethod
        def count():
            global WATER, LAVA, CAVE, DIRT, STONE, MUD, COPPER, GRASS, GOLD
            size = 28

            if not Draw.surface.get_locked():
                pygame.draw.rect(Draw.surface, BLACK, (16, 28, 200, 9 * size))
                Draw.surface.blit(Draw.font.render("WATER: %d" % WATER, True, RED), (16, size))
                Draw.surface.blit(Draw.font.render("LAVA: %d" % LAVA, True, RED), (16, 2 * size))
                Draw.surface.blit(Draw.font.render("CAVE: %d" % CAVE, True, RED), (16, 3 * size))
                Draw.surface.blit(Draw.font.render("DIRT: %d" % DIRT, True, RED), (16, 4 * size))
                Draw.surface.blit(Draw.font.render("STONE: %d" % STONE, True, RED), (16, 5 * size))
                Draw.surface.blit(Draw.font.render("MUD: %d" % MUD, True, RED), (16, 6 * size))
                Draw.surface.blit(Draw.font.render("COPPER: %d" % COPPER, True, RED), (16, 7 * size))
                Draw.surface.blit(Draw.font.render("GOLD: %d" % GOLD, True, RED), (16, 8 * size))
                Draw.surface.blit(Draw.font.render("GRASS: %d" % GRASS, True, RED), (16, 9 * size))
                pygame.display.update((16, 28, 200, 9 * size))

        @staticmethod
        def loop():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
                    pygame.image.save(Draw.surface, "img/" + name + ".png")
                    pygame.quit()
                    pygame.font.quit()
                    return False
            return True

        @staticmethod
        def rect(rect: Rectangle, color: Color = None):
            global SURFACE

            x, y, x1, y1 = rect.get_rect()
            if color is not None:
                pygame.draw.rect(Draw.surface, color, (x, y, x1, y1))
            else:
                for x0 in range(x, x1):
                    for y0 in range(y, y1):
                        color = PixelMaterialColorMap.get_color(x0, y0, Material.BACKGROUND)
                        pygame.draw.rect(Draw.surface, color, (x0, y0, 1, 1))
            pygame.display.update((x, y, x1, y1))

        @staticmethod
        def outline(rect: Rectangle, color: Color):
            global SURFACE

            x, y, x1, y1 = rect.get_rect()
            pygame.draw.rect(Draw.surface, color, (x, y, x1, y1), 1)
            pygame.display.update((x, y, x1, y1))

        @staticmethod
        def pixels(pixels, color: Color = None, material: Material = None):
            global SURFACE

            for x, y in pixels:
                if color is None:
                    if material is None:
                        c = PixelMaterialColorMap.get_color(x, y, Material.BACKGROUND)
                    else:
                        c = PixelMaterialColorMap.get_color(x, y, material)
                    pygame.draw.rect(Draw.surface, c, (x, y, 1, 1))
                else:
                    pygame.draw.rect(Draw.surface, color, (x, y, 1, 1))
            pygame.display.update(get_bounding_rect(pixels).get_rect())

        @staticmethod
        def polygon(points: Tuple[Tuple[int, int], ...], color: Tuple[int, int, int]):
            pygame.draw.polygon(Draw.surface, color, points)

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


    class NodeType(Enum):
        """ Types of BSP tree nodes """
        CAVE = auto()
        ORE = auto()


    class CreateCaveTreeVisitor(TreeVisitor):
        """ Visit BSP Tree and generate caves """

        def visit(self, node: TreeNode):
            if node.leaf and node.type is None:
                rect = Rectangle(node.x, node.y, node.w, node.h)

                if random.random() > 0.2:
                    func = partial(create_cave, rect, ((5, 1),) * 8 + ((5, 8),) * 4,
                                   .75)
                else:
                    func = partial(create_cave, rect, ((3, 4),) * 2 + ((4, 4),) * 2,
                                   .575)

                def success(pixels):
                    global CAVE
                    CAVE += len(pixels)
                    Draw.pixels(pixels, material=Material.CAVE_BACKGROUND)

                def error(err):
                    raise err

                token = parallel.get_token()
                parallel.to_process(func, success, error, token)

                # generate COPPER ORE
                def func1(result):
                    return create_ore(rect, result, 10, 50, 1)

                def success1(pixels):
                    global COPPER
                    COPPER += len(pixels)
                    Draw.pixels(pixels, Colors.COPPER)

                def error1(err):
                    print(err)

                token1 = parallel.get_token()
                parallel.to_thread_after(func1, success1, error1, wait_token=token, token=token1)

                # CREATE GOLD
                def func2(result):
                    return create_ore(rect, result, 10, 50, 1)

                def success2(pixels):
                    global GOLD
                    GOLD += len(pixels)
                    Draw.pixels(pixels, Colors.GOLD)

                def error2(err):
                    print(err)

                token2 = parallel.get_token()
                parallel.to_thread_after(func2, success2, error2, wait_token=token, token=token2)

                if random.random() > 0.8:  # fill caves with water
                    def func(result):
                        return create_water(rect, result, random.randint(10, 50) / 100)

                    def success(pixels):
                        global WATER, LAVA

                        if random.random() > 0.8:
                            LAVA += len(pixels)
                            Draw.pixels(pixels, material=Material.LAVA)
                        else:
                            WATER += len(pixels)
                            Draw.pixels(pixels, material=Material.WATER)

                    def error(err):
                        raise err

                    parallel.to_thread_after(func, success, error, wait_token=token)

                node.type = NodeType.CAVE


    class DrawTreeVisitor(TreeVisitor):
        """ Traverse tree and draw leaf nodes """

        def visit(self, node: TreeNode):
            if node.leaf:
                Draw.outline(Rectangle(node.x, node.y, node.w, node.h), RED)


    class MainScene:
        """ Namespace for generation of everything """

        @staticmethod
        def run():
            global WIDTH, HEIGHT, DEBUG
            global DIRT, MUD, SAND, WATER, LAVA, GRASS, COPPER, STONE

            tree = BSPTree(0, Underground.y, WIDTH, Underground.h + Cavern.h)
            tree.grow(9)

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

            Draw.rect(Space, Colors.BACKGROUND_SURFACE)
            Draw.rect(Surface, Colors.BACKGROUND_SURFACE)
            Draw.rect(Underground, Colors.BACKGROUND_CAVERN)
            Draw.rect(Cavern, Colors.BACKGROUND_CAVERN)
            Draw.rect(Underworld, Colors.BACKGROUND_UNDERWORLD)

            STONE += Underground.w * Underground.h
            STONE += Cavern.w * Cavern.h

            if DEBUG:
                Draw.outline(Space, MAGENTA)
                Draw.outline(Surface, MAGENTA)
                Draw.outline(Underground, MAGENTA)
                Draw.outline(Cavern, MAGENTA)
                Draw.outline(Underworld, MAGENTA)
                tree.traverse(DrawTreeVisitor())

            # CREATE SURFACE
            surface_w_offset_left = 0.075 * WIDTH
            surface_w_offset_right = 0.075 * WIDTH
            surface_w = WIDTH - surface_w_offset_right - surface_w_offset_right
            SURFACE_H = Surface.h
            surface_h_offset_bottom = 0.25 * SURFACE_H
            surface_h_offset_top = 0.35 * SURFACE_H
            surface_h = SURFACE_H - surface_h_offset_bottom - surface_h_offset_top
            surface_x = surface_w_offset_left
            surface_y = Space.h + surface_h_offset_top
            Surf = Rectangle(surface_x, surface_y, surface_w, surface_h)
            surface, ocean_left_y, ocean_right_y = create_surface(
                Surf,
                LOWLAND_LEVEL1_COUNT,
                LOWLAND_LEVEL2_COUNT,
                LOWLAND_LEVEL3_COUNT,
                LOWLAND_LEVEL0_COUNT,
                HILLS_LEVEL1_COUNT,
                HILLS_LEVEL2_COUNT,
                HILLS_LEVEL3_COUNT,
                0.8,
                octaves=4,
                persistence=0.12
            )
            Draw.pixels(surface, Colors.BACKGROUND_UNDERGROUND)
            DIRT += len(surface)
            Coral = Rectangle(
                0,
                Space.h + surface_h_offset_top + surface_h - 1,
                WIDTH,
                surface_h_offset_bottom + 2
            )
            Draw.rect(Coral, Colors.BACKGROUND_UNDERGROUND)
            DIRT += Coral.w * Coral.h

            # PERLIN FUSION
            dirt = perlin_fusion(Underground + Cavern, [], 0.7, 0.2, 30, 6)
            Draw.pixels(dirt, Colors.BACKGROUND_UNDERGROUND)
            DIRT += len(dirt)
            STONE -= len(dirt)

            mud = perlin_fusion(Underground + Cavern, [], 0.2, 0.2, 40, 3)
            Draw.pixels(mud, Colors.MUD)
            MUD += len(mud)

            # CREATE OCEAN
            sand, water = create_ocean(
                Rectangle(0, ocean_left_y, surface_w_offset_left,
                          Surface.h + (Surface.y - ocean_left_y)),
                True, random.randint(2, 10)
            )
            Draw.pixels(sand, material=Material.SAND)
            Draw.pixels(water, material=Material.WATER)
            SAND += len(sand)
            WATER += len(water)

            sand, water = create_ocean(
                Rectangle(WIDTH - surface_w_offset_right, ocean_right_y,
                          surface_w_offset_right, Surface.h + (Surface.y - ocean_right_y)),
                False, random.randint(4, 10)
            )
            Draw.pixels(sand, material=Material.SAND)
            Draw.pixels(water, material=Material.WATER)
            SAND += len(sand)
            WATER += len(water)

            # CREATE GRASS
            grid = make_grid(Surface, Draw.surface, {
                Material.BACKGROUND: 0,
                Material.SAND: 0,
                Material.WATER: 0
            }, 1)
            grass = create_grass(Surface, grid, 0, 1)
            Draw.pixels(grass, Colors.GRASS)
            GRASS += len(grass)

            # CREATE DEPOSIT STONE
            def success(pixels):
                global STONE
                STONE += len(pixels)
                Draw.pixels(pixels, Colors.BACKGROUND_CAVERN)

            def error(err):
                raise err

            Surf = Rectangle(surface_w_offset_left, Surface.y, WIDTH - surface_w_offset_right - surface_w_offset_left,
                             Surface.h)
            surface_grid = Grid.from_pixels(Surf, surface, 1, False)
            for x, y in Rectangle(surface_w_offset_left, Surface.y + Surface.h - surface_h_offset_bottom,
                                  WIDTH - surface_w_offset_right - surface_w_offset_left, surface_h_offset_bottom):
                surface_grid[x, y] = 1
            surface_grid.lock(0)
            parallel.to_process(partial(create_cave, surface_grid, ((5, 1),) * 4 + ((5, 8),) * 4, 0.75, 50), success,
                                error)

            # CREATE DEPOSIT COPPER
            def success(pixels):
                global COPPER
                COPPER += len(pixels)
                Draw.pixels(pixels, Colors.COPPER)

            def error(err):
                raise err

            Surf = Rectangle(surface_w_offset_left, Surface.y, WIDTH - surface_w_offset_right - surface_w_offset_left,
                             Surface.h)
            surface_grid = Grid.from_pixels(Surf, surface, 1, False)
            for x, y in Rectangle(surface_w_offset_left, Surface.y + Surface.h - surface_h_offset_bottom,
                                  WIDTH - surface_w_offset_right - surface_w_offset_left, surface_h_offset_bottom):
                surface_grid[x, y] = 1
            surface_grid.lock(0)
            parallel.to_process(partial(create_cave, surface_grid, ((5, 1),) * 3 + ((5, 8),) * 3, 0.75, 20), success,
                                error)

            # CREATE CAVES
            tree.traverse(CreateCaveTreeVisitor())


    scene_thread = threading.Thread(target=MainScene.run)
    parallel.init()
    scene_thread.start()
    while Draw.loop():
        Draw.count()
        time.sleep(1 / 20)
    parallel.clean()
    scene_thread.join()
