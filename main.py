import copy
import itertools
import math
import random
import string
import threading
from enum import Enum, auto
from functools import partial, lru_cache
from operator import itemgetter
from typing import Tuple, List, Dict, Union, Callable

from noise import pnoise2

WIDTH = 1920
HEIGHT = 1080
HILLS_LEVEL1_COUNT = 1
HILLS_LEVEL2_COUNT = 2
HILLS_LEVEL3_COUNT = 2
LOWLAND_LEVEL0_COUNT = 4
LOWLAND_LEVEL1_COUNT = 2
LOWLAND_LEVEL2_COUNT = 1
LOWLAND_LEVEL3_COUNT = 1
WATER_HEIGHT_MIN = 0.25
WATER_HEIGHT_MAX = 0.75
TUNNELS_MIN = 3
TUNNELS_MAX = 5
TUNNEL_PATHS_MIN = 3
TUNNEL_PATHS_MAX = 4
TUNNEL_PATH_WIDTH_MIN = 5
TUNNEL_PATH_WIDTH_MAX = 10
TUNNEL_WIDTH_MIN = 20
TUNNEL_WIDTH_MAX = 50
TREE_COUNT = 30
MIN_CAVE_PIXELS = 250
MIN_CAVE_REGION_WIDTH = 35
MIN_CAVE_REGION_HEIGHT = 35
LAKE_COUNT = 10
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
    NONE = auto()
    BASE = auto()
    STONE = auto()
    DIRT = auto()
    MUD = auto()
    COPPER = auto()
    GOLD = auto()
    BACKGROUND = auto()
    CAVE_BACKGROUND = auto()
    WATER = auto()
    LAVA = auto()
    SAND = auto()
    GRASS = auto()
    SNOW = auto()
    ICE = auto()


class Colors:
    """ Colors """

    BACKGROUND_SPACE = (8, 0, 60)
    BACKGROUND_SURFACE = (155, 209, 254)
    BACKGROUND_UNDERGROUND = (150, 107, 76)
    BACKGROUND_CAVERN = (127, 127, 127)
    BACKGROUND_UNDERWORLD = (0, 0, 0)
    BACKGROUND_ICE_BIOME = (74, 67, 60)
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

    LOG = (100, 70, 49)
    LEAF = (19, 145, 62)
    SNOW = (211, 236, 241)
    ICE = (144, 195, 232)


Color = Tuple[int, int, int]
PixelArray = List[Tuple[int, int]]
Pixel = Tuple[int, int]

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


class Polygon:
    def __init__(self, vertices: PixelArray):
        self.vertices = vertices
        self.equations = []
        self.bounds = [
            int(min(vertices, key=itemgetter(0))[0]),
            int(min(vertices, key=itemgetter(1))[1]),
            int(max(vertices, key=itemgetter(0))[0]),
            int(max(vertices, key=itemgetter(1))[1])
        ]

        def eq(y, x0, y0, m):
            return x0 + (y - y0) / m

        def eq1(y, x):
            return x

        for v0, v1 in zip(vertices, vertices[1:] + [vertices[0]]):
            x0, y0 = v0
            x1, y1 = v1
            if x0 != x1:
                m = (y1 - y0) / (x1 - x0)
                if m != 0:
                    self.equations.append(partial(eq, x0=x0, y0=y0, m=m))
                else:
                    self.equations.append(None)
            else:
                self.equations.append(partial(eq1, x=x0))
        assert len(self.vertices) == len(self.equations)

    def __iter__(self):
        return zip(self.vertices, self.vertices[1:] + [self.vertices[0]])

    def __getitem__(self, item):
        return tuple(self)[item]

    def contains(self, x: int, y: int):
        eq = []
        i = 0
        for v0, v1 in self:
            x0, y0 = v0
            x1, y1 = v1
            if (y0 <= y <= y1) or (y1 <= y <= y0):
                eq.append(self.equations[i])
            i += 1

        if len(eq) < 2:
            return False

        xs = []
        for eq0, eq1 in zip(eq, eq[1:] + [eq[0]]):
            if eq0 is not None and eq1 is not None:
                xs.append((eq0(y), eq1(y)))

        for x0, x1 in xs:
            if (x0 < x < x1) or (x1 < x < x0):
                return True
        return False

    @lru_cache()
    def get_points(self):
        points = []
        minx, miny, maxx, maxy = self.bounds
        for x in range(minx, maxx):
            for y in range(miny, maxy):
                if self.contains(x, y):
                    points.append((x, y))
        return points


class Grid:

    def __init__(self, x: int, y: int, w: int, h: int, default_state: int = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self._array = [default_state for _ in range(w * h)]
        self._locked = {}

    def __setitem__(self, key, value):
        x, y = key
        if (x, y) in self._locked:
            pass
        else:
            self._array[(y - self.y) * self.w + (x - self.x)] = value

    def __getitem__(self, item):
        x, y = item
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

    @lru_cache()
    def unlocked(self):
        return [(x, y) for x, y in self if not self.is_locked(x, y)]

    @staticmethod
    def from_rect(rect: Rectangle, default_state: int = 0):
        return Grid(rect.x, rect.y, rect.w, rect.h, default_state)

    @staticmethod
    def from_pixels(rect: Rectangle, pixels, state: int = 1, default_state: int = 0):
        grid = Grid.from_rect(rect, default_state)
        for x, y in pixels:
            grid[x, y] = state
        return grid


class PaintingAgent:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self._color = None
        self._restore_pos = x, y

    def _draw(self):
        Draw.pixels([(self.x, self.y)], self._color)

    def color(self, color: Color):
        self._color = color

    def save(self):
        self._restore_pos = self.x, self.y

    def restore(self):
        self.x, self.y = self._restore_pos

    def up(self):
        self.y += -1
        self._draw()

    def down(self):
        self.y += 1
        self._draw()

    def left(self):
        self.x += -1
        self._draw()

    def right(self):
        self.x += 1
        self._draw()

    def to(self, x: int, y: int):
        self.x = x
        self.y = y


def create_convex_polygon(inner: Rectangle, outer: Rectangle, num_of_points: int):
    global DEBUG

    assert num_of_points % 4 == 0 and num_of_points <= 8

    x0, y0 = inner.x, inner.y
    x3, y3 = inner.x + inner.w, inner.y + inner.h
    x4, y4 = outer.x, outer.y
    x7, y7 = outer.x + outer.w, outer.y + outer.h

    if DEBUG:
        Draw.outline(Rectangle(x0, y4, x3 - x0, y0 - y4), RED)
        Draw.outline(Rectangle(x3, y0, x7 - x3, y3 - y0), RED)
        Draw.outline(Rectangle(x0, y3, x7 - x0, y7 - y3), RED)
        Draw.outline(Rectangle(x4, y0, x0 - x4, y3 - y0), RED)
        Draw.outline(inner, BLUE)
        Draw.outline(outer, ORANGE)

    c0 = c1 = c2 = c3 = int(num_of_points / 4)

    points = []
    tmp = []
    for _ in range(c0):
        cx = random.randint(x0, x3)
        cy = random.randint(y4, y0)
        tmp.append((cx, cy))
    tmp.sort(key=itemgetter(0))
    points.extend(tmp)

    tmp = []
    for _ in range(c1):
        cx = random.randint(x3, x7)
        cy = random.randint(y0, y3)
        tmp.append((cx, cy))
    tmp.sort(key=itemgetter(1))
    points.extend(tmp)

    tmp = []
    for _ in range(c2):
        cx = random.randint(x0, x3)
        cy = random.randint(y3, y7)
        tmp.append((cx, cy))
    tmp.sort(key=itemgetter(0), reverse=True)
    points.extend(tmp)

    tmp = []
    for _ in range(c3):
        cx = random.randint(x4, x0)
        cy = random.randint(y0, y3)
        tmp.append((cx, cy))
    tmp.sort(key=itemgetter(1), reverse=True)
    points.extend(tmp)

    if DEBUG:
        for point in points:
            pygame.draw.circle(Draw.surface, RED, point, 5)
    return points


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


def cave_cellular_step(grid: Grid, death_limit: int, birth_limit: int):
    updated_grid = copy.deepcopy(grid)
    for x, y in grid.unlocked():
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
    return updated_grid


def pixels_between(p1: Tuple[int, int], p2: Tuple[int, int], width: int) -> PixelArray:
    if p1[0] > p2[0]:
        tmp = p2
        p2 = p1
        p1 = tmp
    x0, y0 = p1
    x1, y1 = p2
    m = (y1 - y0) / (x1 - x0)
    eq_x = lambda x: m * (x - x0) + y0

    circle = []
    for x in range(0, width):
        for y in range(0, width):
            v = pygame.Vector2(x, y)
            if abs(v.magnitude()) <= width:
                circle.append((x, y))

    pixels = {}
    c = -int(width / 2)
    for x in range(x0 * 10, x1 * 10):
        y = int(eq_x(x / 10))
        for x_, y_ in circle:
            pixels[int(x / 10) + x_ + c, y + y_ + c] = True

    return list(pixels.keys())


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


def extract_regions(grid: Grid, state: int) -> List[PixelArray]:
    visited = {}
    regions = []
    for x, y in grid.extract(state):
        if (x, y) not in visited:
            cells, newly_visited = flood_fill(x, y, 1, grid)
            visited.update(newly_visited)
            regions.append(cells)
    return regions


def create_cave(rect: Union[Rectangle, Grid], config_seq: Tuple[Tuple[int, int], ...], birth_chance: float,
                min_size: int = 75, max_size: int = 10000) -> PixelArray:
    if isinstance(rect, Rectangle):
        grid = Grid.from_rect(rect)
    else:
        grid = copy.deepcopy(rect)

    for x, y in grid.unlocked():
        grid[x, y] = 1 if random.random() > birth_chance else 0

    for step in range(len(config_seq)):
        death_limit, birth_limit = config_seq[step]
        grid = cave_cellular_step(grid, death_limit, birth_limit)

    caves = extract_regions(grid, 1)
    for cave in caves:
        if len(cave) < min_size or len(cave) > max_size:
            for x, y in cave:
                grid[x, y] = 0

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
                   octaves: int = 0, persistence: float = 0.5) -> (PixelArray, PixelArray):
    num_of_levels = l1 + l2 + l3 + b + h1 + h2 + h3 + 2

    level_order = [-1] * l1 + [-2] * l2 + [-3] * l3 + [0] * b + [1] * h1 + [2] * h2 + [3] * h3
    random.shuffle(level_order)
    level_order = [-random.randint(2, 3)] + level_order + [-random.randint(2, 3)]

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

    grass = []
    x = rect.x
    for y in noise:
        grass.append((x, int(rect.y + rect.h - rect.h * y)))
        x += 1

    return pixels, grass


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


def find_points_between_slopes(surface: PixelArray, s0: float, s1: float,
                               min_size: int = 10, max_size: int = 50) -> List[PixelArray]:
    lx, ly = surface[0]
    points = []
    current_points = []
    for x, y in surface[1:]:
        m = (y - ly) / (x - lx)
        if s0 <= m <= s1:
            if len(current_points) < max_size:
                current_points.append((x, y))
            else:
                points.append(current_points)
                current_points = []
                lx = x
                ly = y
        elif min_size <= len(current_points):
            points.append(current_points)
            current_points = []
            lx = x
            ly = y
        else:
            current_points = []
            lx = x
            ly = y
    return points


# TODO
def ore_feasibility_check(rect: Rectangle, mask: PixelArray, count: int):
    return True


def create_ore(rect: Rectangle, mask: PixelArray, min_size: int, max_size: int, count: int,
               iterations: int = 3) -> PixelArray:
    grid = Grid.from_pixels(rect, mask, -1)
    grid.lock(-1)

    pixels = []
    while len(pixels) < count and ore_feasibility_check(rect, mask + pixels, count) and iterations > 0:
        pixels += create_cave(grid, ((5, 1),) * 3 + ((5, 8),) * 3, 0.75, min_size, max_size)
        for x, y in pixels:
            grid[x, y] = 1
        grid.lock(1)
        iterations -= 1

    if len(pixels) < count and (iterations == 0 or not ore_feasibility_check(rect, mask + pixels, count)):
        raise NameError("Ore could not be created.")
    else:
        return pixels


def create_water_helper(rect: Rectangle, prob_per_cave: float) -> List[PixelArray]:
    global WATER_HEIGHT_MIN, WATER_HEIGHT_MAX, MaterialMap
    grid = MaterialMap.grid(rect, {Material.CAVE_BACKGROUND: 1}, 0)
    caves = extract_regions(grid, 1)
    results = []
    for cave in caves:
        if random.random() <= prob_per_cave:
            results += create_water(rect, cave, random.randint(
                int(WATER_HEIGHT_MIN * 100), int(WATER_HEIGHT_MAX * 100)
            ) / 100)
    return results


def create_ore_helper(rect: Rectangle, min_size: int, max_size: int, count: int,
                      iterations: int = 3):
    global MaterialMap
    grid = MaterialMap.grid(rect, {Material.CAVE_BACKGROUND: 1, Material.COPPER: 1, Material.GOLD: 1}, 0)
    return create_ore(rect, grid.extract(1), min_size, max_size, count, iterations)


def create_surface_ore_helper(rect: Rectangle, mask: PixelArray, min_size: int, max_size: int, count: int,
                              iterations: int = 3):
    global MaterialMap
    grid = MaterialMap.grid(rect, {
        Material.CAVE_BACKGROUND: 1,
        Material.COPPER: 1,
        Material.GOLD: 1,
        Material.STONE: 1,
        Material.SAND: 1,
        Material.WATER: 1,
    }, 0)
    mask += grid.extract(1)
    return create_ore(rect, mask, min_size, max_size, count, iterations)


def create_surface_cave_helper(rect: Rectangle, mask: PixelArray, config_seq: Tuple[Tuple[int, int], ...],
                               birth_chance: float,
                               min_size: int = 75, max_size: int = 10000):
    global MaterialMap
    grid = MaterialMap.grid(rect, {
        Material.CAVE_BACKGROUND: -1,
        Material.COPPER: -1,
        Material.GOLD: -1,
        Material.STONE: -1,
        Material.SAND: -1,
        Material.WATER: -1,
    }, 0)
    for x, y in mask:
        grid[x, y] = -1
    grid.lock(-1)
    return create_cave(grid, config_seq, birth_chance, min_size, max_size)


def create_tree_type1(x: int, y: int, height: int):
    agent = PaintingAgent(x, y)
    agent.color(Colors.LOG)

    for _ in range(height):
        agent.up()

    agent.color(Colors.LEAF)
    agent.up()
    agent.up()
    agent.up()
    agent.save()
    agent.left()
    agent.left()
    agent.left()
    agent.restore()
    agent.right()
    agent.right()
    agent.right()
    agent.restore()
    agent.down()
    agent.save()
    agent.left()
    agent.left()
    agent.restore()
    agent.right()
    agent.right()
    agent.restore()
    agent.down()
    agent.save()
    agent.left()
    agent.restore()
    agent.right()
    agent.restore()
    agent.down()


def create_tree_type2(x: int, y: int, height: int):
    agent = PaintingAgent(x, y)
    agent.color(Colors.LOG)

    for _ in range(height):
        agent.up()

    agent.color(Colors.LEAF)
    agent.up()
    agent.up()
    agent.up()
    agent.up()
    agent.up()
    agent.save()
    agent.left()
    agent.down()
    agent.down()
    agent.down()
    agent.down()
    agent.restore()
    agent.right()
    agent.down()
    agent.down()
    agent.down()
    agent.down()
    agent.restore()
    agent.down()
    agent.left()
    agent.left()
    agent.down()
    agent.down()
    agent.up()
    agent.left()
    agent.restore()
    agent.down()
    agent.right()
    agent.right()
    agent.down()
    agent.down()


def create_lianas(rect: Rectangle, max_height: int, prob: Union[float, Callable]):
    global MaterialMap

    grid = MaterialMap.grid(rect, {
        Material.DIRT: 1,
        Material.STONE: 1,
    }, -1)
    grid.lock(-1)

    agent = PaintingAgent(0, 0)
    agent.color(Colors.LEAF)
    p = prob if isinstance(prob, Callable) else (lambda _: prob)
    for x, y in grid.unlocked():
        agent.to(x, y)
        top, right, bottom, left = nbs_neumann(x, y, grid)
        if top == 1 and bottom == -1 and random.random() < p(y - grid.y):
            for y_ in range(random.randint(int(max_height / 2), max_height)):
                if y + y_ < grid.y + grid.h - 1:
                    _, _, b, _ = nbs_neumann(x, y + y_, grid)
                    if b == -1:
                        agent.down()


def create_ocean_desert_left(surface: PixelArray, width: int) -> PixelArray:
    global Surface

    x0, y0 = surface[0]
    height = Surface.y + Surface.h - y0
    m = height / width

    pixels = []
    h = height
    i = 0
    for _ in range(0, width):
        x1, y1 = surface[i]
        h -= m
        for y in range(y1, y1 + int(h) + random.randint(0, 1)):
            pixels.append((x1, y))
        i += 1
    return pixels


def create_ocean_desert_right(surface: PixelArray, width: int) -> PixelArray:
    global Surface

    x0, y0 = surface[-1]
    height = Surface.y + Surface.h - y0
    m = height / width

    pixels = []
    h = height
    i = len(surface) - 1
    for _ in range(0, width):
        x1, y1 = surface[i]
        h -= m
        for y in range(y1, y1 + int(h) + random.randint(0, 1)):
            pixels.append((x1, y))
        i -= 1
    return pixels


def create_lake(surface: PixelArray, at: int, width: int, height: int) -> (PixelArray, PixelArray):
    assert surface[0][0] < at < surface[-1][0]
    assert surface[0][0] < at + width < surface[-1][0]

    i = 0
    for _, y in surface:
        if _ == at:
            break
        i += 1

    x0, y0 = surface[i]
    x1, y1 = surface[i + width]
    ys = max(y0, y1)

    w0 = random.randint(1, width - 1)
    w2 = random.randint(1, width - w0)
    w1 = width - w0 - w2

    if random.random():
        tmp = int(w0)
        w0 = w2
        w2 = tmp

    m0 = height / w0
    m1 = height / w2

    pixels = []
    ch = 0
    for _ in range(0, width):
        x = x0 + _
        if _ < w0:
            ch += m0
        elif _ < w0 + w1:
            pass
        else:
            ch -= m1

        for y in range(ys, ys + int(ch)):
            pixels.append((x, y))

    global Surface
    removal = []
    for _ in range(0, width):
        y_ = surface[i + _][1]
        if y_ < ys:
            for y in range(y_, ys):
                removal.append((x0 + _, y))

    rect = get_bounding_rect(pixels)
    rect.h += int(width / 10)
    grid = Grid.from_pixels(rect, pixels, 1)
    grid.lock(1)

    for x, y in grid.unlocked():
        grid[x, y] = 1 if random.random() > 0.7 else 0
    for _ in range(2):
        grid = cave_cellular_step(grid, 4, 4)
    regions = extract_regions(grid, 1)
    region = max(regions, key=lambda x: len(x))

    return region, removal


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
        Material.SAND: Colors.SAND,
        Material.STONE: Colors.BACKGROUND_CAVERN,
        Material.DIRT: Colors.BACKGROUND_UNDERGROUND,
        Material.MUD: Colors.MUD,
        Material.COPPER: Colors.COPPER,
        Material.GOLD: Colors.GOLD
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


class _MaterialMap:
    global WIDTH, HEIGHT
    _array = [Material.NONE for _ in range(WIDTH * HEIGHT)]
    _lock = threading.Lock()

    def __setitem__(self, key, value):
        assert isinstance(value, Material)
        with self._lock:
            if isinstance(key, tuple):
                x, y = key
                self._array[y * WIDTH + x] = value
            else:
                for x, y in key:
                    self._array[y * WIDTH + x] = value

    def __getitem__(self, item):
        with self._lock:
            x, y = item
            return self._array[y * WIDTH + x]

    def __iter__(self):
        with self._lock:
            return itertools.product(range(0, WIDTH), range(0, HEIGHT))

    def grid(self, rect: Rectangle, mapping: Dict[Material, int], default: int):
        grid = Grid.from_rect(rect)
        with self._lock:
            for x, y in grid:
                key = self._array[y * WIDTH + x]
                if key in mapping:
                    grid[x, y] = mapping[key]
                else:
                    grid[x, y] = default
        return grid


MaterialMap = _MaterialMap()

if __name__ == '__main__':
    import pygame
    import time
    from bsp import BSPTree
    from parallel import ConcurrentExecutor

    scene_thread_running = True


    class Draw:
        """ Drawing interface """
        pygame.init()
        pygame.font.init()
        surface = pygame.display.set_mode([WIDTH, HEIGHT])
        font_size = 18
        font = pygame.font.Font("font.ttf", font_size)

        def __init__(self):
            raise NotImplemented

        @staticmethod
        def count():
            global WATER, LAVA, CAVE, DIRT, STONE, MUD, COPPER, GRASS, GOLD
            size = Draw.font_size

            if not Draw.surface.get_locked():
                pygame.draw.rect(Draw.surface, BLACK, (8, size, 200, 9 * size))
                Draw.surface.blit(Draw.font.render("WATER: %d" % WATER, True, RED), (8, size))
                Draw.surface.blit(Draw.font.render("LAVA: %d" % LAVA, True, RED), (8, 2 * size))
                Draw.surface.blit(Draw.font.render("CAVE: %d" % CAVE, True, RED), (8, 3 * size))
                Draw.surface.blit(Draw.font.render("DIRT: %d" % DIRT, True, RED), (8, 4 * size))
                Draw.surface.blit(Draw.font.render("STONE: %d" % STONE, True, RED), (8, 5 * size))
                Draw.surface.blit(Draw.font.render("MUD: %d" % MUD, True, RED), (8, 6 * size))
                Draw.surface.blit(Draw.font.render("COPPER: %d" % COPPER, True, RED), (8, 7 * size))
                Draw.surface.blit(Draw.font.render("GOLD: %d" % GOLD, True, RED), (8, 8 * size))
                Draw.surface.blit(Draw.font.render("GRASS: %d" % GRASS, True, RED), (8, 9 * size))

        @staticmethod
        def loop():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
                    pygame.image.save(Draw.surface, "img/" + name + ".png")
                    pygame.display.set_mode((1, 1))
                    pygame.display.flip()
                    return False
            pygame.display.flip()
            return True

        @staticmethod
        def rect(rect: Rectangle, color: Color = None):
            x, y, x1, y1 = rect.get_rect()
            if color is not None:
                pygame.draw.rect(Draw.surface, color, (x, y, x1, y1))
            else:
                for x0 in range(x, x1):
                    for y0 in range(y, y1):
                        color = PixelMaterialColorMap.get_color(x0, y0, Material.BACKGROUND)
                        pygame.draw.rect(Draw.surface, color, (x0, y0, 1, 1))

        @staticmethod
        def outline(rect: Rectangle, color: Color):
            x, y, x1, y1 = rect.get_rect()
            pygame.draw.rect(Draw.surface, color, (x, y, x1, y1), 1)

        @staticmethod
        def pixels(pixels, color: Color = None, material: Material = None):
            for x, y in pixels:
                if color is None:
                    if material is None:
                        c = PixelMaterialColorMap.get_color(x, y, Material.BACKGROUND)
                    else:
                        c = PixelMaterialColorMap.get_color(x, y, material)
                    pygame.draw.rect(Draw.surface, c, (x, y, 1, 1))
                else:
                    pygame.draw.rect(Draw.surface, color, (x, y, 1, 1))

        @staticmethod
        def line(p0: Tuple[int, int], p1: Tuple[int, int], color: Color, width: int = 5):
            pygame.draw.line(Draw.surface, color, p0, p1, width)

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


    class MainScene:
        """ Namespace for generation of everything """

        @staticmethod
        def run():
            global WIDTH, HEIGHT, DEBUG
            global DIRT, MUD, SAND, WATER, LAVA, GRASS, COPPER, STONE
            global MIN_CAVE_REGION_WIDTH, MIN_CAVE_REGION_HEIGHT
            global MaterialMap

            tree = BSPTree(0, Underground.y, WIDTH, Underground.h + Cavern.h)
            tree.grow(min_width=MIN_CAVE_REGION_WIDTH, min_height=MIN_CAVE_REGION_HEIGHT)

            PixelMaterialColorMap.add_rect(Space, {
                Material.BACKGROUND: Colors.BACKGROUND_SPACE,
            })

            PixelMaterialColorMap.add_rect(Surface, {
                Material.BACKGROUND: Colors.BACKGROUND_SURFACE,
                Material.CAVE_BACKGROUND: Colors.DIRT
            })

            PixelMaterialColorMap.add_rect(Underground, {
                Material.BACKGROUND: Colors.BACKGROUND_UNDERGROUND,
                Material.CAVE_BACKGROUND: Colors.DIRT,
            })

            PixelMaterialColorMap.add_rect(Cavern, {
                Material.BACKGROUND: Colors.BACKGROUND_CAVERN,
                Material.CAVE_BACKGROUND: Colors.STONE,
            })

            PixelMaterialColorMap.add_rect(Underworld, {
                Material.BACKGROUND: Colors.BACKGROUND_UNDERWORLD,
                Material.CAVE_BACKGROUND: Colors.STONE
            })

            ice_biome = Polygon([(200, 200), (400, 200), (600, 800), (100, 600)])
            for x, y in ice_biome.get_points():
                PixelMaterialColorMap.add_pixel(x, y, {
                    Material.DIRT: Colors.SNOW,
                    Material.STONE: Colors.ICE,
                    Material.CAVE_BACKGROUND: Colors.BACKGROUND_ICE_BIOME
                })

            MaterialMap[Space + Underworld] = Material.BACKGROUND

            Draw.rect(Space, Colors.BACKGROUND_SURFACE)
            Draw.rect(Surface, Colors.BACKGROUND_SURFACE)
            Draw.pixels(Underground, None, Material.STONE)
            Draw.pixels(Cavern, None, Material.STONE)
            Draw.rect(Underworld, Colors.BACKGROUND_UNDERWORLD)

            STONE += Underground.w * Underground.h
            STONE += Cavern.w * Cavern.h
            MaterialMap[Underground + Cavern] = Material.STONE

            if DEBUG:
                Draw.outline(Space, MAGENTA)
                Draw.outline(Surface, MAGENTA)
                Draw.outline(Underground, MAGENTA)
                Draw.outline(Cavern, MAGENTA)
                Draw.outline(Underworld, MAGENTA)
                for node in tree:
                    if node.leaf:
                        Draw.outline(Rectangle(node.x, node.y, node.w, node.h), RED)

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
            surface, grass = create_surface(
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

            Draw.pixels(surface, None, Material.DIRT)
            DIRT += len(surface)
            Coral = Rectangle(
                0,
                Space.h + surface_h_offset_top + surface_h - 1,
                WIDTH,
                surface_h_offset_bottom + 2
            )
            Draw.pixels(Coral, None, Material.DIRT)
            DIRT += Coral.w * Coral.h

            MaterialMap[surface] = Material.DIRT
            MaterialMap[Coral] = Material.DIRT

            if not scene_thread_running:
                return

            # PERLIN FUSION OF DIRT
            dirt = perlin_fusion(Underground + Cavern, [], 0.7, 0.2, 30, 6)
            dirt_dict = dict.fromkeys(dirt, True)
            Draw.pixels(dirt, None, Material.DIRT)
            DIRT += len(dirt)
            STONE -= len(dirt)
            MaterialMap[dirt] = Material.DIRT

            if not scene_thread_running:
                return

            # PERLIN FUSION OF MUD
            mud = perlin_fusion(Underground + Cavern, [], 0.2, 0.2, 40, 3)
            mud_dict = dict.fromkeys(mud, True)
            Draw.pixels(mud, None, Material.MUD)
            MUD += len(mud)
            MaterialMap[mud] = Material.MUD

            if not scene_thread_running:
                return

            # CREATE OCEAN LEFT
            ocean_left_y = grass[0][1]
            ocean_right_y = grass[-1][1]

            sand, water = create_ocean(
                Rectangle(0, ocean_left_y, surface_w_offset_left,
                          Surface.h + (Surface.y - ocean_left_y)),
                True, random.randint(2, 10)
            )
            Draw.pixels(sand, material=Material.SAND)
            Draw.pixels(water, material=Material.WATER)
            SAND += len(sand)
            WATER += len(water)
            MaterialMap[sand] = Material.SAND
            MaterialMap[water] = Material.WATER

            if not scene_thread_running:
                return

            # CREATE OCEAN RIGHT
            sand, water = create_ocean(
                Rectangle(WIDTH - surface_w_offset_right, ocean_right_y,
                          surface_w_offset_right, Surface.h + (Surface.y - ocean_right_y)),
                False, random.randint(4, 10)
            )
            Draw.pixels(sand, material=Material.SAND)
            Draw.pixels(water, material=Material.WATER)
            SAND += len(sand)
            WATER += len(water)
            MaterialMap[sand] = Material.SAND
            MaterialMap[water] = Material.WATER

            if not scene_thread_running:
                return

            # CREATE GRASS
            Draw.pixels(grass, Colors.GRASS)
            GRASS += len(grass)
            MaterialMap[grass] = Material.GRASS

            if not scene_thread_running:
                return

            forbidden_x = []

            # CREATE LAKE
            global LAKE_COUNT
            lake_set = find_points_between_slopes(grass, -0.1, .1, 35, 75)
            for _ in range(LAKE_COUNT if len(lake_set) > LAKE_COUNT else len(lake_set)):
                lake_points = random.choice(lake_set)
                x0, x1 = lake_points[0][0], lake_points[-1][0]
                for x in range(x0, x1):
                    forbidden_x.append(x)
                w = x1 - x0
                lake, air = create_lake(grass, x0, w, random.randint(5, 25))
                MaterialMap[lake] = Material.WATER
                MaterialMap[air] = Material.NONE
                WATER += len(lake)
                DIRT -= len(lake)
                DIRT -= len(air)
                Draw.pixels(lake, Colors.WATER)
                Draw.pixels(air, Colors.BACKGROUND_SURFACE)

            if not scene_thread_running:
                return

            # CREATE OCEAN DESERT LEFT
            sand = create_ocean_desert_left(grass, 50)
            Draw.pixels(sand, material=Material.SAND)
            SAND += len(sand)
            MaterialMap[sand] = Material.SAND

            if not scene_thread_running:
                return

            # CREATE OCEAN DESERT RIGHT
            sand = create_ocean_desert_right(grass, 50)
            Draw.pixels(sand, material=Material.SAND)
            SAND += len(sand)
            MaterialMap[sand] = Material.SAND

            if not scene_thread_running:
                return

            def success(pixels):
                global CAVE, DIRT, MUD, STONE, MaterialMap
                CAVE += len(pixels)
                MaterialMap[pixels] = Material.CAVE_BACKGROUND

                for x, y in pixels:
                    if (x, y) in mud_dict:
                        MUD -= 1
                    elif (x, y) in dirt_dict:
                        DIRT -= 1
                    else:
                        STONE -= 1
                Draw.pixels(pixels, material=Material.CAVE_BACKGROUND)

            def error(err):
                raise err

            # CREATE TOP CAVES
            global TUNNELS_MIN, TUNNELS_MAX, TUNNEL_PATHS_MIN, TUNNEL_PATHS_MAX, TUNNEL_PATH_WIDTH_MIN, \
                TUNNEL_PATH_WIDTH_MAX, TUNNEL_WIDTH_MIN, TUNNEL_WIDTH_MAX
            top_cave_futures = {}
            top_caves = []
            for node in tree:
                if node.leaf and node.y == Surface.y + Surface.h:
                    rect = Rectangle(node.x, node.y, node.w, node.h)
                    top_caves.append(rect)

                    if node.y == Surface.y + Surface.h:  # TOP NODE
                        pass

                    if node.y + node.h == Underworld.y:  # BOTTOM NODE
                        pass

                    func = partial(create_cave, rect, ((5, 1),) * 8 + ((5, 8),) * 4, .75)
                    cave_future = ConcurrentExecutor.submit(func, success, error)
                    top_cave_futures[node] = cave_future

            # CREATE SURFACE TUNNELS
            tunnel_set = find_points_between_slopes(grass, 0.4, 3, TUNNEL_WIDTH_MIN, TUNNEL_WIDTH_MAX)
            tunnel_set += find_points_between_slopes(grass, -3, -0.4, TUNNEL_WIDTH_MIN, TUNNEL_WIDTH_MAX)
            random.shuffle(tunnel_set)
            tunnel_count = random.randint(TUNNELS_MIN,
                                          TUNNELS_MAX if len(tunnel_set) > TUNNELS_MAX else len(tunnel_set))

            for _ in range(tunnel_count):
                tunnel_points = tunnel_set[_]
                assert len(tunnel_points) >= TUNNEL_WIDTH_MIN
                assert len(tunnel_points) <= TUNNEL_WIDTH_MAX

                Y = Surface.y + Surface.h
                P = pygame.Vector2(tunnel_points[0])
                Q = pygame.Vector2(tunnel_points[-1])

                # as starting point select lowest point
                Cv = pygame.Vector2(max(tunnel_points, key=itemgetter(1)))

                d = Q.x - P.x

                if P.y > Q.y:  # Q4
                    B = pygame.Vector2(random.randint(int(Q.x) + int(0.5 * d), int(Q.x + int(d))), Y)
                    A = pygame.Vector2(random.randint(int(P.x) + int(0.5 * d), int(B.x) - int(d / 2)), Y)
                else:  # Q3
                    A = pygame.Vector2(random.randint(int(P.x - int(d)), int(P.x) - int(0.5 * d)), Y)
                    B = pygame.Vector2(random.randint(int(A.x) + int(d / 2), int(Q.x) - int(0.5 * d)), Y)

                APm = (P.y - A.y) / (P.x - A.x)
                BQm = (Q.y - B.y) / (Q.x - B.x)
                APeq = lambda y: (y - P.y + APm * P.x) / APm
                BQeq = lambda y: (y - Q.y + BQm * Q.x) / BQm

                if P.y > Q.y:  # Q3
                    sy = int(P.y)
                else:
                    sy = int(Q.y)

                ey = Surface.y + Surface.h
                cy = sy
                dy = ey - sy
                paths_count = random.randint(TUNNEL_PATHS_MIN, TUNNEL_PATHS_MAX)
                height_per_tunnel = int(dy / paths_count)
                points = [Cv]

                for _ in range(paths_count):
                    y = cy + height_per_tunnel
                    if _ % 2 == 0:
                        if P.y > Q.y:  # Q3
                            points.append(pygame.Vector2(int(BQeq(y)), y))
                        else:
                            points.append(pygame.Vector2(int(APeq(y)), y))

                    else:
                        if P.y > Q.y:  # Q3
                            points.append(pygame.Vector2(int(APeq(y)), y))
                        else:
                            points.append(pygame.Vector2(int(BQeq(y)), y))
                    cy = y

                # choose x caves by their distance
                EP = points[-1]
                magnitudes = []
                for cave in top_caves:
                    CP = pygame.Vector2(cave.get_center())
                    mag = (CP - EP).magnitude()
                    magnitudes.append(mag)
                cave_magnitude = list(zip(magnitudes, top_caves))
                cave_magnitude.sort(key=itemgetter(0))
                caves = tuple(map(itemgetter(1), cave_magnitude[:1]))
                cave = random.choice(caves)

                CP = pygame.Vector2(cave.get_center())
                if CP.x == points[-1].x:
                    CP.x += 1
                points.append(CP)

                min_x = min(P.x, A.x, cave.x)
                max_x = max(Q.x, B.x, cave.x + cave.w)
                min_y = min(P.y, Q.y)
                max_y = max(A.y, Q.y, cave.y + cave.h)
                rect = Rectangle(min_x - TUNNEL_PATH_WIDTH_MAX, min_y, (max_x - min_x) + 2 * TUNNEL_PATH_WIDTH_MAX,
                                 max_y - min_y)

                # prepare for cellular automation
                grid = Grid.from_rect(rect, -1)
                for P1, P2 in zip(points, points[1:]):
                    pixels_outer = dict.fromkeys(
                        pixels_between((int(P1.x), int(P1.y)), (int(P2.x), int(P2.y)), TUNNEL_PATH_WIDTH_MAX)
                    )
                    for x, y in pixels_outer.keys():
                        grid[x, y] = 0

                for P1, P2 in zip(points, points[1:]):
                    pixels_inner = dict.fromkeys(
                        pixels_between((int(P1.x), int(P1.y)), (int(P2.x), int(P2.y)), TUNNEL_PATH_WIDTH_MIN)
                    )
                    for x, y in pixels_inner.keys():
                        grid[x, y] = 1

                if DEBUG:
                    for P1, P2 in zip(points, points[1:]):
                        Draw.pixels(
                            pixels_between((int(P1.x), int(P1.y)), (int(P2.x), int(P2.y)), TUNNEL_PATH_WIDTH_MAX),
                            RED)
                    Draw.line(P, A, BLUE, 3)
                    Draw.line(Q, B, BLUE, 3)
                    Draw.outline(rect, RED)
                    Draw.outline(cave, BLUE)

                grid.lock(-1)
                grid.lock(1)

                # cellular automata
                for x, y in grid.extract(0):
                    grid[x, y] = 1 if random.random() > 0.45 else 0
                for step in range(4):
                    grid = cave_cellular_step(grid, 3, 4)
                tunnel = grid.extract(1)
                Draw.pixels(tunnel, None, Material.CAVE_BACKGROUND)
                MaterialMap[tunnel] = Material.CAVE_BACKGROUND

                # cosmetic lianas
                create_lianas(rect, 10, lambda _: (rect.h - _) / rect.h)

            # CREATE TREES
            global TREE_COUNT
            for _ in range(TREE_COUNT):
                sp = random.choice(grass)
                while sp[0] in forbidden_x:
                    sp = random.choice(grass)
                for x in range(sp[0] - 6, sp[0] + 6):
                    forbidden_x.append(x)

                if random.random() > 0.5:
                    create_tree_type1(*sp, random.randint(12, 18))
                else:
                    create_tree_type2(*sp, random.randint(12, 18))

            if not scene_thread_running:
                return

            # CREATE DEPOSIT STONE
            def success1(pixels):
                global STONE, MaterialMap
                STONE += len(pixels)
                MaterialMap[pixels] = Material.STONE
                Draw.pixels(pixels, None, Material.STONE)

            def error1(err):
                raise err

            Surf = Rectangle(surface_w_offset_left, Surface.y + Surface.h - surface_h_offset_bottom - 1,
                             WIDTH - surface_w_offset_right - surface_w_offset_left,
                             surface_h_offset_bottom + 1)
            surface_grid = Grid.from_pixels(Surface, surface, 0, 1)
            for x, y in Surf:
                surface_grid[x, y] = 0
            stone_deposit_future = ConcurrentExecutor.submit(
                partial(create_surface_cave_helper, Surface, surface_grid.extract(1), ((5, 1),) * 5 + ((5, 8),) * 4,
                        .75),
                success1,
                error1)

            # CREATE DEPOSIT COPPER
            def success2(pixels):
                global COPPER, MaterialMap
                COPPER += len(pixels)
                MaterialMap[pixels] = Material.COPPER
                Draw.pixels(pixels, None, Material.COPPER)

            def error2(err):
                raise err

            for x, y in Surf:
                surface_grid[x, y] = 0
            copper_deposit_future = ConcurrentExecutor.submit(
                partial(create_surface_ore_helper, Surface, surface_grid.extract(1), 10, 50, 1, 1),
                success2,
                error2)
            copper_deposit_future.after(stone_deposit_future, subscribe_for_result=False)

            if not scene_thread_running:
                return

            # CREATE OTHER CAVES
            global MIN_CAVE_PIXELS

            cave_futures = {}
            for node in tree:
                if node.leaf and not node.y == Surface.y + Surface.h:
                    rect = Rectangle(node.x, node.y, node.w, node.h)

                    if node.y == Surface.y + Surface.h:  # TOP NODE
                        pass

                    if node.y + node.h == Underworld.y:  # BOTTOM NODE
                        pass

                    if random.random() > 0.2:
                        func = partial(create_cave, rect, ((5, 1),) * 8 + ((5, 8),) * 4,
                                       .75, min_size=MIN_CAVE_PIXELS)
                    else:
                        func = partial(create_cave, rect, ((3, 4),) * 2 + ((4, 4),) * 2,
                                       .575, min_size=MIN_CAVE_PIXELS)

                    cave_future = ConcurrentExecutor.submit(func, success, error)
                    cave_future.after(*top_cave_futures.values(), subscribe_for_result=False)
                    cave_futures[node] = cave_future

            # GENERATE COPPER, GOLD, LIQUID
            cave_futures.update(top_cave_futures)
            for node in tree:
                if node.leaf:
                    rect = Rectangle(node.x, node.y, node.w, node.h)

                    # CREATE COPPER
                    def success1(pixels):
                        global COPPER, DIRT, MUD, STONE, MaterialMap
                        COPPER += len(pixels)
                        for x, y in pixels:
                            if (x, y) in mud_dict:
                                MUD -= 1
                            elif (x, y) in dirt_dict:
                                DIRT -= 1
                            else:
                                STONE -= 1
                        MaterialMap[pixels] = Material.COPPER
                        Draw.pixels(pixels, None, Material.COPPER)

                    def error1(err):
                        print(err)

                    copper_future = ConcurrentExecutor.submit(
                        partial(create_ore_helper, rect, min_size=10, max_size=50, count=1),
                        success1,
                        error1)
                    copper_future.after(*top_cave_futures.values(), subscribe_for_result=False)
                    copper_future.after(*cave_futures.values(), subscribe_for_result=False)

                    # CREATE GOLD
                    def success2(pixels):
                        global GOLD, DIRT, MUD, STONE, MaterialMap
                        GOLD += len(pixels)
                        for x, y in pixels:
                            if (x, y) in dirt_dict:
                                DIRT -= 1
                            elif (x, y) in mud_dict:
                                MUD -= 1
                            else:
                                STONE -= 1
                        MaterialMap[pixels] = Material.GOLD
                        Draw.pixels(pixels, None, Material.GOLD)

                    def error2(err):
                        print(err)

                    gold_future = ConcurrentExecutor.submit(
                        partial(create_ore_helper, rect, min_size=5, max_size=20, count=1),
                        success2,
                        error2)
                    gold_future.after(*cave_futures.values(), subscribe_for_result=False)

                    # CREATE LIQUID
                    def success3(pixels):
                        global WATER, LAVA, MaterialMap

                        if random.random() > 0.8:
                            LAVA += len(pixels)
                            MaterialMap[pixels] = Material.LAVA
                            Draw.pixels(pixels, material=Material.LAVA)
                        else:
                            WATER += len(pixels)
                            MaterialMap[pixels] = Material.WATER
                            Draw.pixels(pixels, material=Material.WATER)

                    def error3(err):
                        raise err

                    liquid_future = ConcurrentExecutor.submit(
                        partial(create_water_helper, rect, 0.2),
                        success3,
                        error3)
                    liquid_future.after(*cave_futures.values(), subscribe_for_result=False)

            ConcurrentExecutor.run()


    scene_thread = threading.Thread(target=MainScene.run)
    scene_thread.start()
    while Draw.loop():
        Draw.count()
        time.sleep(1 / 25)
    scene_thread_running = False
    scene_thread.join()
    ConcurrentExecutor.terminate()
    ConcurrentExecutor.join()
