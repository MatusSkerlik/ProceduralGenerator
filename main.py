import math
import random
import string
import time
from abc import abstractmethod, ABC
from collections import ChainMap
from copy import copy
from enum import Enum, auto
from typing import Tuple, List, Dict

import pygame
from PIL import Image, ImageDraw

from csp import MinConflictsSolver, Problem, FunctionConstraint

WIDTH = 1920
HEIGHT = 600


class Config(dict):

    def __init__(self):
        super(Config, self).__init__()
        self['WORLD_WIDTH'] = WIDTH
        self['WORLD_HEIGHT'] = HEIGHT
        self['HILLS_LEVEL1_COUNT'] = 1
        self['HILLS_LEVEL2_COUNT'] = 3
        self['HILLS_LEVEL3_COUNT'] = 2
        self['LOWLAND_LEVEL0_COUNT'] = 3
        self['LOWLAND_LEVEL1_COUNT'] = 2
        self['LOWLAND_LEVEL2_COUNT'] = 1
        self['LOWLAND_LEVEL3_COUNT'] = 1
        self['DEBUG'] = True

    def is_debug(self):
        return self['DEBUG']

    def get_width(self):
        return self['WORLD_WIDTH']

    def get_height(self):
        return self['WORLD_HEIGHT']

    def get_surface_params(self):
        return self['HILLS_LEVEL1_COUNT'], self['HILLS_LEVEL2_COUNT'], self['HILLS_LEVEL3_COUNT'], \
               self['LOWLAND_LEVEL0_COUNT'], self['LOWLAND_LEVEL1_COUNT'], self['LOWLAND_LEVEL2_COUNT'], \
               self['LOWLAND_LEVEL3_COUNT']


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


class Color(Enum):
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


class Rectangle(ABC):
    """ Rectangle """
    x: int
    y: int
    w: int
    h: int
    color: Tuple[int, int, int]

    def get_x(self) -> int:
        return self.x

    def set_x(self, x: int):
        self.x = x

    def set_y(self, y: int):
        self.y = y

    def get_y(self) -> int:
        return self.y

    def get_width(self) -> int:
        return self.w

    def get_height(self) -> int:
        return self.h

    def get_rect(self):
        return self.x, self.y, self.w, self.h

    def get_points(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def get_center(self):
        return int(self.x + (self.w / 2)), int(self.y + (self.h / 2))

    def get_color(self):
        return self.color

    def is_inside(self, rect: 'Rectangle'):
        x, y, w, h = rect.get_rect()
        return self.x < x and self.x + self.w > x + w and self.y < y and self.y + self.h > y + h

    def is_pixel_inside(self, x: int, y: int):
        return self.x <= x <= (self.x + self.w) and self.y <= y <= (self.y + self.h)


class Area(Rectangle):
    """ Represents area of canvas with color mappings """

    material_to_color: Dict[Material, Color]

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def get_material_color(self, material: 'Material'):
        return self.material_to_color[material].value


class TranslatedArea(Area):
    """ Translated area, used with Horizontal Area as helper class """

    def get_points(self):
        return self.x, self.y, self.x + self.w, self.y - self.h

    def is_inside(self, area: 'Area'):
        x, y, w, h = area.get_rect()
        return self.x < x and self.x + self.w > x + w and self.y - self.h < y and self.h > y + h


class HorizontalArea(Area):
    """ Horizontal canvas separator """

    def __init__(self, y, offset_y):
        super(HorizontalArea, self).__init__(0, y, WIDTH, offset_y - y)

    def translate(self, area: Area):
        t_area = TranslatedArea(area.get_x(), self.y + self.h - area.get_y(), area.get_width(), area.get_height())
        t_area.material_to_color = area.material_to_color
        return t_area


class HorizontalAreaGroup(Area):
    """ Group of Horizontal Areas """

    def __init__(self, areas: Tuple[HorizontalArea, ...]):
        sy = None
        by = None
        for area in areas:
            if sy is None:
                sy = area.y
            if by is None:
                by = area.y + area.h

            if area.y < sy:
                sy = area.y

            if area.y + area.h > by:
                by = area.y + area.h

        super(HorizontalAreaGroup, self).__init__(0, sy, WIDTH, by - sy)


class Space(HorizontalArea):
    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_SURFACE
    }

    def __init__(self, height):
        super(Space, self).__init__(0, 2 * height / 20)


class Surface(HorizontalArea):
    color = Color.BACKGROUND_SURFACE.value

    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_UNDERGROUND,
        Material.BASE: Color.DIRT,
        Material.SECONDARY: Color.STONE,
        Material.TERTIARY: Color.MUD,
    }

    def __init__(self, height):
        super(Surface, self).__init__(2 * height / 20, 5 * height / 20)


class Underground(HorizontalArea):
    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_UNDERGROUND,
        Material.BASE: Color.DIRT,
        Material.SECONDARY: Color.STONE,
        Material.TERTIARY: Color.COPPER,
    }

    def __init__(self, height):
        super(Underground, self).__init__(5 * height / 20, 8 * height / 20)


class Cavern(HorizontalArea):
    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_CAVERN,
        Material.BASE: Color.STONE,
        Material.SECONDARY: Color.DIRT,
        Material.TERTIARY: Color.COPPER,
    }

    def __init__(self, height):
        super(Cavern, self).__init__(8 * height / 20, 17 * height / 20)


class Underworld(HorizontalArea):
    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_UNDERWORLD
    }

    def __init__(self, height):
        super(Underworld, self).__init__(17 * height / 20, height)


class HorizontalAreas:
    """ Helper class which clusters horizontal areas """

    Space = None
    Surface = None
    Underground = None
    Cavern = None
    Underworld = None

    @classmethod
    def init(cls, config: Config):
        h = config.get_height()
        setattr(cls, "Space", Space(h))
        setattr(cls, "Surface", Surface(h))
        setattr(cls, "Underground", Underground(h))
        setattr(cls, "Cavern", Cavern(h))
        setattr(cls, "Underworld", Underworld(h))


class ForestPixelMapping(Area):
    """ Area with specific color mapping """

    material_to_color = {
        Material.BACKGROUND: Color.BACKGROUND_UNDERGROUND,

        Material.BASE: Color.DIRT,
        Material.SECONDARY: Color.STONE,
        Material.TERTIARY: Color.MUD,
        Material.MATERIAL3: Color.IRON,
        Material.MATERIAL4: Color.SILVER,
        Material.MATERIAL5: Color.GOLD
    }


class JunglePixelMapping(Area):
    """ Area with specific color mapping """

    material_to_color = {
        Material.BACKGROUND: Color.MUD,

        Material.BASE: Color.STONE,
        Material.SECONDARY: Color.STONE,
        Material.TERTIARY: Color.MUD,

        Material.MATERIAL3: Color.IRON,
        Material.MATERIAL4: Color.SILVER,
        Material.MATERIAL5: Color.GOLD
    }


class PixelMapping:
    """ Class which maps colors of materials into individual pixel """

    # biomes are in specific layers so they can rewrite their material to color maps
    # biomes in same layer cannot intersect with each other
    biomes_layer_0: List[HorizontalArea] = []
    biomes_layer_1: List[Area] = []

    # every pixel has chain map with material to color mapping
    # needs to be generated before actual rendering
    pixel_to_material_color_map: Dict[Tuple[int, int], ChainMap[Material, Color]] = {}

    @classmethod
    def generate_jungle(cls, config: Config):
        area = HorizontalAreaGroup((HorizontalAreas.Surface, HorizontalAreas.Cavern))
        x, y = area.get_x(), area.get_y()
        w, h = area.get_width(), area.get_height()

        return JunglePixelMapping(
            random.randint(int(w / 10), int(w - w / 10)),
            y,
            random.randint(int(w / 5), int(w / 4)),
            h
        )

    @classmethod
    def generate_forest(cls, config: Config):
        area = HorizontalAreaGroup((HorizontalAreas.Surface, HorizontalAreas.Underground))
        x, y = area.get_x(), area.get_y()
        w, h = area.get_width(), area.get_height()

        return ForestPixelMapping(
            random.randint(int(w / 10), int(w - w / 10)),
            y,
            random.randint(int(w / 8), int(w / 7)),
            h
        )

    @classmethod
    def init(cls, config: Config):
        """
        Generate biomes and map every pixel to material to color map,
        so every pixel has its color palette

        :param config: nothing else to say :))
        """
        cls.biomes_layer_0.append(HorizontalAreas.Space)
        cls.biomes_layer_0.append(HorizontalAreas.Surface)
        cls.biomes_layer_0.append(HorizontalAreas.Underground)
        cls.biomes_layer_0.append(HorizontalAreas.Cavern)
        cls.biomes_layer_0.append(HorizontalAreas.Underworld)

        # TODO CSP biomes must be separated
        cls.biomes_layer_1.append(cls.generate_jungle(config))
        cls.biomes_layer_1.append(cls.generate_forest(config))

        w, h = config.get_width(), config.get_height()

        # generation of material to color map for every pixel
        for x in range(w):
            for y in range(h):

                material_to_color_0 = {}
                material_to_color_1 = {}

                for biome in cls.biomes_layer_0:
                    if biome.is_pixel_inside(x, y):
                        material_to_color_0 = biome.material_to_color
                        break

                for biome in cls.biomes_layer_1:
                    if biome.is_pixel_inside(x, y):
                        material_to_color_1 = biome.material_to_color
                        break

                cls.pixel_to_material_color_map[(x, y)] = ChainMap(
                    material_to_color_1,
                    material_to_color_0
                )

    @classmethod
    def get_pixel_mapping(cls, x, y) -> ChainMap[Material, Color]:
        return cls.pixel_to_material_color_map[(x, y)]


class Generator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass


class RegionGenerator(Generator, ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Tuple['Region', ...]:
        pass


class PointGenerator(Generator, ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Tuple[Tuple[int, int], ...]:
        pass


class Region(Rectangle, Generator, ABC):
    """ Represents region of canvas, with specific generation algorithm implemented """

    def __init__(self, x, y, w, h, material: Material):
        self.pixels = []
        self.material = material
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def get_material(self):
        return self.material

    def get_pixels(self):
        return tuple(self.pixels)

    def set_x(self, x: int):
        for pixel in self.pixels:
            pixel[0] = (pixel[0] - self.x + x)
        super().set_x(x)

    def set_y(self, y: int):
        for pixel in self.pixels:
            pixel[1] = (pixel[1] - self.y + y)
        super().set_y(y)

    def update_dimensions(self):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0
        initialized = False

        for x, y in self.pixels:
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

        self.x = min_x
        self.y = min_y
        self.w = max_x - min_x + 1
        self.h = max_y - min_y + 1


class OreRegion(Region):

    def generate(self, count: int = 100):
        """ Generates ore in stochastic fashion """
        if count > 0:
            i = 0
            coords = {}
            pixels = [[self.x, self.y]]
            sides = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            while i < count:
                # TODO should be more deterministic
                _ = int(i - i * 0.07)
                if (i - _) < 10:  # TODO can get stuck here
                    _ = 0
                current = pixels[random.randint(_, i)]
                x0, y0 = sides[random.randint(0, 3)]
                x1 = current[0] + x0
                y1 = current[1] + y0

                if not coords.get((x1, y1), False):
                    coords[(x1, y1)] = True
                    pixels.append([x1, y1])
                    i += 1
            self.pixels.extend(pixels)
            self.update_dimensions()


class CaveRegion(Region):

    def generate(self, count: int = 100):
        """ Generates caves """
        if count > 0:
            i = 0
            coords = {}
            pixels = [[self.x, self.y]]
            sides = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            while i < count:
                # TODO should be more deterministic
                _ = int(i - i * 0.6)
                if (i - _) < 10:  # TODO can get stuck here
                    _ = 0
                current = pixels[random.randint(_, i)]
                x0, y0 = sides[random.randint(0, 3)]
                x1 = current[0] + x0
                y1 = current[1] + y0

                if not coords.get((x1, y1), False):
                    coords[(x1, y1)] = True
                    pixels.append([x1, y1])
                    i += 1
            self.pixels.extend(pixels)
            self.update_dimensions()


class SurfaceRegion(Region):
    """ Represents region of surface ( hill, lowland, flat ) """

    def generate(self, y_array: Tuple[float, ...]):
        area = HorizontalAreaGroup((HorizontalAreas.Surface,))
        x, y = area.get_x(), area.get_y()
        w, h = area.get_width(), area.get_height()

        for x, _y in enumerate(y_array):
            # y + h is surface base
            # (y_array[x] * .75) * h will scale y in range 0:1 to range 0h:.75h
            #  - (h / 4) will lift everything to .25 %
            for __y in range(int(y + h - (y_array[x] * .75) * h - (h / 4)), y + h):
                self.pixels.append((self.x + x, __y))


class Drawer(ABC):

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def free(self):
        pass

    @abstractmethod
    def draw_rect(self, rect: Rectangle, color: Tuple[int, int, int] = None):
        pass

    @abstractmethod
    def draw_area(self, area: Area):
        pass

    @abstractmethod
    def draw_region(self, region: Region):
        pass

    @abstractmethod
    def draw_outline(self, region: Region):
        pass

    @abstractmethod
    def draw_progress(self, text: string):
        pass

    @abstractmethod
    def draw_polygon(self, points: Tuple[Tuple[int, int], ...], color: Tuple[int, int, int]):
        pass

    @abstractmethod
    def fill(self, color):
        pass


class PillowDrawer(Drawer):
    """ Drawing interface """

    def __init__(self, config):
        self.config = config
        self.canvas = Image.new("RGB", (config.get_width(), config.get_height()))
        self.drawer = ImageDraw.Draw(self.canvas)

    def init(self):
        return

    def free(self):
        name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        self.canvas.save("img/" + name + ".png")
        self.canvas.show()

    def draw_rect(self, rect: Rectangle, color: Tuple[int, int, int] = None):
        pass

    def draw_area(self, area: Area):
        x, y, x1, y1 = area.get_points()
        color = area.get_color()
        self.drawer.rectangle((x, y, x1, y1), color)

    def draw_region(self, region: Region):
        for block in region.get_pixels():
            self.draw_area(block)

    def draw_outline(self, region: Region):
        x, y, x1, y1 = region.get_points()
        color = (255, 0, 0)
        self.drawer.rectangle((x, y, x1, y1), fill=None, outline=color)

    def draw_progress(self, text: string):
        pass

    def draw_polygon(self, points: Tuple[Tuple[int, int], ...], color: Tuple[int, int, int]):
        pass

    def fill(self, color):
        w = self.config.get_width()
        h = self.config.get_height()
        self.drawer.rectangle((0, 0, w, h), color)


class PygameDrawer(Drawer):
    """ Drawing interface """

    def __init__(self, config: Config):
        self.config = config
        self.canvas = None
        self.running = False
        self._font = None

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                pygame.font.quit()
                exit(0)

    def init(self):
        pygame.init()
        pygame.font.init()
        w = self.config.get_width()
        h = self.config.get_height() + 20
        self.canvas = pygame.display.set_mode([w, h])
        self._font = pygame.font.Font("font.ttf", 12)

    def free(self):
        name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        pygame.image.save(self.canvas, "img/" + name + ".png")
        while True:
            time.sleep(0.016)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    pygame.font.quit()
                    exit(0)

    def draw_rect(self, rect: Rectangle, color: Tuple[int, int, int] = None):
        if color is None:
            color = rect.get_color()

        x, y, x1, y1 = rect.get_points()
        pygame.draw.rect(self.canvas, color, (x, y, x1, y1))
        pygame.display.update((x, y, x1, y1))

    def draw_area(self, area: Area):
        x, y, x1, y1 = area.get_points()
        for _x in range(x, x1):
            for _y in range(y, y1):
                color = PixelMapping.get_pixel_mapping(_x, _y)[Material.BACKGROUND].value  # TODO handler
                pygame.draw.rect(self.canvas, color, (_x, _y, 1, 1))
            pygame.display.update((_x, y, _x + 1, y1))
            self._handle_events()

    def draw_region(self, region: Region):
        material = region.get_material()
        for pixel in region.get_pixels():
            x, y = pixel
            color = PixelMapping.get_pixel_mapping(x, y)[material].value  # TODO handler
            pygame.draw.rect(self.canvas, color, (x, y, 1, 1))

        self._handle_events()
        x, y, x1, y1 = region.get_rect()
        pygame.display.update((x, y, x1, y1))

    def draw_outline(self, region: Region):
        x, y, x1, y1 = region.get_rect()
        color = (255, 0, 0)
        pygame.draw.rect(self.canvas, color, (x, y, x1, y1), 1)
        pygame.display.update((x, y, x1, y1))
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

    def draw_progress(self, text: string):
        w = self.config.get_width()
        h = self.config.get_height()

        pygame.draw.rect(self.canvas, (0, 128, 128), (0, h, w, h + 64))
        font = self._font.render(text, False, (255, 255, 255))
        self.canvas.blit(font, (4, h + 4))
        pygame.display.update((0, h, w, h))

    def fill(self, color):
        # pygame.display.flip()
        pass


class OreDistributionProblem(Problem):
    """ Will distribute regions so they dont have intersection """

    def __init__(self, area: Area, regions: Tuple[OreRegion, ...], step_size: int = 10):
        super(Problem, self).__init__()
        self._constraints = []
        self._variables = {}
        self._regions = regions
        self._area = area
        assignments = {}

        # add variables
        d = 0
        for i, region in enumerate(regions):
            x, y, w, h = region.get_x(), region.get_y(), region.get_width(), region.get_height()
            self.addVariable("x" + str(i),
                             range(int(region.get_width() / 2), area.get_width() - int(region.get_width() / 2),
                                   step_size))
            self.addVariable("y" + str(i), range(area.get_y() + int(region.get_height() / 2),
                                                 area.get_y() + area.get_height() - int(region.get_height() / 2),
                                                 step_size))

            assignments["x" + str(i)] = random.randint(int(region.get_width() / 2),
                                                       area.get_width() - int(region.get_width() / 2))
            assignments["y" + str(i)] = random.randint(area.get_y() + int(region.get_height() / 2),
                                                       area.get_y() + area.get_height() - int(region.get_height() / 2))

            if d < w:
                d = w
            if d < h:
                d = h

        d *= .8

        # add constraints
        def distance_constraint(x0, y0, x1, y1):
            return math.sqrt(((x1 - x0) ** 2) + ((y1 - y0) ** 2)) > d

        for i in range(0, len(regions)):
            for j in range(i + 1, len(regions)):

                if i == j:
                    continue

                self.addConstraint(
                    FunctionConstraint(distance_constraint), [
                        "x" + str(i),
                        "y" + str(i),
                        "x" + str(j),
                        "y" + str(j)
                    ])

        self._solver = MinConflictsSolver(assignments)

    def getSolution(self) -> Tuple[Region, ...]:
        solution = super(OreDistributionProblem, self).getSolution()
        regions = copy(self._regions)
        if solution is not None:
            k = 0
            i = 0
            for _ in solution:
                region = regions[i]
                if k == 0:
                    value = solution["x" + str(i)]
                    region.set_x(value - int(region.get_width() / 2))
                else:
                    value = solution["y" + str(i)]
                    region.set_y(value - int(region.get_height() / 2))
                if k == 1:
                    i += 1
                    k = 0
                else:
                    k += 1
            return regions
        else:
            raise Exception("No solution was found")


def cerp(v0: float, v1: float, t: float):
    mu2 = (1 - math.cos(t * math.pi)) / 2
    return v0 * (1 - mu2) + v1 * mu2


def lerp(v0: float, v1: float, t: float):
    return v0 + t * (v1 - v0)


class NonConstantPerlin1D(tuple):

    def __new__(cls, fq: int = 10):
        activations = [random.randint(0, 1) for _ in range(fq)]

        # clear multiple ones
        one = True
        for i, bit in enumerate(activations):
            if one and bit:
                activations[i] = 0
                continue

            one = bit
        gradients = [random.random() if bit else 0 for bit in activations]
        return tuple.__new__(NonConstantPerlin1D, gradients)


class LevelNoise(tuple):

    # TODO add brownian motion
    def __new__(cls, config: Config):
        w, h = config.get_width(), config.get_height()
        h1, h2, h3, l0, l1, l2, l3 = config.get_surface_params()

        # TODO zero pair shouldn't be always after level
        # TODO cumulative count should be exactly width
        def generate_levels(
                left_offset: float,
                right_offset: float,
                space_fill: float = 1.0
        ) -> Tuple[Tuple[int, int], ...]:
            """
            :param left_offset: how match space from left of full width should be set to level 0
            :param right_offset: how match space from right of full width should be set to level 0
            :param space_fill: how match should inner levels left space and fill zeros between levels
            """
            levels = [1] * h1 + [2] * h2 + [3] * h3 + [0] * l0 + [-1] * l1 + [-2] * l2 + [-3] * l3
            random.shuffle(levels)

            offset = left_offset + right_offset
            max_level_width = int((1 - offset) * w / (h1 + h2 + h3 + l0 + l1 + l2 + l3))
            min_level_width = int(space_fill * max_level_width)
            inner_levels = []

            while len(levels) > 0:
                level = levels.pop()
                count = random.randint(min_level_width, max_level_width)
                lc_pair = (level, count)
                inner_levels.append(lc_pair)
                if max_level_width - count > 0:
                    zero_pair = (0, max_level_width - count)
                    inner_levels.append(zero_pair)

            return ((0, int(left_offset * w)),) + tuple(inner_levels) + ((0, int(right_offset * w)),)

        def level_transform(level: int) -> float:
            if level == 0:
                return 0.5
            elif level == 1:
                return 0.5 + 0.5 / 3
            elif level == 2:
                return 0.5 + 2 * 0.5 / 3
            elif level == 3:
                return 1
            elif level == -1:
                return 0.5 - 0.5 / 3
            elif level == -2:
                return 0.5 - 2 * 0.5 / 3
            elif level == -3:
                return 0
            else:
                raise Exception

        def interval_transform(
                seq: Tuple[Tuple[int, int], ...],
                fade_min: float = 0.1,
                fade_max: float = 0.4
        ) -> Tuple[Tuple[int, float], ...]:
            """
            Transforms sequence into sequence of points
            :param seq: sequence of pairs (level, count)
            :param fade_min: minimal percentage of width for fade effect
            :param fade_max: maximal percentage of width for fade effect
            """
            i = 0
            cumulative_x = 0
            intervals = []
            for level0, count in seq:
                if i + 1 < len(seq):
                    level1, _ = seq[i + 1]

                    level0 = level_transform(level0)
                    level1 = level_transform(level1)
                    fade_out = random.randint(int(fade_min * count), int(fade_max * count))

                    if len(intervals) == 0:
                        intervals.append((cumulative_x, level0))
                    intervals.append((cumulative_x + (count - fade_out), level0))
                    intervals.append((cumulative_x + count, level1))

                    cumulative_x += count
                else:
                    level0 = level_transform(level0)
                    intervals.append((cumulative_x + count, level0))
                i += 1

            return tuple(intervals)

        def interpolation_transform(seq: Tuple[Tuple[int, float]]) -> Tuple[float, ...]:
            """
            Transform sequence of points into sequence of y in range 0:1
            :param seq: sequence of points
            """

            i = 0
            ys = []
            for x0, y0 in seq:
                if i + 1 < len(seq):
                    x1, y1 = seq[i + 1]
                    for x in range(x0, x1):
                        ys.append(cerp(y0, y1, (x - x0) / (x1 - x0)))
                else:
                    pass
                i += 1
            return tuple(ys)

        result0 = generate_levels(.1, .1, 1)
        result1 = interval_transform(result0, 0.15, 0.35)
        result2 = interpolation_transform(result1)
        return tuple.__new__(NonConstantPerlin1D, (result0, result1, result2))


class SurfaceGenerator(RegionGenerator):
    """ Implements CSP and generates caves """

    def __init__(self, config: Config):
        self.config = config

    def generate(self):
        area = HorizontalAreaGroup((HorizontalAreas.Surface,))
        x, y = area.get_x(), area.get_y()
        w, h = area.get_width(), area.get_height()

        noise0, noise1, noise2 = LevelNoise(self.config)

        regions = []
        cumulated_x = 0
        for _, count in noise0:
            region = SurfaceRegion(cumulated_x, y, 1, 1, Material.BACKGROUND)
            region.generate(noise2[cumulated_x:cumulated_x + count])
            region.update_dimensions()
            regions.append(region)

            cumulated_x += count
        return regions


class CaveGenerator(RegionGenerator):
    """ Implements CSP and generates caves """

    def __init__(self, config: Config):
        self.config = config

    def generate(self, count: int = 1, size_min: int = 100, size_max: int = 1000):
        area = HorizontalAreaGroup((HorizontalAreas.Underground, HorizontalAreas.Cavern))
        regions = []

        for i in range(0, count):
            cave_region = CaveRegion(0, 0, 1, 1, Material.BASE)
            cave_region.generate(random.randint(size_min, size_max))
            regions.append(cave_region)

        problem = OreDistributionProblem(area, regions)
        return problem.getSolution()


class Scene(ABC):
    """ Generation scene """
    config: Config
    drawer: Drawer

    @abstractmethod
    def play(self):
        pass


class MainScene(Scene):

    def __init__(self, config: Config):
        self.config = config
        self.drawer = PygameDrawer(cfg)

        HorizontalAreas.init(config)
        PixelMapping.init(config)

    def draw_regions(self, regions: Tuple[Region, ...]):
        for region in regions:
            self.drawer.draw_region(region)
            if self.config.is_debug():
                self.drawer.draw_outline(region)

    def play(self):
        self.drawer.init()

        self.drawer.draw_progress("Init ...")
        self.drawer.fill((0, 0, 0))
        self.drawer.draw_area(HorizontalAreas.Space)
        self.drawer.draw_rect(HorizontalAreas.Surface)
        self.drawer.draw_area(HorizontalAreas.Underground)
        self.drawer.draw_area(HorizontalAreas.Cavern)
        self.drawer.draw_area(HorizontalAreas.Underworld)

        if self.config.is_debug():
            self.drawer.draw_outline(HorizontalAreas.Space)
            self.drawer.draw_outline(HorizontalAreas.Surface)
            self.drawer.draw_outline(HorizontalAreas.Underground)
            self.drawer.draw_outline(HorizontalAreas.Cavern)
            self.drawer.draw_outline(HorizontalAreas.Underworld)

        self.drawer.draw_progress("Generating caves ...")
        cave_generator = CaveGenerator(self.config)
        regions = cave_generator.generate(50, 500, 2000)
        self.draw_regions(regions)

        self.drawer.draw_progress("Generating ores ...")
        ore_generator = CaveGenerator(self.config)
        regions = ore_generator.generate(250, 10, 50)
        self.draw_regions(regions)

        self.drawer.draw_progress("Generating surface ...")
        surface_generator = SurfaceGenerator(self.config)
        regions = surface_generator.generate()
        self.draw_regions(regions)

        self.drawer.draw_progress("Done ...")

        self.drawer.free()


if __name__ == '__main__':
    cfg = Config()
    scene = MainScene(cfg)
    scene.play()
