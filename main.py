import math
import random
import string
import time
from abc import abstractmethod, ABC
from copy import copy
from typing import Tuple, List, Union

import pygame
from PIL import Image, ImageDraw

from csp import MinConflictsSolver, Problem, FunctionConstraint

WIDTH = 800
HEIGHT = 600


############################### ABCS #####################################

class Config(dict):

    def __init__(self):
        super(Config, self).__init__()
        self['WORLD_WIDTH'] = WIDTH
        self['WORLD_HEIGHT'] = HEIGHT

    def get_width(self):
        return self['WORLD_WIDTH']

    def get_height(self):
        return self['WORLD_HEIGHT']


class Area(ABC):
    """ Represents rectangular area of canvas """

    x: int
    y: int
    w: int
    h: int
    color: Tuple[int, int, int] = (255, 0, 0)

    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

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

    def is_inside(self, area: 'Area'):
        x, y, w, h = area.get_rect()
        return self.x < x and self.x + self.w > x + w and self.y < y and self.y + self.h > y + h

    def set_color(self, color: Tuple[int, int, int]):
        self.color = color


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
        return TranslatedArea(area.get_x(), self.y + self.h - area.get_y(), area.get_width(), area.get_height())


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


class Block(Area):
    """ Area with width and height of 1 """

    w: int = 1
    h: int = 1

    def __init__(self, x: int, y: int):
        super(Block, self).__init__(x, y, self.w, self.h)

    @classmethod
    def with_color(cls, x: int, y: int, color: Tuple[int, int, int]):
        block = Block(x, y)
        block.set_color(color)
        return block


class Generator(ABC):

    @abstractmethod
    def generate(self, **kwargs):
        pass


class Region(Generator, Area, ABC):
    """ Represents region of canvas, with specific generation algorithm implemented """

    def __init__(self, x, y, w, h):
        super(Region, self).__init__(x, y, w, h)
        self.blocks = []

    def get_color(self):
        return 255, 0, 0

    def get_blocks(self):
        return self.blocks

    def set_x(self, x: int):
        for block in self.blocks:
            block.set_x(block.x - self.x + x)
        super().set_x(x)

    def set_y(self, y: int):
        for block in self.blocks:
            block.set_y(block.y - self.y + y)
        super().set_y(y)

    def update_dimensions(self):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0
        initialized = False

        for block in self.blocks:
            if not initialized:
                min_x = block.get_x()
                min_y = block.get_y()
                max_x = block.get_x()
                max_y = block.get_y()
                initialized = True
            else:
                if block.get_x() < min_x:
                    min_x = block.get_x()
                if block.get_x() > max_x:
                    max_x = block.get_x()
                if block.get_y() < min_y:
                    min_y = block.get_y()
                if block.get_y() > max_y:
                    max_y = block.get_y()

        self.x = min_x
        self.y = min_y
        self.w = max_x - min_x + 1
        self.h = max_y - min_y + 1


class RegionGenerator(Generator, ABC):

    @abstractmethod
    def generate(self, **kwargs) -> Tuple[Region, ...]:
        pass


class Drawer(ABC):

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def free(self):
        pass

    @abstractmethod
    def draw_rect(self, area: Area):
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
        self.canvas.save("img/" + name + ".jpg")
        self.canvas.show()

    def draw_rect(self, area: Area):
        x, y, x1, y1 = area.get_points()
        color = area.get_color()
        self.drawer.rectangle((x, y, x1, y1), color)

    def draw_region(self, region: Region):
        for block in region.get_blocks():
            self.draw_rect(block)

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
        pygame.image.save(self.canvas, "img/" + name + ".jpg")
        while True:
            time.sleep(0.016)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    pygame.font.quit()
                    exit(0)

    def draw_rect(self, area: Area):
        x, y, x1, y1 = area.get_rect()
        color = area.get_color()
        pygame.draw.rect(self.canvas, color, (x, y, x1, y1))
        pygame.display.update((x, y, x1, y1))
        self._handle_events()

    def draw_region(self, region: Region):
        for block in region.get_blocks():
            x, y, x1, y1 = block.get_rect()
            color = block.get_color()
            pygame.draw.rect(self.canvas, color, (x, y, x1, y1))
            self._handle_events()
        x, y, x1, y1 = region.get_rect()
        pygame.display.update((x, y, x1, y1))

    def draw_outline(self, region: Region):
        x, y, x1, y1 = region.get_rect()
        color = region.get_color()
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


class Layer(Generator):
    """ Layer of canvas """

    Space: HorizontalArea
    Surface: HorizontalArea
    Underground: HorizontalArea
    Cavern: HorizontalArea
    Underworld: HorizontalArea

    def __init__(self, drawer: Drawer, config: Config):
        self.drawer = drawer
        self.config = config
        self._regions = []

    def add_region(self, region: Region):
        self._regions.append(region)

    def remove_region(self, region: Region):
        self._regions.remove(region)

    def extend_regions(self, regions: Union[Tuple[Region, ...], List[Region]]):
        for region in regions:
            self.add_region(region)

    def get_regions(self):
        return self._regions

    @abstractmethod
    def generate(self):
        pass


############################### IMPLEMENTATION #####################################


class OreRegion(Region, ABC):

    def generate(self, count: int = 100):
        """ Generates ore in stochastic fashion """
        if count > 0:
            i = 0
            coords = {}
            blocks = [Block.with_color(self.x, self.y, self.color)]
            sides = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            while i < count:
                # TODO should be more deterministic
                _ = int(i - i * 0.07)
                if (i - _) < 10:  # TODO can get stuck here
                    _ = 0
                current = blocks[random.randint(_, i)]
                x0, y0 = sides[random.randint(0, 3)]
                x1 = current.get_x() + x0
                y1 = current.get_y() + y0

                if not coords.get((x1, y1), False):
                    coords[(x1, y1)] = True
                    blocks.append(Block.with_color(x1, y1, self.color))
                    i += 1
            self.blocks.extend(blocks)
            self.update_dimensions()


class CopperRegion(OreRegion):
    color = (148, 68, 28)


class GoldRegion(OreRegion):
    color = (183, 162, 29)


class SilverRegion(OreRegion):
    color = (215, 222, 222)


class IronRegion(OreRegion):
    color = (127, 127, 127)


class CaveRegion(Region, ABC):

    def generate(self, count: int = 100):
        """ Generates caves """
        if count > 0:
            i = 0
            coords = {}
            blocks = [Block.with_color(self.x, self.y, self.color)]
            sides = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            while i < count:
                # TODO should be more deterministic
                _ = int(i - i * 0.04)
                if (i - _) < 10:  # TODO can get stuck here
                    _ = 0
                current = blocks[random.randint(_, i)]
                x0, y0 = sides[random.randint(0, 3)]
                x1 = current.get_x() + x0
                y1 = current.get_y() + y0

                if not coords.get((x1, y1), False):
                    coords[(x1, y1)] = True
                    blocks.append(Block.with_color(x1, y1, self.color))
                    i += 1
            self.blocks.extend(blocks)
            self.update_dimensions()


class DirtCaveRegion(CaveRegion):
    color = (88, 63, 50)


class Space(HorizontalArea):
    color = (8, 0, 60)

    def __init__(self):
        super(Space, self).__init__(0, 1 * HEIGHT / 20)


class Surface(HorizontalArea):
    color = (155, 209, 254)

    def __init__(self):
        super(Surface, self).__init__(1 * HEIGHT / 20, 4 * HEIGHT / 20)


class Underground(HorizontalArea):
    color = (150, 107, 76)

    def __init__(self):
        super(Underground, self).__init__(4 * HEIGHT / 20, 7 * HEIGHT / 20)


class Cavern(HorizontalArea):
    color = (127, 127, 127)

    def __init__(self):
        super(Cavern, self).__init__(7 * HEIGHT / 20, 17 * HEIGHT / 20)


class Underworld(HorizontalArea):
    color = (0, 0, 0)

    def __init__(self):
        super(Underworld, self).__init__(17 * HEIGHT / 20, HEIGHT)


############################### CSP ################################################

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


############################## NOISE ###############################################

class NonConstantPerlin1D(tuple):

    def __new__(cls, points: int = 10):
        activations = [random.randint(0, 1) for _ in range(points)]

        # clear multiple ones
        one = True
        for i, bit in enumerate(activations):
            if one and bit:
                activations[i] = 0
                continue

            one = bit
        gradients = [random.random() if bit else 0 for bit in activations]
        return tuple.__new__(NonConstantPerlin1D, gradients)


############################## GENERATORS #########################################
def clamp(x: float, lowerlimit: float, upperlimit: float) -> float:
    if x < lowerlimit:
        return lowerlimit
    elif x > upperlimit:
        return upperlimit
    else:
        return x


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (3 * x ** 2) - (2 * x ** 3)


def lerp(v0: float, v1: float, t: float) -> float:
    return v0 + t * (v1 - v0)


class SurfaceGenerator(RegionGenerator):
    """ Implements CSP and generates caves """

    def __init__(self, config: Config, layer: Layer):
        self.config = config
        self.layer = layer

    def generate(self, fq: int):
        area = HorizontalAreaGroup((self.layer.Surface,))
        x, y = area.get_x(), area.get_y()
        w, h = area.get_width(), area.get_height()

        noise = NonConstantPerlin1D(fq)
        points = [(0, y + h + 1)]
        b_eval = (y + h) - (h / 4)
        for _x, _y in enumerate(noise):
            if _y > 0:
                b_eval = (y + h) - (h / 4) - (_y * h * .5)
            points.append((_x * w / fq, b_eval))

        points.append((w, (y + h) - (h / 4)))
        points.append((w, y + h + 1))

        return points


class CaveGenerator(RegionGenerator):
    """ Implements CSP and generates caves """

    def __init__(self, config: Config, layer: Layer):
        self.config = config
        self.layer = layer

    def generate(self, count: int = 1, size_min: int = 100, size_max: int = 1000):
        area = HorizontalAreaGroup((self.layer.Underground, self.layer.Cavern))
        regions = []

        for i in range(0, count):
            cave_region = DirtCaveRegion(0, 0, 1, 1)
            cave_region.generate(random.randint(size_min, size_max))
            regions.append(cave_region)

        problem = OreDistributionProblem(area, regions)
        return problem.getSolution()


# static variables
setattr(Layer, "Space", Space())
setattr(Layer, "Surface", Surface())
setattr(Layer, "Underground", Underground())
setattr(Layer, "Cavern", Cavern())
setattr(Layer, "Underworld", Underworld())


class Layer0(Layer):

    def generate(self):
        self.drawer.draw_progress("Generating Layer0 ...")

        self.drawer.fill((0, 0, 0))
        self.drawer.draw_rect(self.Space)
        self.drawer.draw_rect(self.Surface)
        self.drawer.draw_rect(self.Underground)
        self.drawer.draw_rect(self.Cavern)
        self.drawer.draw_rect(self.Underworld)

        self.drawer.draw_progress("Generating caves ...")
        cave_generator = CaveGenerator(self.config, self)
        regions = cave_generator.generate(100, 10, 50)
        self.extend_regions(regions)
        self.drawer.draw_progress("Done ...")

        # TODO CSP

    def add_region(self, region: Region):
        super(Layer0, self).add_region(region)
        self.drawer.draw_region(region)
        self.drawer.draw_outline(region)


class Layer1(Layer):

    def generate(self):
        self.drawer.draw_progress("Generating Layer1 ...")
        self.drawer.draw_progress("Generating surface ...")
        surface_generator = SurfaceGenerator(self.config, self)
        points = surface_generator.generate(40)
        self.drawer.draw_polygon(points, (150, 107, 76))
        self.drawer.draw_progress("Done ...")

        # TODO CSP

    def add_region(self, region: Region):
        super(Layer1, self).add_region(region)
        self.drawer.draw_region(region)
        self.drawer.draw_outline(region)


if __name__ == '__main__':
    cfg = Config()
    draw = PygameDrawer(cfg)
    draw.init()

    layer0 = Layer0(draw, cfg)
    layer0.generate()
    layer1 = Layer1(draw, cfg)
    layer1.generate()

    draw.free()
