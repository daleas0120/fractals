from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display
from PIL import Image
from dataclasses import dataclass
from math import log

np.warnings.filterwarnings("ignore")

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c:complex):
        z = 0
        for _ in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > 2:
                return False
        return True

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def stability(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.escape_count(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def escape_count(self, c: complex, smooth=False) -> Union[int, float]:
        z = 0
        for iteration in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iteration + 1 - log(log(abs(z))) / log(2)
                return iteration
        return self.max_iterations


@dataclass
class Viewport:
    image: Image.Image
    center: complex
    width: float

    @property
    def height(self):
        return self.scale * self.image.height

    @property
    def offset(self):
        return self.center + complex(-self.width, self.height) / 2

    @property
    def scale(self):
        return self.width / self.image.width

    def __iter__(self):
        for y in range(self.image.height):
            for x in range(self.image.width):
                yield Pixel(self, x, y)



@dataclass
class Pixel:
    viewport: Viewport
    x: int
    y: int

    @property
    def color(self):
        return self.viewport.image.getpixel((self.x, self.y))

    @color.setter
    def color(self, value):
        self.viewport.image.putpixel((self.x, self.y), value)

    def __complex__(self):
        return (
                complex(self.x, -self.y) * self.viewport.scale + self.viewport.offset
        )


def sequence(c, z=0):
    while True:
        yield z
        z = z**2 + c


def get_sequence(num_terms, complex_constant, starting_point=0):
    series = []
    for n, z in enumerate(sequence(c=complex_constant, z=starting_point)):
        if n>num_terms:
            break
        series[n] = z
    return series


def mandelbrot(candidate):
    return sequence(z=0, c=candidate)


def julia(candidate, parameter):
    return sequence(z=candidate, c=parameter)


def complex_matrix(x_min, x_max, y_min, y_max, pixel_density):
    re = np.linspace(x_min, x_max, int((x_max - x_min) * pixel_density))
    im = np.linspace(y_min, y_max, int((y_max - y_min) * pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def is_stable(c, max_iterations):
    z = 0
    for _ in range(max_iterations):
        z = z**2 + c
        if abs(z) > 2:
            return False
    return True

def get_search_area(width, height, center_pt=None, scale=0.002):

    if center_pt is None:
        center_pt = 0

    image = Image.new(mode="RGB", size=(width, height))
    viewport = Viewport(image, center=center_pt, width=scale)
    return viewport, image

def get_fractal(mandelbrot_set, viewport, image, center_pt=None):
    colormap = plt.cm.get_cmap("twilight").colors

    def denormalize(palette):
        return [tuple(int(channel * 255) for channel in color) for color in palette]

    palette = denormalize(colormap)

    def paint(mandelbrot_set, viewport, palette, smooth):
        for pixel in viewport:
            stability = mandelbrot_set.stability(complex(pixel), smooth)
            index = int(min(stability * len(palette), len(palette) - 1))
            pixel.color = palette[index % len(palette)]

    paint(mandelbrot_set, viewport, palette, smooth=True)

    display(image)
    image.save(str(center_pt)+".png", format="png")


