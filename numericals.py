# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Numerical representation of geometry."""
from __future__ import annotations

import math
from typing import Any, Optional, Union
import geometry as gm
import numpy as np
from numpy.random import uniform as unif  # pylint: disable=g-importing-member


ATOM = 1e-12


# Some variables are there for better code reading.
# pylint: disable=unused-assignment
# pylint: disable=unused-argument
# pylint: disable=unused-variable

# Naming in geometry is a little different
# we stick to geometry naming to better read the code.
# pylint: disable=invalid-name


class Point:
  """Numerical point."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __lt__(self, other: Point) -> bool:
    return (self.x, self.y) < (other.x, other.y)

  def __gt__(self, other: Point) -> bool:
    return (self.x, self.y) > (other.x, other.y)

  def __add__(self, p: Point) -> Point:
    return Point(self.x + p.x, self.y + p.y)

  def __sub__(self, p: Point) -> Point:
    return Point(self.x - p.x, self.y - p.y)

  def __mul__(self, f: float) -> Point:
    return Point(self.x * f, self.y * f)

  def __rmul__(self, f: float) -> Point:
    return self * f

  def __truediv__(self, f: float) -> Point:
    return Point(self.x / f, self.y / f)

  def __floordiv__(self, f: float) -> Point:
    div = self / f  # true div
    return Point(int(div.x), int(div.y))

  def __str__(self) -> str:
    return 'P({},{})'.format(self.x, self.y)

  def close(self, point: Point, tol: float = 1e-12) -> bool:
    return abs(self.x - point.x) < tol and abs(self.y - point.y) < tol

  def midpoint(self, p: Point) -> Point:
    return Point(0.5 * (self.x + p.x), 0.5 * (self.y + p.y))

  def distance(self, p: Union[Point, Line, Circle]) -> float:
    if isinstance(p, Line):
      return p.distance(self)
    if isinstance(p, Circle):
      return abs(p.radius - self.distance(p.center))
    dx = self.x - p.x
    dy = self.y - p.y
    return np.sqrt(dx * dx + dy * dy)

  def dot(self, other: Point) -> float:
    return self.x * other.x + self.y * other.y
  
  def parallel_line(self, line: Line) -> Line: 
    return line.parallel_line(self)


class Line:
  """Numerical line."""

  def __init__(
      self,
      p1: Point = None,
      p2: Point = None,
      coefficients: tuple[int, int, int] = None,
  ):
    if p1 is None and p2 is None and coefficients is None:
      self.coefficients = None, None, None
      return

    a, b, c = coefficients or (
        p1.y - p2.y,
        p2.x - p1.x,
        p1.x * p2.y - p2.x * p1.y,
    )

    # Make sure a is always positive (or always negative for that matter)
    # With a == 0, Assuming a = +epsilon > 0
    # Then b such that ax + by = 0 with y>0 should be negative.
    if a < 0.0 or a == 0.0 and b > 0.0:
      a, b, c = -a, -b, -c

    self.coefficients = a, b, c

  def parallel_line(self, p: Point) -> Line: 
    a, b, _ = self.coefficients 
    return Line(coefficients=(a, b, -a * p.x - b * p.y))  # pylint: disable=invalid-unary-operand-type 


  def intersect(self, obj: Union[Line, Circle]) -> tuple[Point, ...]:
    if isinstance(obj, Line):
      return line_line_intersection(self, obj)


  def __call__(self, x: Point, y: Point = None) -> float:
    if isinstance(x, Point) and y is None:
      return self(x.x, x.y)
    a, b, c = self.coefficients
    return x * a + y * b + c


class InvalidLineIntersectError(Exception):
  pass


def _perpendicular_bisector(p1: Point, p2: Point) -> Line:
  midpoint = (p1 + p2) * 0.5
  return Line(midpoint, midpoint + Point(p2.y - p1.y, p1.x - p2.x))


class Circle:
  """Numerical circle."""

  def __init__(
      self,
      center: Optional[Point] = None,
      radius: Optional[float] = None,
      p1: Optional[Point] = None,
      p2: Optional[Point] = None,
      p3: Optional[Point] = None,
  ):
    if not center:
      if not (p1 and p2 and p3):
        self.center = self.radius = self.r2 = None
        return
        # raise ValueError('Circle without center need p1 p2 p3')

      l12 = _perpendicular_bisector(p1, p2)
      l23 = _perpendicular_bisector(p2, p3)
      center = line_line_intersection(l12, l23)

    self.center = center
    self.a, self.b = center.x, center.y

    if not radius:
      if not (p1 or p2 or p3):
        raise ValueError('Circle needs radius or p1 or p2 or p3')
      p = p1 or p2 or p3
      self.r2 = (self.a - p.x) ** 2 + (self.b - p.y) ** 2
      self.radius = math.sqrt(self.r2)
    else:
      self.radius = radius
      self.r2 = radius * radius

class InvalidQuadSolveError(Exception):
  pass



def line_line_intersection(l1: Line, l2: Line) -> Point:
  a1, b1, c1 = l1.coefficients
  a2, b2, c2 = l2.coefficients
  # a1x + b1y + c1 = 0
  # a2x + b2y + c2 = 0
  d = a1 * b2 - a2 * b1
  if d == 0:
    raise InvalidLineIntersectError
  return Point((c2 * b1 - c1 * b2) / d, (c1 * a2 - c2 * a1) / d)


def check_too_close(
    newpoints: list[Point], points: list[Point], tol: int = 0.1
) -> bool:
  if not points:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  mindist = min([p.distance(avg) for p in points])
  for p0 in newpoints:
    for p1 in points:
      if p0.distance(p1) < tol * mindist:
        return True
  return False


def check_too_far(
    newpoints: list[Point], points: list[Point], tol: int = 4
) -> bool:
  if len(points) < 2:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  maxdist = max([p.distance(avg) for p in points])
  for p in newpoints:
    if p.distance(avg) > maxdist * tol:
      return True
  return False




def check(name: str, args: list[Union[gm.Point, Point]]) -> bool:
  """Numerical check."""
  if name == 'on_line':
    name = 'coll'
  elif name in ['rcompute', 'acompute']:
    return True
  elif name in ['fixl', 'fixc', 'fixb', 'fixt', 'fixp']:
    return True
  fn_name = 'check_' + name
  if fn_name not in globals():
    return None

  fun = globals()['check_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  return fun(args)

def check_coll(points: list[Point]) -> bool:
  a, b = points[:2]
  l = Line(a, b)
  for p in points[2:]:
    if abs(l(p.x, p.y)) > ATOM:
      return False
  return True


def check_ncoll(points: list[Point]) -> bool:
  return not check_coll(points)


def check_sameside(points: list[Point]) -> bool:
  b, a, c, y, x, z = points
  # whether b is to the same side of a & c as y is to x & z
  ba = b - a
  bc = b - c
  yx = y - x
  yz = y - z
  return ba.dot(bc) * yx.dot(yz) > 0



def same_clock(
    a: Point, b: Point, c: Point, d: Point, e: Point, f: Point
) -> bool:
  ba = b - a
  cb = c - b
  ed = e - d
  fe = f - e
  return (ba.x * cb.y - ba.y * cb.x) * (ed.x * fe.y - ed.y * fe.x) > 0



def check_cong(points: list[Point]) -> bool:
  a, b, c, d = points
  return close_enough(a.distance(b), c.distance(d))


def close_enough(a: float, b: float, tol: float = 1e-12) -> bool:
  return abs(a - b) < tol


def assert_close_enough(a: float, b: float, tol: float = 1e-12) -> None:
  assert close_enough(a, b, tol), f'|{a}-{b}| = {abs(a-b)} >= {tol}'


def ang_of(tail: Point, head: Point) -> float:
  vector = head - tail
  arctan = np.arctan2(vector.y, vector.x) % (2 * np.pi)
  return arctan


def ang_between(tail: Point, head1: Point, head2: Point) -> float:
  ang1 = ang_of(tail, head1)
  ang2 = ang_of(tail, head2)
  diff = ang1 - ang2
  # return diff % (2*np.pi)
  if diff > np.pi:
    return diff - 2 * np.pi
  if diff < -np.pi:
    return 2 * np.pi + diff
  return diff


def head_from(tail: Point, ang: float, length: float = 1) -> Point:
  vector = Point(np.cos(ang) * length, np.sin(ang) * length)
  return tail + vector


def random_points(n: int = 3) -> list[Point]:
  return [Point(unif(-1, 1), unif(-1, 1)) for _ in range(n)]


def random_rfss(*points: list[Point]) -> list[Point]:
  """Random rotate-flip-scale-shift a point cloud."""
  # center point cloud.
  average = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
  points = [p - average for p in points]

  # rotate
  ang = unif(0.0, 2 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  # scale and shift
  scale = unif(0.5, 2.0)
  shift = Point(unif(-1, 1), unif(-1, 1))
  points = [p.rotate(sin, cos) * scale + shift for p in points]

  # randomly flip
  if np.random.rand() < 0.5:
    points = [p.flip() for p in points]

  return points


def reduce(
    objs: list[Union[Point, Line, Circle]],
    existing_points: list[Point],
) -> list[Point]:
  """Reduce intersecting objects into one point of intersections."""
  if all(isinstance(o, Point) for o in objs):
    return objs

  # elif len(objs) == 1:
  #   return objs[0].sample_within(existing_points)

  elif len(objs) == 2:
    a, b = objs
    result = a.intersect(b)
    if isinstance(result, Point):
      return [result]
    a, b = result
    a_close = any([a.close(x) for x in existing_points])
    if a_close:
      return [b]
    b_close = any([b.close(x) for x in existing_points])
    if b_close:
      return [a]
    return [np.random.choice([a, b])]

  else:
    raise ValueError(f'Cannot reduce {objs}')


def sketch(
    name: str, args: list[Union[Point, gm.Point]]
) -> list[Union[Point, Line, Circle, HalfLine, HoleCircle]]:
  print(f'Sketching {name} with args {args}')
  fun = globals()['sketch_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  out = fun(args)

  # out can be one or multiple {Point/Line/HalfLine}
  if isinstance(out, (tuple, list)):
    return list(out)
  return [out]


def sketch_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  ac = unif(0.5, 2.0)
  ang = unif(0.2, 0.8) * np.pi
  c = head_from(a, ang, ac)
  return a, b, c


def sketch_midp(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  return (a + b) * 0.5

def sketch_pline(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return a.parallel_line(Line(b, c))
