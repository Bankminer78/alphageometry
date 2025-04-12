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

"""Implements the graph representation of the proof state."""

# pylint: disable=g-multiple-import
from __future__ import annotations

from collections import defaultdict  # pylint: disable=g-importing-member
from typing import Callable, Generator, Optional, Type, Union

from absl import logging
import ar
import geometry as gm
from geometry import Angle, Direction, Length, Ratio
from geometry import Circle, Line, Point, Segment
from geometry import Measure, Value
import graph_utils as utils
import numericals as nm
import problem
from problem import Dependency, EmptyDependency


np = nm.np


FREE = [
    'free',
    'segment',
    'r_triangle',
    'risos',
    'triangle',
    'triangle12',
    'ieq_triangle',
    'eq_quadrangle',
    'eq_trapezoid',
    'eqdia_quadrangle',
    'quadrangle',
    'r_trapezoid',
    'rectangle',
    'isquare',
    'trapezoid',
    'pentagon',
    'iso_triangle',
]

INTERSECT = [
    'angle_bisector',
    'angle_mirror',
    'eqdistance',
    'lc_tangent',
    'on_aline',
    'on_bline',
    'on_circle',
    'on_line',
    'on_pline',
    'on_tline',
    'on_dia',
    's_angle',
    'on_opline',
    'eqangle3',
]


# pylint: disable=protected-access
# pylint: disable=unused-argument


class DepCheckFailError(Exception):
  pass


class PointTooCloseError(Exception):
  pass


class PointTooFarError(Exception):
  pass


class Graph:
  """Graph data structure representing proof state."""

  def __init__(self):
    self.type2nodes = {
        Point: [],
        Line: [],
        Segment: [],
        Circle: [],
        Direction: [],
        Length: [],
        Angle: [],
        Ratio: [],
        Measure: [],
        Value: [],
    }
    self._name2point = {}
    self._name2node = {}

    self.rconst = {}  # contains all constant ratios
    self.aconst = {}  # contains all constant angles.

    self.halfpi, _ = self.get_or_create_const_ang(1, 2)
    self.vhalfpi = self.halfpi.val

    self.atable = ar.AngleTable()
    self.dtable = ar.DistanceTable()
    self.rtable = ar.RatioTable()

    # to quick access deps.
    self.cache = {}

    self._pair2line = {}
    self._triplet2circle = {}


  def _create_const_ang(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    ang = self.aconst[(n, d)] = self.new_node(Angle, f'{n}pi/{d}')
    ang.set_directions(None, None)
    self.connect_val(ang, deps=None)

  def get_or_create_const_ang(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    if (n, d) not in self.aconst:
      self._create_const_ang(n, d)
    ang1 = self.aconst[(n, d)]

    n, d = ar.simplify(d - n, d)
    if (n, d) not in self.aconst:
      self._create_const_ang(n, d)
    ang2 = self.aconst[(n, d)]
    return ang1, ang2
  
  @classmethod
  def build_problem(
      cls,
      pr: problem.Problem,
      definitions: dict[str, problem.Definition],
      verbose: bool = True,
      init_copy: bool = True,
  ) -> tuple[Graph, list[Dependency]]:
    """Build a problem into a gr.Graph object."""
    check = False
    g = None
    added = None
    if verbose:
      logging.info(pr.url)
      logging.info(pr.txt())
    while not check:
      try:
        g = Graph()
        added = []
        plevel = 0
        for clause in pr.clauses:
          adds, plevel = g.add_clause(
              clause, plevel, definitions, verbose=verbose
          )
          added += adds
        g.plevel = plevel

      except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
        continue
      except DepCheckFailError:
        continue
      except (PointTooCloseError, PointTooFarError):
        continue

      if not pr.goal:
        break

      args = list(map(lambda x: g.get(x, lambda: int(x)), pr.goal.args))
      check = nm.check(pr.goal.name, args)

    g.url = pr.url
    g.build_def = (pr, definitions)
    # for add in added:
    #   g.add_algebra(add, level=0)

    return g, added

  def all_points(self) -> list[Point]:
    """Return all nodes of type Point."""
    return list(self.type2nodes[Point])

  def all_nodes(self) -> list[gm.Node]:
    """Return all nodes."""
    return list(self._name2node.values())

  def names2nodes(self, pnames: list[str]) -> list[gm.Node]:
    return [self._name2node[name] for name in pnames]

  def names2points(
      self, pnames: list[str], create_new_point: bool = False
  ) -> list[Point]:
    """Return Point objects given names."""
    result = []
    for name in pnames:
      if name not in self._name2node and not create_new_point:
        raise ValueError(f'Cannot find point {name} in graph')
      elif name in self._name2node:
        obj = self._name2node[name]
      else:
        obj = self.new_node(Point, name)
      result.append(obj)

    return result

  def get(self, pointname: str, default_fn: Callable[str, Point]) -> Point:
    if pointname in self._name2point:
      return self._name2point[pointname]
    if pointname in self._name2node:
      return self._name2node[pointname]
    return default_fn()

  def new_node(self, oftype: Type[gm.Node], name: str = '') -> gm.Node:
    node = oftype(name, self)

    self.type2nodes[oftype].append(node)
    self._name2node[name] = node

    if isinstance(node, Point):
      self._name2point[name] = node

    return node

  def merge(self, nodes: list[gm.Node], deps: Dependency) -> gm.Node:
    """Merge all nodes."""
    if len(nodes) < 2:
      return

    node0, *nodes1 = nodes
    all_nodes = self.type2nodes[type(node0)]

    # find node0 that exists in all_nodes to be the rep
    # and merge all other nodes into node0
    for node in nodes:
      if node in all_nodes:
        node0 = node
        nodes1 = [n for n in nodes if n != node0]
        break
    return self.merge_into(node0, nodes1, deps)

  def merge_into(
      self, node0: gm.Node, nodes1: list[gm.Node], deps: Dependency
  ) -> gm.Node:
    """Merge nodes1 into a single node0."""
    node0.merge(nodes1, deps)
    for n in nodes1:
      if n.rep() != n:
        self.remove([n])

    nodes = [node0] + nodes1
    if any([node._val for node in nodes]):
      for node in nodes:
        self.connect_val(node, deps=None)

      vals1 = [n._val for n in nodes1]
      node0._val.merge(vals1, deps)

      for v in vals1:
        if v.rep() != v:
          self.remove([v])

    return node0

  def remove(self, nodes: list[gm.Node]) -> None:
    """Remove nodes out of self because they are merged."""
    if not nodes:
      return

    for node in nodes:
      all_nodes = self.type2nodes[type(nodes[0])]

      if node in all_nodes:
        all_nodes.remove(node)

      if node.name in self._name2node.values():
        self._name2node.pop(node.name)

  def connect(self, a: gm.Node, b: gm.Node, deps: Dependency) -> None:
    a.connect_to(b, deps)
    b.connect_to(a, deps)

  def connect_val(self, node: gm.Node, deps: Dependency) -> gm.Node:
    """Connect a node into its value (equality) node."""
    if node._val:
      return node._val
    name = None
    if isinstance(node, Line):
      name = 'd(' + node.name + ')'
    if isinstance(node, Angle):
      name = 'm(' + node.name + ')'
    if isinstance(node, Segment):
      name = 'l(' + node.name + ')'
    if isinstance(node, Ratio):
      name = 'r(' + node.name + ')'
    v = self.new_node(gm.val_type(node), name)
    self.connect(node, v, deps=deps)
    return v

  def is_equal(self, x: gm.Node, y: gm.Node, level: int = None) -> bool:
    return gm.is_equal(x, y, level)

  def add_piece(
      self, name: str, args: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new predicate."""
    if name in ['coll', 'collx']:
      return self.add_coll(args, deps)
    elif name == 'para':
      return self.add_para(args, deps)
    elif name == 'cong':
      return self.add_cong(args, deps)
    # elif name in ['eqangle', 'eqangle6']:
    #   return self.add_eqangle(args, deps)
    # elif name in ['eqratio', 'eqratio6']:
    #   return self.add_eqratio(args, deps)
    # numerical!
    # composite pieces:
    elif name == 'eqratio3':
      return self.add_eqratio3(args, deps)
    elif name == 'simtri':
      return self.add_simtri(args, deps)
    elif name == 'contri':
      return self.add_contri(args, deps)
    elif name == 'simtri*':
      return self.add_simtri_check(args, deps)
    elif name == 'contri*':
      return self.add_contri_check(args, deps)
    raise ValueError(f'Not recognize {name}')

  def check(self, name: str, args: list[Point]) -> bool:
    """Symbolically check if a predicate is True."""
    print(name)
    if name == 'ncoll':
      return self.check_ncoll(args)
    if name == 'cong':
      return self.check_cong(args)
    if name in 'diff':
      a, b = args
      return not a.num.close(b.num)
    raise ValueError(f'Not recognize {name}')

  def _get_line(self, a: Point, b: Point) -> Optional[Line]:
    linesa = a.neighbors(Line)
    for l in b.neighbors(Line):
      if l in linesa:
        return l
    return None

  def get_line_thru_pair(self, p1: Point, p2: Point) -> Line:
    if (p1, p2) in self._pair2line:
      return self._pair2line[(p1, p2)]
    if (p2, p1) in self._pair2line:
      return self._pair2line[(p2, p1)]
    return self.get_new_line_thru_pair(p1, p2)

  def get_new_line_thru_pair(self, p1: Point, p2: Point) -> Line:
    if p1.name.lower() > p2.name.lower():
      p1, p2 = p2, p1
    name = p1.name.lower() + p2.name.lower()
    line = self.new_node(Line, name)
    line.num = nm.Line(p1.num, p2.num)
    line.points = p1, p2

    self.connect(p1, line, deps=None)
    self.connect(p2, line, deps=None)
    self._pair2line[(p1, p2)] = line
    return line

  def get_line_thru_pair_why(
      self, p1: Point, p2: Point
  ) -> tuple[Line, list[Dependency]]:
    """Get one line thru two given points and the corresponding dependency list."""
    if p1.name.lower() > p2.name.lower():
      p1, p2 = p2, p1
    if (p1, p2) in self._pair2line:
      return self._pair2line[(p1, p2)].rep_and_why()

    l, why = gm.line_of_and_why([p1, p2])
    if l is None:
      l = self.get_new_line_thru_pair(p1, p2)
      why = []
    return l, why


  def add_coll(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a predicate that `points` are collinear."""
    points = list(set(points))
    og_points = list(points)

    all_lines = []
    for p1, p2 in utils.comb2(points):
      all_lines.append(self.get_line_thru_pair(p1, p2))
      
    points = sum([l.neighbors(Point) for l in all_lines], [])
    points = list(set(points))

    existed = set()
    new = set()
    for p1, p2 in utils.comb2(points):
      if p1.name > p2.name:
        p1, p2 = p2, p1
      if (p1, p2) in self._pair2line:
        line = self._pair2line[(p1, p2)]
        existed.add(line)
      else:
        line = self.get_new_line_thru_pair(p1, p2)
        new.add(line)

    existed = sorted(existed, key=lambda l: l.name)
    new = sorted(new, key=lambda l: l.name)

    existed, new = list(existed), list(new)
    if not existed:
      line0, *lines = new
    else:
      line0, lines = existed[0], existed[1:] + new

    add = []
    line0, why0 = line0.rep_and_why()
    a, b = line0.points
    for line in lines:
      c, d = line.points
      args = list({a, b, c, d})
      if len(args) < 3:
        continue

      whys = []
      for x in args:
        if x not in og_points:
          whys.append(self.coll_dep(og_points, x))

      abcd_deps = deps
      if whys + why0:
        dep0 = deps.populate('coll', og_points)
        abcd_deps = EmptyDependency(level=deps.level, rule_name=None)
        abcd_deps.why = [dep0] + whys

      is_coll = self.check_coll(args)
      dep = abcd_deps.populate('coll', args)
      self.cache_dep('coll', args, dep)
      self.merge_into(line0, [line], dep)

      if not is_coll:
        add += [dep]

    return add
  
  def coll_dep(self, points: list[Point], p: Point) -> list[Dependency]:
    """Return the dep(.why) explaining why p is coll with points."""
    for p1, p2 in utils.comb2(points):
      if self.check_coll([p1, p2, p]):
        dep = Dependency('coll', [p1, p2, p], None, None)
        return dep.why_me_or_cache(self, None)

  def check_coll(self, points: list[Point]) -> bool:
    print('Checking coll')
    points = list(set(points))
    if len(points) < 3:
      return True
    line2count = defaultdict(lambda: 0)
    for p in points:
      #print('finding neighbors')
      for l in p.neighbors(Line):
        line2count[l] += 1
    return any([count == len(points) for _, count in line2count.items()])

  def check_ncoll(self, points: list[Point]) -> bool:
    if self.check_coll(points):
      return False
    return not nm.check_coll([p.num for p in points])

  def check_sameside(self, points: list[Point]) -> bool:
    return nm.check_sameside([p.num for p in points])

  def make_equal(self, x: gm.Node, y: gm.Node, deps: Dependency) -> None:
      """Make that two nodes x and y are equal, i.e. merge their value node."""
      #print("[make_equal] === MAKING NODES EQUAL ===")
      if x.val is None:
          #print("[make_equal] x.val is None, swapping x and y")
          x, y = y, x

      #print("[make_equal] Connecting value nodes for x and y")
      self.connect_val(x, deps=None)
      self.connect_val(y, deps=None)
      vx = x._val
      vy = y._val

      if vx == vy:
         # print("[make_equal] Value nodes are already equal; exiting merge.")
          return

      merges = [vx, vy]
      #print(f"[make_equal] Prepared merges list with {len(merges)} nodes.")

      if (isinstance(x, Angle) and 
          x not in self.aconst.values() and 
          y not in self.aconst.values() and 
          x.directions == y.directions[::-1] and 
          x.directions[0] != x.directions[1]):
          #print("[make_equal] Reversed directions detected in Angles. Adjusting merge list.")
          merges = [self.vhalfpi, vx, vy]

      #print(f"[make_equal] Merging {len(merges)} nodes with dependency {deps}")
      self.merge(merges, deps)
      #print("[make_equal] Merge complete.")

  def why_equal(self, x: gm.Node, y: gm.Node, level: int) -> list[Dependency]:
    return gm.why_equal(x, y, level)

  def add_para(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new predicate that 4 points (2 lines) are parallel."""
    a, b, c, d = points
    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)

    is_equal = self.is_equal(ab, cd)

    (a, b), (c, d) = ab.points, cd.points

    dep0 = deps.populate('para', points)
    deps = EmptyDependency(level=deps.level, rule_name=None)

    deps = deps.populate('para', [a, b, c, d])
    deps.why = [dep0] + why1 + why2

    self.make_equal(ab, cd, deps)
    deps.algebra = ab._val, cd._val

    self.cache_dep('para', [a, b, c, d], deps)
    if not is_equal:
      return [deps]
    return []

  def _get_angle(
      self, d1: Direction, d2: Direction
  ) -> tuple[Angle, Optional[Angle]]:
    for a in self.type2nodes[Angle]:
      if a.directions == (d1, d2):
        return a, a.opposite
    return None, None

  def _get_or_create_angle(
      self, l1: Line, l2: Line, deps: Dependency
  ) -> tuple[Angle, Angle, list[Dependency]]:
    return self.get_or_create_angle_d(l1._val, l2._val, deps)

  def check_perpl(self, ab: Line, cd: Line) -> bool:
    if ab.val is None or cd.val is None:
      return False
    if ab.val == cd.val:
      return False
    a12, a21 = self._get_angle(ab.val, cd.val)
    if a12 is None or a21 is None:
      return False
    return self.is_equal(a12, a21)

  def check_perp(self, points: list[Point]) -> bool:
    a, b, c, d = points
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    if not ab or not cd:
      return False
    return self.check_perpl(ab, cd)

  def _get_segment(self, p1: Point, p2: Point) -> Optional[Segment]:
    for s in self.type2nodes[Segment]:
      if s.points == {p1, p2}:
        return s
    return None

  def _get_or_create_segment(
      self, p1: Point, p2: Point, deps: Dependency
  ) -> Segment:
    """Get or create a Segment object between two Points p1 and p2."""
    if p1 == p2:
      raise ValueError(f'Creating same 0-length segment {p1.name}')

    for s in self.type2nodes[Segment]:
      if s.points == {p1, p2}:
        return s

    if p1.name > p2.name:
      p1, p2 = p2, p1
    s = self.new_node(Segment, name=f'{p1.name.upper()}{p2.name.upper()}')
    self.connect(p1, s, deps=deps)
    self.connect(p2, s, deps=deps)
    s.points = {p1, p2}
    return s

  def add_cong(self, points: list[Point], deps: EmptyDependency) -> list[Dependency]:
      """Add that two segments (4 points) are congruent."""
      print('==== ADDING CONGRUENCE =======')
      if len(points) != 4:
          raise ValueError("add_cong expects exactly 4 points")
      
      a, b, c, d = points
      print(f"[add_cong] Points: {a.name}, {b.name}, {c.name}, {d.name}")
      
      ab = self._get_or_create_segment(a, b, deps=None)
      cd = self._get_or_create_segment(c, d, deps=None)
      
      is_equal = self.is_equal(ab, cd)
      print(f"[add_cong] Segments {ab.name} and {cd.name} are {'equal' if is_equal else 'not equal'}")
      
      dep = deps.populate('cong', [a, b, c, d])
      print("[add_cong] Populated dependency for congruence using the four points")
      
      self.make_equal(ab, cd, deps=dep)
      print(f"[add_cong] Merged the value nodes of segments {ab.name} and {cd.name}")
      
      dep.algebra = (ab._val, cd._val)
      print(f"[add_cong] Recorded algebra: ({ab._val}, {cd._val})")
      
      self.cache_dep('cong', [a, b, c, d], dep)
      print("[add_cong] Cached dependency under 'cong'")
      
      result = []
      if not is_equal:
          result += [dep]
          print("[add_cong] Added dependency to result because segments were not already equal")
      
      if a not in [c, d] and b not in [c, d]:
          print("[add_cong] No shared endpoints detected. Returning result.")
          return result
      
      if b in [c, d]:
          print("[add_cong] Detected b is among [c, d]: swapping a and b")
          a, b = b, a
      if a == d:
          print("[add_cong] Detected a equals d: swapping c and d")
          c, d = d, c  # pylint: disable=unused-variable
      
      print("[add_cong] Finished processing congruence. Returning result.")
      return result

  def _maybe_add_cyclic_from_cong(
      self, a: Point, b: Point, c: Point, cong_ab_ac: Dependency
  ) -> list[Dependency]:
    """Maybe add a new cyclic predicate from given congruent segments."""
    ab = self._get_or_create_segment(a, b, deps=None)

    # all eq segs with one end being a.
    segs = [s for s in ab.val.neighbors(Segment) if a in s.points]

    # all points on circle (a, b)
    points = []
    for s in segs:
      x, y = list(s.points)
      points.append(x if y == a else y)

    # for sure both b and c are in points
    points = [p for p in points if p not in [b, c]]

    if len(points) < 2:
      return []


  def check_cong(self, points: list[Point]) -> bool:
    a, b, c, d = points
    if {a, b} == {c, d}:
      return True

    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)
    if ab is None or cd is None:
      return False
    return self.is_equal(ab, cd)




  def make_equal_pairs(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Line,
      cd: Line,
      mn: Line,
      pq: Line,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add ab/cd = mn/pq in case either two of (ab,cd,mn,pq) are equal."""

    print("🚀 make_equal_pairs function called!")
    depname = 'eqratio' if isinstance(ab, Segment) else 'eqangle'
    eqname = 'cong' if isinstance(ab, Segment) else 'para'

    is_equal = self.is_equal(mn, pq)

    if ab != cd:
      dep0 = deps.populate(depname, [a, b, c, d, m, n, p, q])
      deps = EmptyDependency(level=deps.level, rule_name=None)

      dep = Dependency(eqname, [a, b, c, d], None, deps.level)
      deps.why = [dep0, dep.why_me_or_cache(self, None)]

    elif eqname == 'para':  # ab == cd.
      colls = [a, b, c, d]
      if len(set(colls)) > 2:
        dep0 = deps.populate(depname, [a, b, c, d, m, n, p, q])
        deps = EmptyDependency(level=deps.level, rule_name=None)

        dep = Dependency('collx', colls, None, deps.level)
        deps.why = [dep0, dep.why_me_or_cache(self, None)]

    deps = deps.populate(eqname, [m, n, p, q])
    self.make_equal(mn, pq, deps=deps)

    deps.algebra = mn._val, pq._val
    self.cache_dep(eqname, [m, n, p, q], deps)

    if is_equal:
      return []
    return [deps]

  def maybe_make_equal_pairs(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Line,
      cd: Line,
      mn: Line,
      pq: Line,
      deps: EmptyDependency,
  ) -> Optional[list[Dependency]]:
    """Add ab/cd = mn/pq in case maybe either two of (ab,cd,mn,pq) are equal."""
    level = deps.level
    if self.is_equal(ab, cd, level):
      return self.make_equal_pairs(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)
    elif self.is_equal(mn, pq, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          m,
          n,
          p,
          q,
          a,
          b,
          c,
          d,
          mn,
          pq,
          ab,
          cd,
          deps,
      )
    elif self.is_equal(ab, mn, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          a,
          b,
          m,
          n,
          c,
          d,
          p,
          q,
          ab,
          mn,
          cd,
          pq,
          deps,
      )
    elif self.is_equal(cd, pq, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          c,
          d,
          p,
          q,
          a,
          b,
          m,
          n,
          cd,
          pq,
          ab,
          mn,
          deps,
      )
    else:
      return None


  def add_eqangle(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add eqangle made by 8 points in `points`."""
    if deps:
      deps = deps.copy()
    a, b, c, d, m, n, p, q = points
    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)
    mn, why3 = self.get_line_thru_pair_why(m, n)
    pq, why4 = self.get_line_thru_pair_why(p, q)

    a, b = ab.points
    c, d = cd.points
    m, n = mn.points
    p, q = pq.points

    if deps and why1 + why2 + why3 + why4:
      dep0 = deps.populate('eqangle', points)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why1 + why2 + why3 + why4

    add = self.maybe_make_equal_pairs(
        a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps
    )

    if add is not None:
      return add

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)
    self.connect_val(mn, deps=None)
    self.connect_val(pq, deps=None)

    add = []
    # if (
    #     ab.val != cd.val
    #     and mn.val != pq.val
    #     and (ab.val != mn.val or cd.val != pq.val)
    # ):
    #   add += self._add_eqangle(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)

    # if (
    #     ab.val != mn.val
    #     and cd.val != pq.val
    #     and (ab.val != cd.val or mn.val != pq.val)
    # ):
    #   add += self._add_eqangle(  # pylint: disable=arguments-out-of-order
    #       a,
    #       b,
    #       m,
    #       n,
    #       c,
    #       d,
    #       p,
    #       q,
    #       ab,
    #       mn,
    #       cd,
    #       pq,
    #       deps,
    #   )

    return add


  def check_eqangle(self, points: list[Point]) -> bool:
    """Check if two angles are equal."""
    a, b, c, d, m, n, p, q = points

    if {a, b} == {c, d} and {m, n} == {p, q}:
      return True
    if {a, b} == {m, n} and {c, d} == {p, q}:
      return True

    if (a == b) or (c == d) or (m == n) or (p == q):
      return False
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    mn = self._get_line(m, n)
    pq = self._get_line(p, q)

    if {a, b} == {c, d} and mn and pq and self.is_equal(mn, pq):
      return True
    if {a, b} == {m, n} and cd and pq and self.is_equal(cd, pq):
      return True
    if {p, q} == {m, n} and ab and cd and self.is_equal(ab, cd):
      return True
    if {p, q} == {c, d} and ab and mn and self.is_equal(ab, mn):
      return True

    if not ab or not cd or not mn or not pq:
      return False

    if self.is_equal(ab, cd) and self.is_equal(mn, pq):
      return True
    if self.is_equal(ab, mn) and self.is_equal(cd, pq):
      return True

    if not (ab.val and cd.val and mn.val and pq.val):
      return False

    if (ab.val, cd.val) == (mn.val, pq.val) or (ab.val, mn.val) == (
        cd.val,
        pq.val,
    ):
      return True

    for ang1, _, _ in gm.all_angles(ab._val, cd._val):
      for ang2, _, _ in gm.all_angles(mn._val, pq._val):
        if self.is_equal(ang1, ang2):
          return True

    if self.check_perp([a, b, m, n]) and self.check_perp([c, d, p, q]):
      return True
    if self.check_perp([a, b, p, q]) and self.check_perp([c, d, m, n]):
      return True

    return False

  def _get_or_create_ratio(
      self, s1: Segment, s2: Segment, deps: Dependency
  ) -> tuple[Ratio, Ratio, list[Dependency]]:
    return self._get_or_create_ratio_l(s1._val, s2._val, deps)

  def _get_or_create_ratio_l(
      self, l1: Length, l2: Length, deps: Dependency
  ) -> tuple[Ratio, Ratio, list[Dependency]]:
    """Get or create a new Ratio from two Lenghts l1 and l2."""
    for r in self.type2nodes[Ratio]:
      if r.lengths == (l1.rep(), l2.rep()):
        l1_, l2_ = r._l
        why1 = l1.why_equal([l1_], None) + l1_.why_rep()
        why2 = l2.why_equal([l2_], None) + l2_.why_rep()
        return r, r.opposite, why1 + why2

    l1, why1 = l1.rep_and_why()
    l2, why2 = l2.rep_and_why()
    r12 = self.new_node(Ratio, f'{l1.name}/{l2.name}')
    r21 = self.new_node(Ratio, f'{l2.name}/{l1.name}')
    self.connect(l1, r12, deps)
    self.connect(l2, r21, deps)
    self.connect(r12, r21, deps)
    r12.set_lengths(l1, l2)
    r21.set_lengths(l2, l1)
    r12.opposite = r21
    r21.opposite = r12
    return r12, r21, why1 + why2
  
  def add_eqratio3(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add three eqratios through a list of 6 points (due to parallel lines)."""
    a, b, c, d, m, n = points
    #   a -- b
    #  m  --  n
    # c   --   d
    add = []
    add += self.add_eqratio([m, a, m, c, n, b, n, d], deps)
    add += self.add_eqratio([a, m, a, c, b, n, b, d], deps)
    add += self.add_eqratio([c, m, c, a, d, n, d, b], deps)
    if m == n:
      add += self.add_eqratio([m, a, m, c, a, b, c, d], deps)
    return add

  def _add_eqratio(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Segment,
      cd: Segment,
      mn: Segment,
      pq: Segment,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add a new eqratio from 8 points (core)."""
    if deps:
      deps = deps.copy()

    args = [a, b, c, d, m, n, p, q]
    i = 0
    # for x, y, xy in [(a, b, ab), (c, d, cd), (m, n, mn), (p, q, pq)]:
    #   if {x, y} == set(xy.points):
    #     continue
    #   x_, y_ = list(xy.points)
    #   if deps:
    #     deps = deps.extend(self, 'eqratio', list(args), 'cong', [x, y, x_, y_])
    #   args[2 * i - 2] = x_
    #   args[2 * i - 1] = y_

    add = []
    ab_cd, cd_ab, why1 = self._get_or_create_ratio(ab, cd, deps=None)
    mn_pq, pq_mn, why2 = self._get_or_create_ratio(mn, pq, deps=None)

    why = why1 + why2
    if why:
      dep0 = deps.populate('eqratio', args)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    lab, lcd = ab_cd._l
    lmn, lpq = mn_pq._l

    # a, b = lab._obj.points
    # c, d = lcd._obj.points
    # m, n = lmn._obj.points
    # p, q = lpq._obj.points

    is_eq1 = self.is_equal(ab_cd, mn_pq)
    deps1 = None
    if deps:
      deps1 = deps.populate('eqratio', [a, b, c, d, m, n, p, q])
      deps1.algebra = [ab._val, cd._val, mn._val, pq._val]
    if not is_eq1:
      add += [deps1]
    self.cache_dep('eqratio', [a, b, c, d, m, n, p, q], deps1)
    self.make_equal(ab_cd, mn_pq, deps=deps1)

    is_eq2 = self.is_equal(cd_ab, pq_mn)
    deps2 = None
    if deps:
      deps2 = deps.populate('eqratio', [c, d, a, b, p, q, m, n])
      deps2.algebra = [cd._val, ab._val, pq._val, mn._val]
    if not is_eq2:
      add += [deps2]
    self.cache_dep('eqratio', [c, d, a, b, p, q, m, n], deps2)
    self.make_equal(cd_ab, pq_mn, deps=deps2)
    return add

  def add_eqratio(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new eqratio from 8 points."""
    if deps:
      deps = deps.copy()
    a, b, c, d, m, n, p, q = points
    ab = self._get_or_create_segment(a, b, deps=None)
    cd = self._get_or_create_segment(c, d, deps=None)
    mn = self._get_or_create_segment(m, n, deps=None)
    pq = self._get_or_create_segment(p, q, deps=None)

    add = self.maybe_make_equal_pairs(
        a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps
    )

    if add is not None:
      return add

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)
    self.connect_val(mn, deps=None)
    self.connect_val(pq, deps=None)

    add = []
    if (
        ab.val != cd.val
        and mn.val != pq.val
        and (ab.val != mn.val or cd.val != pq.val)
    ):
      add += self._add_eqratio(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)

    if (
        ab.val != mn.val
        and cd.val != pq.val
        and (ab.val != cd.val or mn.val != pq.val)
    ):
      add += self._add_eqratio(  # pylint: disable=arguments-out-of-order
          a,
          b,
          m,
          n,
          c,
          d,
          p,
          q,
          ab,
          mn,
          cd,
          pq,
          deps,
      )
    return add

  def add_simtri_check(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    if nm.same_clock(*[p.num for p in points]):
      return self.add_simtri(points, deps)
    #return self.add_simtri2(points, deps)

  def add_contri_check(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    if nm.same_clock(*[p.num for p in points]):
      return self.add_contri(points, deps)
    #return self.add_contri2(points, deps)

  def enum_sides(
      self, points: list[Point]
  ) -> Generator[list[Point], None, None]:
    a, b, c, x, y, z = points
    yield [a, b, x, y]
    yield [b, c, y, z]
    yield [c, a, z, x]

  def enum_triangle(
      self, points: list[Point]
  ) -> Generator[list[Point], None, None]:
    a, b, c, x, y, z = points
    yield [a, b, a, c, x, y, x, z]
    yield [b, a, b, c, y, x, y, z]
    yield [c, a, c, b, z, x, z, y]

  def add_simtri(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two similar triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]

    for args in self.enum_triangle(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_triangle(points):
      if problem.hashed('eqratio6', args) in hashs:
        continue
      add += self.add_eqratio(args, deps=deps)

    return add
  
  def add_contri(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two congruent triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]
    for args in self.enum_triangle(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_sides(points):
      if problem.hashed('cong', args) in hashs:
        continue
      add += self.add_cong(args, deps=deps)
    return add

  def cache_dep(
      self, name: str, args: list[Point], premises: list[Dependency]
  ) -> None:
    hashed = problem.hashed(name, args)
    if hashed in self.cache:
      return
    self.cache[hashed] = premises

 
  def add_clause(
      self,
      clause: problem.Clause,
      plevel: int,
      definitions: dict[str, problem.Definition],
      verbose: int = False,
  ) -> tuple[list[Dependency], int]:
    """Add a new clause of construction, e.g. a new excenter."""
    print(f"Processing clause: {clause.points} | {[c.name for c in clause.constructions]}")
    existing_points = self.all_points()
    new_points = [Point(name) for name in clause.points]

    new_points_dep_points = set()
    new_points_dep = []

    print(clause.constructions.__sizeof__())
    # Step 1: check for all deps.
    for c in clause.constructions:
      print(f"Checking construction: {c.name}")
      cdef = definitions[c.name]

      print(cdef.deps.txt())

      if len(cdef.construction.args) != len(c.args):
        if len(cdef.construction.args) - len(c.args) == len(clause.points):
          c.args = clause.points + c.args
        else:
          correct_form = ' '.join(cdef.points + ['=', c.name] + cdef.args)
          raise ValueError('Argument mismatch. ' + correct_form)

      mapping = dict(zip(cdef.construction.args, c.args))
      c_name = 'midp' if c.name == 'midpoint' else c.name
      deps = EmptyDependency(level=0, rule_name=problem.CONSTRUCTION_RULE)
      deps.construction = Dependency(c_name, c.args, rule_name=None, level=0)

      print(cdef.deps.constructions.__sizeof__())
      for d in cdef.deps.constructions:
        print('I was here first')
        args = self.names2points([mapping[a] for a in d.args])
        new_points_dep_points.update(args)
        if not self.check(d.name, args):
          raise DepCheckFailError(
              d.name + ' ' + ' '.join([x.name for x in args])
          )
        deps.why += [
            Dependency(
                d.name, args, rule_name=problem.CONSTRUCTION_RULE, level=0
            )
        ]

      new_points_dep += [deps]

    # Step 2: draw.
    def range_fn() -> (
        list[Union[nm.Point, nm.Line, nm.Circle, nm.HalfLine, nm.HoleCircle]]
    ):
      print(f"Generating objects for clause: {clause.points}")
      to_be_intersected = []
      for c in clause.constructions:
        cdef = definitions[c.name]
        mapping = dict(zip(cdef.construction.args, c.args))
        for n in cdef.numerics:
          args = [mapping[a] for a in n.args]
          args = list(map(lambda x: self.get(x, lambda: int(x)), args))
          print('Before a print')
          to_be_intersected += nm.sketch(n.name, args)

      return to_be_intersected

    is_total_free = (
        len(clause.constructions) == 1 and clause.constructions[0].name in FREE
    )
    is_semi_free = (
        len(clause.constructions) == 1
        and clause.constructions[0].name in INTERSECT
    )

    existing_points = [p.num for p in existing_points]

    def draw_fn() -> list[nm.Point]:
      to_be_intersected = range_fn()
      return nm.reduce(to_be_intersected, existing_points)

    rely_on = set()
    for c in clause.constructions:
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))
      for n in cdef.numerics:
        args = [mapping[a] for a in n.args]
        args = list(map(lambda x: self.get(x, lambda: int(x)), args))
        rely_on.update([a for a in args if isinstance(a, Point)])

    for p in rely_on:
      p.change.update(new_points)

    nums = draw_fn()
    for p, num, num0 in zip(new_points, nums, clause.nums):
      p.co_change = new_points
      if isinstance(num0, nm.Point):
        num = num0
      elif isinstance(num0, (tuple, list)):
        x, y = num0
        num = nm.Point(x, y)

      p.num = num

    # check two things.
    if nm.check_too_close(nums, existing_points):
      raise PointTooCloseError()
    if nm.check_too_far(nums, existing_points):
      raise PointTooFarError()

    # Commit: now that all conditions are passed.
    # add these points to current graph.
    for p in new_points:
      self._name2point[p.name] = p
      self._name2node[p.name] = p
      self.type2nodes[Point].append(p)

    for p in new_points:
      p.why = sum([d.why for d in new_points_dep], [])  # to generate txt logs.
      p.group = new_points
      p.dep_points = new_points_dep_points
      p.dep_points.update(new_points)
      p.plevel = plevel

    # movement dependency:
    rely_dict_0 = defaultdict(lambda: [])

    for c in clause.constructions:
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))
      for p, ps in cdef.rely.items():
        p = mapping[p]
        ps = [mapping[x] for x in ps]
        rely_dict_0[p].append(ps)

    rely_dict = {}
    for p, pss in rely_dict_0.items():
      ps = sum(pss, [])
      if len(pss) > 1:
        ps = [x for x in ps if x != p]

      p = self._name2point[p]
      ps = self.names2nodes(ps)
      rely_dict[p] = ps

    for p in new_points:
      p.rely_on = set(rely_dict.get(p, []))
      for x in p.rely_on:
        if not hasattr(x, 'base_rely_on'):
          x.base_rely_on = set()
      p.base_rely_on = set.union(*[x.base_rely_on for x in p.rely_on] + [set()])
      if is_total_free or is_semi_free:
        p.rely_on.add(p)
        p.base_rely_on.add(p)

    plevel_done = set()
    added = []
    basics = []
    # Step 3: build the basics.
    for c, deps in zip(clause.constructions, new_points_dep):
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))

      # not necessary for proofing, but for visualization.
      # c_args = list(map(lambda x: self.get(x, lambda: int(x)), c.args))
      # self.additionally_draw(c.name, c_args)

      for points, bs in cdef.basics:
        if points:
          points = self.names2nodes([mapping[p] for p in points])
          points = [p for p in points if p not in plevel_done]
          for p in points:
            p.plevel = plevel
          plevel_done.update(points)
          plevel += 1
        else:
          continue

        for b in bs:
          if b.name != 'rconst':
            args = [mapping[a] for a in b.args]
          else:
            num, den = map(int, b.args[-2:])
            args = [mapping[a] for a in b.args[:-2]]

          args = list(map(lambda x: self.get(x, lambda: int(x)), args))

          print(f"Adding {b.name} with args: {[arg.name  for arg in args]}")
          adds = self.add_piece(name=b.name, args=args, deps=deps)
          basics.append((b.name, args, deps))
          if adds:
            added += adds
            for add in adds:
              self.cache_dep(add.name, add.args, add)

    assert len(plevel_done) == len(new_points)
    for p in new_points:
      p.basics = basics

    return added, plevel

  def all_eqangle_same_lines(self) -> Generator[tuple[Point, ...], None, None]:
    for l1, l2 in utils.perm2(self.type2nodes[Line]):
      for a, b, c, d, e, f, g, h in utils.all_8points(l1, l2, l1, l2):
        if (a, b, c, d) != (e, f, g, h):
          yield a, b, c, d, e, f, g, h

  def all_eqangles_8points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 8 points that make two equal angles."""
    # Case 1: (l1-l2) = (l3-l4), including because l1//l3, l2//l4 (para-para)
    angss = []
    for measure in self.type2nodes[Measure]:
      angs = measure.neighbors(Angle)
      angss.append(angs)

    # include the angs that do not have any measure.
    angss.extend([[ang] for ang in self.type2nodes[Angle] if ang.val is None])

    line_pairss = []
    for angs in angss:
      line_pairs = set()
      for ang in angs:
        d1, d2 = ang.directions
        if d1 is None or d2 is None:
          continue
        l1s = d1.neighbors(Line)
        l2s = d2.neighbors(Line)
        line_pairs.update(set(utils.cross(l1s, l2s)))
      line_pairss.append(line_pairs)

    # include (d1, d2) in which d1 does not have any angles.
    noang_ds = [d for d in self.type2nodes[Direction] if not d.neighbors(Angle)]

    for d1 in noang_ds:
      for d2 in self.type2nodes[Direction]:
        if d1 == d2:
          continue
        l1s = d1.neighbors(Line)
        l2s = d2.neighbors(Line)
        if len(l1s) < 2 and len(l2s) < 2:
          continue
        line_pairss.append(set(utils.cross(l1s, l2s)))
        line_pairss.append(set(utils.cross(l2s, l1s)))

    # Case 2: d1 // d2 => (d1-d3) = (d2-d3)
    # include lines that does not have any direction.
    nodir_ls = [l for l in self.type2nodes[Line] if l.val is None]

    for line in nodir_ls:
      for d in self.type2nodes[Direction]:
        l1s = d.neighbors(Line)
        if len(l1s) < 2:
          continue
        l2s = [line]
        line_pairss.append(set(utils.cross(l1s, l2s)))
        line_pairss.append(set(utils.cross(l2s, l1s)))

    record = set()
    for line_pairs in line_pairss:
      for pair1, pair2 in utils.perm2(list(line_pairs)):
        (l1, l2), (l3, l4) = pair1, pair2
        if l1 == l2 or l3 == l4:
          continue
        if (l1, l2) == (l3, l4):
          continue
        if (l1, l2, l3, l4) in record:
          continue
        record.add((l1, l2, l3, l4))
        for a, b, c, d, e, f, g, h in utils.all_8points(l1, l2, l3, l4):
          yield (a, b, c, d, e, f, g, h)

    for a, b, c, d, e, f, g, h in self.all_eqangle_same_lines():
      yield a, b, c, d, e, f, g, h

  def all_eqangles_6points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 6 points that make two equal angles."""
    record = set()
    for a, b, c, d, e, f, g, h in self.all_eqangles_8points():
      if (
          a not in (c, d)
          and b not in (c, d)
          or e not in (g, h)
          and f not in (g, h)
      ):
        continue

      if b in (c, d):
        a, b = b, a  # now a in c, d
      if f in (g, h):
        e, f = f, e  # now e in g, h
      if a == d:
        c, d = d, c  # now a == c
      if e == h:
        g, h = h, g  # now e == g
      if (a, b, c, d, e, f, g, h) in record:
        continue
      record.add((a, b, c, d, e, f, g, h))
      yield a, b, c, d, e, f, g, h  # where a==c, e==g

  def all_paras(self) -> Generator[tuple[Point, ...], None, None]:
    for d in self.type2nodes[Direction]:
      for l1, l2 in utils.perm2(d.neighbors(Line)):
        for a, b, c, d in utils.all_4points(l1, l2):
          yield a, b, c, d

  def all_perps(self) -> Generator[tuple[Point, ...], None, None]:
    for ang in self.vhalfpi.neighbors(Angle):
      d1, d2 = ang.directions
      if d1 is None or d2 is None:
        continue
      if d1 == d2:
        continue
      for l1, l2 in utils.cross(d1.neighbors(Line), d2.neighbors(Line)):
        for a, b, c, d in utils.all_4points(l1, l2):
          yield a, b, c, d

  def all_congs(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Length]:
      for s1, s2 in utils.perm2(l.neighbors(Segment)):
        (a, b), (c, d) = s1.points, s2.points
        for x, y in [(a, b), (b, a)]:
          for m, n in [(c, d), (d, c)]:
            yield x, y, m, n

  def all_eqratios_8points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 8 points that make two equal ratios."""
    ratss = []
    for value in self.type2nodes[Value]:
      rats = value.neighbors(Ratio)
      ratss.append(rats)

    # include the rats that do not have any val.
    ratss.extend([[rat] for rat in self.type2nodes[Ratio] if rat.val is None])

    seg_pairss = []
    for rats in ratss:
      seg_pairs = set()
      for rat in rats:
        l1, l2 = rat.lengths
        if l1 is None or l2 is None:
          continue
        s1s = l1.neighbors(Segment)
        s2s = l2.neighbors(Segment)
        seg_pairs.update(utils.cross(s1s, s2s))
      seg_pairss.append(seg_pairs)

    # include (l1, l2) in which l1 does not have any ratio.
    norat_ls = [l for l in self.type2nodes[Length] if not l.neighbors(Ratio)]

    for l1 in norat_ls:
      for l2 in self.type2nodes[Length]:
        if l1 == l2:
          continue
        s1s = l1.neighbors(Segment)
        s2s = l2.neighbors(Segment)
        if len(s1s) < 2 and len(s2s) < 2:
          continue
        seg_pairss.append(set(utils.cross(s1s, s2s)))
        seg_pairss.append(set(utils.cross(s2s, s1s)))

    # include Seg that does not have any Length.
    nolen_ss = [s for s in self.type2nodes[Segment] if s.val is None]

    for seg in nolen_ss:
      for l in self.type2nodes[Length]:
        s1s = l.neighbors(Segment)
        if len(s1s) == 1:
          continue
        s2s = [seg]
        seg_pairss.append(set(utils.cross(s1s, s2s)))
        seg_pairss.append(set(utils.cross(s2s, s1s)))

    record = set()
    for seg_pairs in seg_pairss:
      for pair1, pair2 in utils.perm2(list(seg_pairs)):
        (s1, s2), (s3, s4) = pair1, pair2
        if s1 == s2 or s3 == s4:
          continue
        if (s1, s2) == (s3, s4):
          continue
        if (s1, s2, s3, s4) in record:
          continue
        record.add((s1, s2, s3, s4))
        a, b = s1.points
        c, d = s2.points
        e, f = s3.points
        g, h = s4.points

        for x, y in [(a, b), (b, a)]:
          for z, t in [(c, d), (d, c)]:
            for m, n in [(e, f), (f, e)]:
              for p, q in [(g, h), (h, g)]:
                yield (x, y, z, t, m, n, p, q)

    segss = []
    # finally the list of ratios that is equal to 1.0
    for length in self.type2nodes[Length]:
      segs = length.neighbors(Segment)
      segss.append(segs)

    segs_pair = list(utils.perm2(list(segss)))
    segs_pair += list(zip(segss, segss))
    for segs1, segs2 in segs_pair:
      for s1, s2 in utils.perm2(list(segs1)):
        for s3, s4 in utils.perm2(list(segs2)):
          if (s1, s2) == (s3, s4) or (s1, s3) == (s2, s4):
            continue
          if (s1, s2, s3, s4) in record:
            continue
          record.add((s1, s2, s3, s4))
          a, b = s1.points
          c, d = s2.points
          e, f = s3.points
          g, h = s4.points

          for x, y in [(a, b), (b, a)]:
            for z, t in [(c, d), (d, c)]:
              for m, n in [(e, f), (f, e)]:
                for p, q in [(g, h), (h, g)]:
                  yield (x, y, z, t, m, n, p, q)

  def all_eqratios_6points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 6 points that make two equal angles."""
    record = set()
    for a, b, c, d, e, f, g, h in self.all_eqratios_8points():
      if (
          a not in (c, d)
          and b not in (c, d)
          or e not in (g, h)
          and f not in (g, h)
      ):
        continue
      if b in (c, d):
        a, b = b, a
      if f in (g, h):
        e, f = f, e
      if a == d:
        c, d = d, c
      if e == h:
        g, h = h, g
      if (a, b, c, d, e, f, g, h) in record:
        continue
      record.add((a, b, c, d, e, f, g, h))
      yield a, b, c, d, e, f, g, h  # now a==c, e==g


  def all_colls(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Line]:
      for x, y, z in utils.perm3(l.neighbors(Point)):
        yield x, y, z

  def all_midps(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Line]:
      for a, b, c in utils.perm3(l.neighbors(Point)):
        if self.check_cong([a, b, a, c]):
          yield a, b, c

  def all_circles(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Length]:
      p2p = defaultdict(list)
      for s in l.neighbors(Segment):
        a, b = s.points
        p2p[a].append(b)
        p2p[b].append(a)
      for p, ps in p2p.items():
        if len(ps) >= 3:
          for a, b, c in utils.perm3(ps):
            yield p, a, b, c

  def all_cyclics(self) -> Generator[tuple[Point, ...], None, None]: 
      for c in self.type2nodes[Circle]: 
        for x, y, z, t in utils.perm4(c.neighbors(Point)): 
          yield x, y, z, t 
