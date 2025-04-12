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

"""Implements Deductive Database (DD)."""

# pylint: disable=g-multiple-import,g-importing-member
import time
from typing import Any, Callable, Generator

import geometry as gm
import graph as gh
import numericals as nm
import problem as pr
from problem import Dependency, EmptyDependency

# pylint: disable=protected-access
# pylint: disable=unused-argument



def rotate_simtri(
    a: gm.Point, b: gm.Point, c: gm.Point, x: gm.Point, y: gm.Point, z: gm.Point
) -> Generator[tuple[gm.Point, ...], None, None]:
  """Rotate points around for similar triangle predicates."""
  yield (z, y, x, c, b, a)
  for p in [
      (b, c, a, y, z, x),
      (c, a, b, z, x, y),
      (x, y, z, a, b, c),
      (y, z, x, b, c, a),
      (z, x, y, c, a, b),
  ]:
    yield p
    yield p[::-1]


def match_cong_cong_eqangle6_ncoll_contri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cong A B P Q, cong B C Q R, eqangle6 B A B C Q P Q R, ncoll A B C => contri* A B C P Q R."""
  print('DD THEOREM: match_cong_cong_eqangle6_ncoll_contri')
  record = set()
  for a, b, p, q in g_matcher('cong'):
    for c in g.type2nodes[gm.Point]:
      if c in (a, b):
        continue
      for r in g.type2nodes[gm.Point]:
        if r in (p, q):
          continue

        in_record = False
        for x in [
            (c, b, a, r, q, p),
            (p, q, r, a, b, c),
            (r, q, p, c, b, a),
        ]:
          if x in record:
            in_record = True
            break

        if in_record:
          continue

        if not g.check_cong([b, c, q, r]):
          continue
        if not g.check_ncoll([a, b, c]):
          continue

        if nm.same_clock(a.num, b.num, c.num, p.num, q.num, r.num):
          if g.check_eqangle([b, a, b, c, q, p, q, r]):
            record.add((a, b, c, p, q, r))
            yield dict(zip('ABCPQR', [a, b, c, p, q, r]))
        else:
          if g.check_eqangle([b, a, b, c, q, r, q, p]):
            record.add((a, b, c, p, q, r))
            yield dict(zip('ABCPQR', [a, b, c, p, q, r]))


def match_eqratio6_eqangle6_ncoll_simtri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio6 B A B C Q P Q R, eqratio6 C A C B R P R Q, ncoll A B C => simtri* A B C P Q R."""
  enums = g_matcher('eqratio6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    if nm.same_clock(a.num, b.num, c.num, p.num, q.num, r.num):
      if g.check_eqangle([b, a, b, c, q, p, q, r]):
        record.add((a, b, c, p, q, r))
        yield dict(zip('ABCPQR', [a, b, c, p, q, r]))
    elif g.check_eqangle([b, a, b, c, q, r, q, p]):
      record.add((a, b, c, p, q, r))
      yield dict(zip('ABCPQR', [a, b, c, p, q, r]))



def match_all(
    name: str, g: gh.Graph
) -> Generator[tuple[gm.Point, ...], None, None]:
  """Match all instances of a certain relation."""
  print(f'MATCHING {name}')
  if name in ['ncoll', 'npara', 'nperp']:
    return []
  if name == 'coll':
    print('MATCHING coll')
    return g.all_colls()
  if name == 'para':
    return g.all_paras()
  if name == 'perp':
    return g.all_perps()
  if name == 'cong':
    return g.all_congs()
  if name == 'eqangle':
    return g.all_eqangles_8points()
  if name == 'eqangle6':
    return g.all_eqangles_6points()
  if name == 'eqratio':
    return g.all_eqratios_8points()
  if name == 'eqratio6':
    return g.all_eqratios_6points()
  if name == 'cyclic':
    return g.all_cyclics()
  if name == 'midp':
    return g.all_midps()
  if name == 'circle':
    return g.all_circles()
  raise ValueError(f'Unrecognize {name}')


def cache_match(
    graph: gh.Graph,
) -> Callable[str, list[tuple[gm.Point, ...]]]:
  """Cache throughout one single BFS level."""
  cache = {}

  def match_fn(name: str) -> list[tuple[gm.Point, ...]]:
    if name in cache:
      return cache[name]

    result = list(match_all(name, graph))
    cache[name] = result
    return result

  return match_fn


def try_to_map(
    clause_enum: list[tuple[pr.Construction, list[tuple[gm.Point, ...]]]],
    mapping: dict[str, gm.Point],
) -> Generator[dict[str, gm.Point], None, None]:
  """Recursively try to match the remaining points given current mapping."""
  if not clause_enum:
    yield mapping
    return

  clause, enum = clause_enum[0]
  for points in enum:
    mpcpy = dict(mapping)

    fail = False
    for p, a in zip(points, clause.args):
      if a in mpcpy and mpcpy[a] != p or p in mpcpy and mpcpy[p] != a:
        fail = True
        break
      mpcpy[a] = p
      mpcpy[p] = a

    if fail:
      continue

    for m in try_to_map(clause_enum[1:], mpcpy):
      yield m


def match_generic(
    g: gh.Graph,
    cache: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem
) -> Generator[dict[str, gm.Point], None, None]:
  """Match any generic rule that is not one of the above match_*() rules."""
  clause2enum = {}

  clauses = []
  numerical_checks = []
  for clause in theorem.premise:
    if clause.name in ['ncoll', 'npara', 'nperp', 'sameside']:
      numerical_checks.append(clause)
      continue

    enum = cache(clause.name)
    if len(enum) == 0:  # pylint: disable=g-explicit-length-test
      return 0

    clause2enum[clause] = enum
    clauses.append((len(set(clause.args)), clause))

  clauses = sorted(clauses, key=lambda x: x[0], reverse=True)
  _, clauses = zip(*clauses)

  for mapping in try_to_map([(c, clause2enum[c]) for c in clauses], {}):
    if not mapping:
      continue

    checks_ok = True
    for check in numerical_checks:
      args = [mapping[a] for a in check.args]
      if check.name == 'ncoll':
        checks_ok = g.check_ncoll(args)
      elif check.name == 'sameside':
        checks_ok = g.check_sameside(args)
      if not checks_ok:
        break
    if not checks_ok:
      continue

    yield mapping


BUILT_IN_FNS = {
    'cong_cong_eqangle6_ncoll_contri*': match_cong_cong_eqangle6_ncoll_contri,
    'eqratio6_eqangle6_ncoll_simtri*': match_eqratio6_eqangle6_ncoll_simtri
}


SKIP_THEOREMS = set()


def set_skip_theorems(theorems: set[str]) -> None:
  SKIP_THEOREMS.update(theorems)


MAX_BRANCH = 50_000


def match_one_theorem(
    g: gh.Graph,
    cache: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem
) -> Generator[dict[str, gm.Point], None, None]:
  """Match all instances of a single theorem (rule)."""
  if cache is None:
    cache = cache_match(g)

  if theorem.name in SKIP_THEOREMS:
    return []

  if theorem.name.split('_')[-1] in SKIP_THEOREMS:
    return []

  if theorem.name in BUILT_IN_FNS:
    mps = BUILT_IN_FNS[theorem.name](g, cache, theorem)
  else:
    print(f'MATCHING {theorem.name}')
    mps = match_generic(g, cache, theorem)

  mappings = []
  for mp in mps:
    mappings.append(mp)
    if len(mappings) > MAX_BRANCH:  # cap branching at this number.
      break

  return mappings


def match_all_theorems(
    g: gh.Graph, theorems: list[pr.Theorem], goal: pr.Clause
) -> dict[pr.Theorem, dict[pr.Theorem, dict[str, gm.Point]]]:
  """Match all instances of all theorems (rules)."""
  cache = cache_match(g)
  # for BFS, collect all potential matches
  # and then do it at the same time
  theorem2mappings = {}

  # Step 1: list all matches
  for _, theorem in theorems.items():
    name = theorem.name
    if name.split('_')[-1] in [
        'acompute',
        'rcompute',
        'fixl',
        'fixc',
        'fixb',
        'fixt',
        'fixp',
    ]:
      if goal and goal.name != name:
        continue

    mappings = match_one_theorem(g, cache, theorem)
    if len(mappings):  # pylint: disable=g-explicit-length-test
      print(f'Matching {name} with {len(mappings)} mappings')
      theorem2mappings[theorem] = list(mappings)

  
  return theorem2mappings


def bfs_one_level(
    g: gh.Graph,
    theorems: list[pr.Theorem],
    level: int,
    controller: pr.Problem,
    verbose: bool = False,
    nm_check: bool = False,
    timeout: int = 600,
) -> tuple[
    list[pr.Dependency],
    dict[str, list[tuple[gm.Point, ...]]],
    dict[str, list[tuple[gm.Point, ...]]],
    int,
]:
  """Forward deduce one breadth-first level."""

  # Step 1: match all theorems:
  theorem2mappings = match_all_theorems(g, theorems, controller.goal)

  # Step 2: traceback for each deduce:
  theorem2deps = {}
  t0 = time.time()
  for theorem, mappings in theorem2mappings.items():
    if time.time() - t0 > timeout:
      break
    mp_deps = []
    for mp in mappings:
      deps = EmptyDependency(level=level, rule_name=theorem.rule_name)
      fail = False  # finding why deps might fail.

      for p in theorem.premise:
        p_args = [mp[a] for a in p.args]
        # Trivial deps.
        if p.name == 'cong':
          a, b, c, d = p_args
          if {a, b} == {c, d}:
            continue
        if p.name == 'para':
          a, b, c, d = p_args
          if {a, b} == {c, d}:
            continue

        if theorem.name in [
            'cong_cong_eqangle6_ncoll_contri*',
            'eqratio6_eqangle6_ncoll_simtri*',
        ]:
          if p.name in ['eqangle', 'eqangle6']:  # SAS or RAR
            b, a, b, c, y, x, y, z = (  # pylint: disable=redeclared-assigned-name,unused-variable
                p_args
            )
            if not nm.same_clock(a.num, b.num, c.num, x.num, y.num, z.num):
              p_args = b, a, b, c, y, z, y, x

        dep = Dependency(p.name, p_args, rule_name='', level=level)
        try:
          dep = dep.why_me_or_cache(g, level)
        except:  # pylint: disable=bare-except
          fail = True
          break

        if dep.why is None:
          fail = True
          break
        g.cache_dep(p.name, p_args, dep)
        deps.why.append(dep)

      if fail:
        continue

      mp_deps.append((mp, deps))
    theorem2deps[theorem] = mp_deps

  theorem2deps = list(theorem2deps.items())

  # Step 3: add conclusions to graph.
  # Note that we do NOT mix step 2 and 3, strictly going for BFS.
  added = []
  for theorem, mp_deps in theorem2deps:
    for mp, deps in mp_deps:
      if time.time() - t0 > timeout:
        break
      name, args = theorem.conclusion_name_args(mp)
      hash_conclusion = pr.hashed(name, args)
      if hash_conclusion in g.cache:
        continue

      print(name)

      add = g.add_piece(name, args, deps=deps)
      added += add

  branching = len(added)

  # Check if goal is found
  if controller.goal:
    args = []

    for a in controller.goal.args:
      if a in g._name2node:
        a = g._name2node[a]
      elif a.isdigit():
        a = int(a)
      args.append(a)

    if g.check(controller.goal.name, args):
      return added, {}, {}, branching

  derives = []
  eq4s = []
  return added, derives, eq4s, branching

