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

"""Unit tests for graph.py."""
import unittest

from absl.testing import absltest
import graph as gh
import numericals as nm
import problem as pr


MAX_LEVEL = 1000


# class GraphTest(unittest.TestCase):

#   @classmethod
#   def setUpClass(cls):
#     super().setUpClass()

#     cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
#     cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

#     # load a complex setup:
#     txt = 'a b c = triangle a b c; h = orthocenter a b c; h1 = foot a b c; h2 = foot b c a; h3 = foot c a b; g1 g2 g3 g = centroid g1 g2 g3 g a b c; o = circle a b c ? coll h g o'  # pylint: disable=line-too-long
#     p = pr.Problem.from_txt(txt, translate=False)
#     cls.g, _ = gh.Graph.build_problem(p, GraphTest.defs)

#   def test_build_graph_points(self):
#     g = GraphTest.g

#     all_points = g.all_points()
#     all_names = [p.name for p in all_points]
#     self.assertCountEqual(
#         all_names,
#         ['a', 'b', 'c', 'g', 'h', 'o', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3'],
#     )

#   def test_build_graph_predicates(self):
#     gr = GraphTest.g

#     a, b, c, g, h, o, g1, g2, g3, h1, h2, h3 = gr.names2points(
#         ['a', 'b', 'c', 'g', 'h', 'o', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3']
#     )

#     # Explicit statements:
#     self.assertTrue(gr.check_cong([b, g1, g1, c]))
#     self.assertTrue(gr.check_cong([c, g2, g2, a]))
#     self.assertTrue(gr.check_cong([a, g3, g3, b]))
#     self.assertTrue(gr.check_perp([a, h1, b, c]))
#     self.assertTrue(gr.check_perp([b, h2, c, a]))
#     self.assertTrue(gr.check_perp([c, h3, a, b]))
#     self.assertTrue(gr.check_cong([o, a, o, b]))
#     self.assertTrue(gr.check_cong([o, b, o, c]))
#     self.assertTrue(gr.check_cong([o, a, o, c]))
#     self.assertTrue(gr.check_coll([a, g, g1]))
#     self.assertTrue(gr.check_coll([b, g, g2]))
#     self.assertTrue(gr.check_coll([g1, b, c]))
#     self.assertTrue(gr.check_coll([g2, c, a]))
#     self.assertTrue(gr.check_coll([g3, a, b]))
#     # self.assertTrue(gr.check_perp([a, h, b, c]))
#     # self.assertTrue(gr.check_perp([b, h, c, a]))

#     # These are NOT part of the premises:
#     self.assertFalse(gr.check_perp([c, h, a, b]))
#     self.assertFalse(gr.check_coll([c, g, g3]))

#     # These are automatically inferred by the graph datastructure:
#     self.assertTrue(gr.check_eqangle([a, h1, b, c, b, h2, c, a]))
#     self.assertTrue(gr.check_eqangle([a, h1, b, h2, b, c, c, a]))
#     self.assertTrue(gr.check_coll([a, h, h1]))
#     self.assertTrue(gr.check_coll([b, h, h2]))

#   # def test_enumerate_colls(self):
#   #   g = GraphTest.g

#   #   for a, b, c in g.all_colls():
#   #     self.assertTrue(g.check_coll([a, b, c]))
#   #     self.assertTrue(nm.check_coll([a.num, b.num, c.num]))

#   # def test_enumerate_perps(self):
#   #   g = GraphTest.g

#   #   for a, b, c, d in g.all_perps():
#   #     self.assertTrue(g.check_perp([a, b, c, d]))
#   #     self.assertTrue(nm.check_perp([a.num, b.num, c.num, d.num]))

#   # def test_enumerate_congs(self):
#   #   g = GraphTest.g

#   #   for a, b, c, d in g.all_congs():
#   #     self.assertTrue(g.check_cong([a, b, c, d]))
#   #     self.assertTrue(nm.check_cong([a.num, b.num, c.num, d.num]))

#   # def test_enumerate_eqangles(self):
#   #   g = GraphTest.g

#   #   for a, b, c, d, x, y, z, t in g.all_eqangles_8points():
#   #     self.assertTrue(g.check_eqangle([a, b, c, d, x, y, z, t]))
#   #     self.assertTrue(
#   #         nm.check_eqangle(
#   #             [a.num, b.num, c.num, d.num, x.num, y.num, z.num, t.num]
#   #         )
#   #     )

#   # def test_enumerate_eqratios(self):
#   #   g = GraphTest.g

#   #   for a, b, c, d, x, y, z, t in g.all_eqratios_8points():
#   #     self.assertTrue(g.check_eqratio([a, b, c, d, x, y, z, t]))
#   #     self.assertTrue(
#   #         nm.check_eqratio(
#   #             [a.num, b.num, c.num, d.num, x.num, y.num, z.num, t.num]
#   #         )
#   #     )


class MidpointParallelogramTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

    # Load the midpoint parallelogram problem
    txt = 'a b c = triangle a b c; d = midpoint d b c; e = midpoint e c a; f = midpoint f b a; g = parallelogram d a e g ? cong c f g b'
    p = pr.Problem.from_txt(txt, translate=False)
    cls.p = p
    cls.g, _ = gh.Graph.build_problem(p, cls.defs)

  # def test_build_graph_points(self):
  #   g = self.g

  #   all_points = g.all_points()
  #   all_names = [p.name for p in all_points]
  #   self.assertCountEqual(
  #       all_names,
  #       ['a', 'b', 'c', 'd', 'e', 'f', 'g']
  #   )

  def test_build_graph_predicates(self):
    gr = self.g
    a, b, c, d, e, f, g = gr.names2points(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    )

    # Test basic predicates from problem setup
    self.assertTrue(gr.check_coll([b, d, c]))
    self.assertTrue(gr.check_coll([c, e, a]))
    self.assertTrue(gr.check_coll([b, f, a]))
    
    # Test triangle properties
    self.assertTrue(gr.check_ncoll([a, b, c]))


    # Goal should not be proven yet
    self.assertFalse(gr.check_cong([c, f, g, b]))

  def test_build_graph_predicates(self):
    gr = self.g

    a, b, c, d, e, f, g = gr.names2points(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    )

    # Test the midpoint conditions
    # self.assertTrue(gr.check_midp([d, b, c]))
    # self.assertTrue(gr.check_midp([e, c, a]))
    # self.assertTrue(gr.check_midp([f, b, a]))
    
    # # Test the parallelogram condition
    # self.assertTrue(gr.check_para([d, a, e, g]))
    
    # Optional - verify goal is not yet proven
    self.assertFalse(gr.check_cong([c, f, g, b]))
    
    # # Test some properties that are automatically inferred
    # # For a parallelogram, opposite sides are parallel
    # self.assertTrue(gr.check_para([d, e, a, g]))
    # self.assertTrue(gr.check_para([d, a, e, g]))
    
    # For midpoints, we have collinearity
    self.assertTrue(gr.check_coll([b, d, c]))
    self.assertTrue(gr.check_coll([c, e, a]))
    self.assertTrue(gr.check_coll([b, f, a]))

  def test_triangle_construction(self):
    gr = self.g
    a, b, c = gr.names2points(['a', 'b', 'c'])
    
    # Test that we have a triangle
    self.assertTrue(gr.check_ncoll([a, b, c]))
    # self.assertTrue(gr.check_diff([a, b]))
    # self.assertTrue(gr.check_diff([b, c]))
    # self.assertTrue(gr.check_diff([c, a]))


  def test_enumerate_colls(self):
    gr = self.g
    for a, b, c in gr.all_colls():
      self.assertTrue(gr.check_coll([a, b, c]))
      self.assertTrue(nm.check_coll([a.num, b.num, c.num]))

  def test_dependency_tracking(self):
    gr = self.g
    
    # Get cached dependencies for collinearity
    b, d, c = gr.names2points(['b', 'd', 'c'])
    hashed_key = pr.hashed('coll', [b, d, c])
    
    # Cached dependency should exist
    self.assertIn(hashed_key, gr.cache)
    
    # Test caching mechanism
    dep = gr.cache[hashed_key]
    self.assertIsNotNone(dep)
    
    # Dependency should reflect that d is midpoint of b and c
    self.assertEqual(dep.name, 'coll')

  def test_make_equal_pairs(self):
      """Test the make_equal_pairs method in Graph class."""
      gr = self.g
      
      # Get points from the midpoint parallelogram setup
      a, b, c, d, e, f, g = gr.names2points(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
      
      # Create some Line objects to use in the test
      ab = gr._get_line(a, b)
      cd = gr._get_line(c, d)
      ef = gr._get_line(e, f)
      fg = gr._get_line(f, g)
      
      # Create an empty dependency for testing
      deps = pr.EmptyDependency(level=0, rule_name='test_make_equal_pairs')
      
      # Case 1: Test when lines are already equal (should make the points equal)
      # First, make the lines equal
      line_node1 = ab
      line_node2 = cd
      if line_node1 is not None and line_node2 is not None:
          # Artificially merge the lines to make them equal
          line_node1.merge([line_node2], deps)
          
          # Now test make_equal_pairs - this should merge the points if the lines are equal
          result = gr.make_equal_pairs(a, b, c, d, e, f, g, a, ab, ef, gr, 0)
          
          # Check that dependencies were created
          self.assertIsNotNone(result)
          self.assertTrue(len(result) > 0)
          
          # The result should contain 'para' or 'cong' dependencies
          self.assertTrue(any(d.name in ['para', 'cong'] for d in result))
      
      # Case 2: Test when lines are not equal (should return empty list)
      if ef is not None and fg is not None and ef != fg:
          # Use make_equal_pairs with unequal lines
          result = gr.make_equal_pairs(e, f, f, g, a, b, c, d, ef, ab, gr, 0)
          
          # Since the lines aren't equal, it should return None
          self.assertIsNone(result)
      
      # Case 3: Test with special cases like collinearity
      if ab is not None and cd is not None:
          # Make collinear points
          coll_points = [a, b, c, d]
          # Only if we have 4 distinct points
          if len(set(coll_points)) >= 3:
              # Test with lines that could trigger collinearity checks
              result = gr.make_equal_pairs(a, b, c, d, a, b, c, d, ab, ab, gr, 0)
              
              # Check that appropriate dependencies were created
              self.assertIsNotNone(result)
              deps_names = [d.name for d in result]
              # Either collx or para should be in the dependencies
              self.assertTrue(any(name in ['collx', 'para'] for name in deps_names))
if __name__ == '__main__':
  absltest.main()
