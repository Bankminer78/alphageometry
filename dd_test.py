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

"""Unit tests for dd."""
import unittest

from absl.testing import absltest
import dd
import graph as gh
import problem as pr


MAX_LEVEL = 10

class DDTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_imo_2022_p4_should_succeed(self):
    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d = midpoint d b c; e = midpoint e c a; f = midpoint f b a; g = parallelogram d a e g ? cong c f g b', translate=False)
    g, _ = gh.Graph.build_problem(p, DDTest.defs)
    goal_args = g.names2nodes(p.goal.args)

    success = False
    for level in range(MAX_LEVEL):
      added, _, _, _ = dd.bfs_one_level(g, DDTest.rules, level, p)
      if g.check(p.goal.name, goal_args):
        success = True
        break
      if not added:  # saturated
        break

    self.assertTrue(success)

if __name__ == '__main__':
  absltest.main()
