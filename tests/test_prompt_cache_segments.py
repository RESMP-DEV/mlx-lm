import unittest

from mlx_lm.server import _system_prefix_end


class TestSystemPrefixEnd(unittest.TestCase):
    def test_exact_prefix_uses_prefix_length(self):
        sys_tokens = [1, 2, 3, 4]
        prompt = [1, 2, 3, 4, 5, 6]
        self.assertEqual(_system_prefix_end(sys_tokens, prompt), 4)

    def test_mismatch_uses_first_difference(self):
        sys_tokens = [1, 2, 9]
        prompt = [1, 2, 3, 4]
        self.assertEqual(_system_prefix_end(sys_tokens, prompt), 2)

    def test_identical_sequences_have_no_boundary(self):
        tokens = [1, 2, 3]
        self.assertEqual(_system_prefix_end(tokens, list(tokens)), 0)

    def test_empty_system_render_has_no_boundary(self):
        self.assertEqual(_system_prefix_end([], [1, 2, 3]), 0)

    def test_longer_system_render_clamps_to_prompt(self):
        self.assertEqual(_system_prefix_end([1, 2, 3, 4], [1, 2, 3]), 0)


if __name__ == "__main__":
    unittest.main()
