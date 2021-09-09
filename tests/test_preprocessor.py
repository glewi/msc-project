import pytest


class TestPreprocessor:
    def test_simple_dataframe_preprocess(self):
        assert False

    def test_dataframe_preprocess(self):
        assert False

    def test_simple_string_preprocess(self):
        assert False

    def test_string_preprocess(self, s):
        assert False

    @pytest.mark.parametrize("num, output", [(1, 11), (2, 22), (3, 33), (4, 44)])
    def test_multiplication_11(self, num, output):
        assert 11 * num == output
