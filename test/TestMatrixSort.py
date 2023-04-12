import unittest
import numpy as np
from matrix_test.test004 import *
from matrix_test.MatrixSort import generate_random_matrix


class TestMatrixSort(unittest.TestCase):

    def test_1(self):
        # 测试矩阵生成的形状是否为(n, n)。
        n = 6
        matrix = generate_random_matrix(6)
        print(get_goal_positions(n))


if __name__ == '__main__':
    unittest.main()

