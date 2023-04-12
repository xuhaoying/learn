import unittest
import numpy as np
from matrix_test.MatrixSort import MatrixSort, generate_random_matrix


class TestMatrixSort(unittest.TestCase):

    def test_matrix_shape(self):
        # 测试矩阵生成的形状是否为(n, n)。
        n = 6
        matrix = generate_random_matrix()
        self.assertEqual(matrix.shape, (n, n))

    def test_matrix_values(self):
        n = 6
        matrix = generate_random_matrix()
        expected_seq = np.random.permutation(n * n) + 1
        expected_matrix = expected_seq.reshape((n, n))
        # 比较生成的矩阵中的元素是否与预期序列中的元素相同
        self.assertTrue(np.array_equal(np.sort(matrix.ravel()), np.arange(1, n * n + 1)))
        self.assertFalse(np.array_equal(matrix, expected_matrix))

    def test_matrix_sort_001(self):
        n = 4
        ms = MatrixSort(4)
        ms.matrix = np.arange(1, 17).reshape(4, 4)
        print("origin matrix:\n", ms.matrix)
        # print("expectation matrix:\n", ms.expectation_matrix)
        # ms.move_left(1)
        # print("after move_left matrix:\n", ms.matrix)
        ms.move_up(1)
        print("after move_up matrix:\n", ms.matrix)

    def test_move(self):
        ms = MatrixSort(4)
        ms.matrix = np.arange(1, 17).reshape(4, 4)
        print("origin matrix:\n", ms.matrix)
        ms.move(1, "L")
        print("after move_up matrix:\n", ms.matrix)

    def test_manhattan_distance(self):
        ms = MatrixSort(4)
        print(ms.manhattan_distance())

    def test_matrix(self):
        matrix = np.arange(1, 17).reshape(4, 4)
        print(np.roll(matrix, -1, axis=1))


if __name__ == '__main__':
    unittest.main()

