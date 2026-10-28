"""Module 5 zyLab starter: inverses and block matrices."""


def validate_matrix(matrix):
    """Raise ValueError if matrix is empty or has uneven row lengths."""
    if len(matrix) == 0:
        raise ValueError("matrix must have at least one row")

    width = len(matrix[0])
    if width == 0:
        raise ValueError("matrix rows must have at least one entry")

    for row in matrix:
        if len(row) != width:
            raise ValueError("all matrix rows must have the same length")


def matrix_size(matrix):
    """Return the size of a matrix as (rows, columns)."""
    validate_matrix(matrix)
    return len(matrix), len(matrix[0])


def identity_matrix(n):
    """Return the n by n identity matrix."""
    if n <= 0:
        raise ValueError("identity matrix size must be positive")

    result = []
    for row in range(n):
        new_row = []
        for col in range(n):
            if row == col:
                new_row.append(1)
            else:
                new_row.append(0)
        result.append(new_row)
    return result


def matrix_multiply(A, B):
    """Return the matrix product AB."""
    rows_A, cols_A = matrix_size(A)
    rows_B, cols_B = matrix_size(B)

    if cols_A != rows_B:
        raise ValueError("number of columns of A must equal number of rows of B")

    result = []
    for row in range(rows_A):
        new_row = []
        for col in range(cols_B):
            total = 0
            for shared in range(cols_A):
                total += A[row][shared] * B[shared][col]
            new_row.append(total)
        result.append(new_row)
    return result


def matrix_add(A, B):
    """Return the matrix sum A + B."""
    rows_A, cols_A = matrix_size(A)
    rows_B, cols_B = matrix_size(B)

    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError("matrices must have the same size")

    result = []
    for row in range(rows_A):
        new_row = []
        for col in range(cols_A):
            new_row.append(A[row][col] + B[row][col])
        result.append(new_row)
    return result


def matrix_vector_product(A, vector):
    """Return A times vector."""
    rows, cols = matrix_size(A)
    if len(vector) != cols:
        raise ValueError("vector length must match number of matrix columns")

    result = []
    for row in range(rows):
        total = 0
        for col in range(cols):
            total += A[row][col] * vector[col]
        result.append(total)
    return result


def matrices_close(A, B, tolerance=1e-9):
    """Return True when corresponding entries are within tolerance."""
    rows_A, cols_A = matrix_size(A)
    rows_B, cols_B = matrix_size(B)

    if rows_A != rows_B or cols_A != cols_B:
        return False

    for row in range(rows_A):
        for col in range(cols_A):
            if abs(A[row][col] - B[row][col]) > tolerance:
                return False
    return True


def validate_2x2_block_matrix(blocks):
    """Raise ValueError unless blocks is a 2 by 2 list of matrix blocks."""
    if len(blocks) != 2 or len(blocks[0]) != 2 or len(blocks[1]) != 2:
        raise ValueError("block matrix must contain 2 by 2 blocks")

    for block_row in blocks:
        for block in block_row:
            validate_matrix(block)


def block_matrix_multiply_2x2(A_blocks, B_blocks):
    """Multiply two matrices partitioned into compatible 2 by 2 blocks."""
    validate_2x2_block_matrix(A_blocks)
    validate_2x2_block_matrix(B_blocks)

    A11, A12 = A_blocks[0]
    A21, A22 = A_blocks[1]
    B11, B12 = B_blocks[0]
    B21, B22 = B_blocks[1]

    C11 = matrix_add(matrix_multiply(A11, B11), matrix_multiply(A12, B21))
    C12 = matrix_add(matrix_multiply(A11, B12), matrix_multiply(A12, B22))
    C21 = matrix_add(matrix_multiply(A21, B11), matrix_multiply(A22, B21))
    # Level 1: Find and fix the incorrect block in the C22 formula.
    C22 = matrix_add(matrix_multiply(A21, B12), matrix_multiply(A22, B22))

    return [
        [C11, C12],
        [C21, C22],
    ]


def determinant_2x2(matrix):
    """Return the determinant of a 2 by 2 matrix."""
    rows, cols = matrix_size(matrix)
    if rows != 2 or cols != 2:
        raise ValueError("matrix must be 2 by 2")

    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c


def are_inverses(A, B, tolerance=1e-9):
    """Return True when A and B satisfy both inverse identities."""
    rows_A, cols_A = matrix_size(A)
    rows_B, cols_B = matrix_size(B)

    if cols_A != rows_B or cols_B != rows_A:
        return False

    product_AB = matrix_multiply(A, B)
    product_BA = matrix_multiply(B, A)
    AB_is_identity = matrices_close(
        product_AB, identity_matrix(rows_A), tolerance
    )
    BA_is_identity = matrices_close(
        product_BA, identity_matrix(rows_B), tolerance
    )

    # Level 1: Find and fix the Boolean error on this line.
    return AB_is_identity and BA_is_identity


def inverse_2x2(matrix, tolerance=1e-12):
    """Return the inverse of an invertible 2 by 2 matrix."""
    # Level 2:
    # 1. Check that matrix is 2 by 2.
    rows, cols = matrix_size(matrix)
    if rows != 2 or cols != 2:
        raise ValueError("matrix must be 2 by 2")

    a, b = matrix[0]
    c, d = matrix[1]
    
    # 2. Compute ad - bc and reject a zero determinant.
    det = a * d - b * c
    if abs(det) <= tolerance:
        raise ValueError("matrix is singular")

    # 3. Swap the diagonal entries, negate the off-diagonal entries,
    #    divide every result by the determinant, and return the matrix.
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]


def copy_matrix(matrix):
    """Return a row-by-row copy of a matrix."""
    validate_matrix(matrix)
    return [row[:] for row in matrix]


def swap_rows(matrix, row_a, row_b):
    """Return a copy of matrix with two rows switched."""
    result = copy_matrix(matrix)
    result[row_a], result[row_b] = result[row_b], result[row_a]
    return result


def find_pivot_row(matrix, start_row, pivot_col, tolerance=1e-12):
    """Return the first row at or below start_row with a nonzero pivot."""
    for row in range(start_row, len(matrix)):
        if abs(matrix[row][pivot_col]) > tolerance:
            return row
    return None


def normalize_pivot_row(matrix, pivot_row, pivot_col):
    """Return a copy with the selected pivot scaled to 1."""
    result = [[float(value) for value in row] for row in matrix]
    pivot = result[pivot_row][pivot_col]

    for col in range(len(result[pivot_row])):
        result[pivot_row][col] = result[pivot_row][col] / pivot

    return result


def eliminate_column(matrix, pivot_row, pivot_col, tolerance=1e-12):
    """Return a copy with zeros above and below the selected pivot."""
    result = [[float(value) for value in row] for row in matrix]
    col_count = len(result[0])

    for row in range(len(result)):
        if row == pivot_row:
            continue

        factor = result[row][pivot_col]
        if abs(factor) <= tolerance:
            continue

        for col in range(col_count):
            result[row][col] -= factor * result[pivot_row][col]

    return result


def gauss_jordan_elimination(matrix):
    """Return the reduced-row-echelon-form copy of a matrix."""
    validate_matrix(matrix)
    result = [[float(value) for value in row] for row in matrix]
    row_count = len(result)
    col_count = len(result[0])
    pivot_row = 0

    for pivot_col in range(col_count):
        candidate = find_pivot_row(result, pivot_row, pivot_col)
        if candidate is None:
            continue

        if candidate != pivot_row:
            result = swap_rows(result, pivot_row, candidate)

        result = normalize_pivot_row(result, pivot_row, pivot_col)
        result = eliminate_column(result, pivot_row, pivot_col)

        pivot_row += 1
        if pivot_row == row_count:
            break

    return result


def inverse_via_gauss_jordan(matrix, tolerance=1e-9):
    """Return the inverse of a square matrix using Gauss-Jordan elimination."""
    # Level 3:
    # 1. Verify that matrix is square.
    rows, cols = matrix_size(matrix)
    if rows != cols:
        raise ValueError("matrix must be square")

    n = rows
    I_n = identity_matrix(n)
    
    # 2. Build the augmented matrix [matrix | identity].
    augmented = []
    for i in range(n):
        augmented.append(matrix[i] + I_n[i])

    # 3. Call the provided gauss_jordan_elimination helper once.
    reduced = gauss_jordan_elimination(augmented)

    # 4. Verify that the left block became the identity matrix.
    # 5. Return the right block, which is the inverse.
    inverse = []
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            if abs(reduced[i][j] - expected) > tolerance:
                raise ValueError("matrix is singular")
        
        inverse.append(reduced[i][n:])

    return inverse


def main():
    print("Level 1")
    A = [
        [1, 3],
        [2, 7],
    ]
    B = [
        [7, -3],
        [-2, 1],
    ]
    print("Inverse pair:", are_inverses(A, B))
    print("One-sided pair:", are_inverses([[1, 0]], [[1], [0]]))

    A_blocks = [
        [[[1]], [[1]]],
        [[[0]], [[1]]],
    ]
    B_blocks = [
        [[[1]], [[2]]],
        [[[3]], [[4]]],
    ]
    print("Block product:", block_matrix_multiply_2x2(A_blocks, B_blocks))

    print()
    print("Level 2")
    C = [
        [3, 5],
        [2, 4],
    ]
    print("Inverse:", inverse_2x2(C))

    print()
    print("Level 3")
    D = [
        [1, 0, 0],
        [2, 2, 0],
        [4, 4, 4],
    ]
    print("Inverse:", inverse_via_gauss_jordan(D))


if __name__ == "__main__":
    main()