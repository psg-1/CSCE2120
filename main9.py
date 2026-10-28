def validate_square_matrix(matrix):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a nonempty list of rows")
    size = len(matrix)
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("matrix must be square")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")
    return size


def det2(matrix):
    if validate_square_matrix(matrix) != 2:
        raise ValueError("det2 requires a 2-by-2 matrix")
    return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])


def minor_matrix(matrix, row, col):
    size = validate_square_matrix(matrix)
    valid_indices = (isinstance(row, int) and not isinstance(row, bool)
                     and isinstance(col, int) and not isinstance(col, bool)
                     and 0 <= row < size and 0 <= col < size)
    if size < 2 or not valid_indices:
        raise ValueError("minor requires valid indices in a matrix of size at least 2")
    return [[matrix[r][c] for c in range(size) if c != col]
            for r in range(size) if r != row]


def det3(matrix):
    if validate_square_matrix(matrix) != 3:
        raise ValueError("det3 requires a 3-by-3 matrix")
    total = 0.0
    for col in range(3):
        # Cofactor expansion along row 0: signs alternate +, -, +.
        sign = 1 if col % 2 == 0 else -1
        smaller = minor_matrix(matrix, 0, col)
        total += sign * matrix[0][col] * det2(smaller)
    return total


def determinant_cofactor(matrix):
    size = validate_square_matrix(matrix)
    if size == 1:
        return float(matrix[0][0])
    if size == 2:
        return det2(matrix)
    if size == 3:
        return det3(matrix)

    total = 0.0
    for col in range(size):
        sign = 1 if col % 2 == 0 else -1
        smaller = minor_matrix(matrix, 0, col)
        total += sign * matrix[0][col] * determinant_cofactor(smaller)
    return total


def is_invertible(matrix, tolerance=1e-9):
    if (isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0):
        raise ValueError("tolerance must be a nonnegative real number")
    return abs(determinant_cofactor(matrix)) > tolerance


def triangle_report(points, tolerance=1e-9):
    if not isinstance(points, list) or len(points) != 3:
        raise ValueError("points must be a list of exactly three points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("each point must be a two-number list")
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("point coordinates must be real numbers")
    if (isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0):
        raise ValueError("tolerance must be a nonnegative real number")

    augmented = [[point[0], point[1], 1] for point in points]
    signed_twice_area = determinant_cofactor(augmented)
    collinear = abs(signed_twice_area) <= tolerance
    if collinear:
        orientation = "collinear"
    elif signed_twice_area > 0:
        orientation = "counterclockwise"
    else:
        orientation = "clockwise"

    return {
        "area": abs(signed_twice_area) / 2.0,
        "orientation": orientation,
        "collinear": collinear,
    }


def pivot_column_indices(matrix, tolerance=1e-9):
    """Return the zero-based pivot-column indices using Gaussian elimination."""
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a nonempty rectangular list")
    if not isinstance(matrix[0], list) or not matrix[0]:
        raise ValueError("matrix rows must be nonempty lists")
    row_count = len(matrix)
    column_count = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != column_count:
            raise ValueError("matrix must be rectangular")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")
    if (isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0):
        raise ValueError("tolerance must be a nonnegative real number")

    work = [[float(value) for value in row] for row in matrix]
    pivots = []
    pivot_row = 0
    for column in range(column_count):
        if pivot_row == row_count:
            break
        best_row = max(range(pivot_row, row_count),
                       key=lambda r: abs(work[r][column]))
        if abs(work[best_row][column]) <= tolerance:
            continue
        work[pivot_row], work[best_row] = work[best_row], work[pivot_row]
        pivot_value = work[pivot_row][column]
        for row_index in range(pivot_row + 1, row_count):
            factor = work[row_index][column] / pivot_value
            for col_index in range(column, column_count):
                work[row_index][col_index] -= factor * work[pivot_row][col_index]
            work[row_index][column] = 0.0
        pivots.append(column)
        pivot_row += 1
    return pivots


def rref(matrix, tolerance=1e-9):
    """Return the reduced row echelon form (RREF) of a matrix."""
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix must be a nonempty rectangular list")
    if not isinstance(matrix[0], list) or not matrix[0]:
        raise ValueError("matrix rows must be nonempty lists")
    row_count = len(matrix)
    column_count = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != column_count:
            raise ValueError("matrix must be rectangular")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")
    if (isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0):
        raise ValueError("tolerance must be a nonnegative real number")

    work = [[float(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(column_count):
        if pivot_row == row_count:
            break
        best_row = max(range(pivot_row, row_count),
                       key=lambda r: abs(work[r][column]))
        if abs(work[best_row][column]) <= tolerance:
            continue
        work[pivot_row], work[best_row] = work[best_row], work[pivot_row]
        pivot_value = work[pivot_row][column]
        for col_index in range(column_count):
            work[pivot_row][col_index] /= pivot_value
        work[pivot_row][column] = 1.0
        for row_index in range(row_count):
            if row_index != pivot_row:
                factor = work[row_index][column]
                for col_index in range(column_count):
                    work[row_index][col_index] -= factor * work[pivot_row][col_index]
                work[row_index][column] = 0.0
        pivot_row += 1

    for r in range(row_count):
        for c in range(column_count):
            if abs(work[r][c]) <= tolerance:
                work[r][c] = 0.0
    return work


if __name__ == "__main__":
    matrix_3x3 = [[2, 1, 0], [0, 3, 1], [1, 0, 2]]
    matrix_4x4 = [[1, 2, 0, 0], [0, 2, 1, 0],
                  [3, 0, 1, 1], [0, 0, 0, 2]]
    print("Level 1 determinant:", det3(matrix_3x3))
    print("Level 2 determinant:", determinant_cofactor(matrix_4x4))
    print("Level 2 invertible:", is_invertible(matrix_4x4))
    print("Level 3:", triangle_report([[1, 1], [4, 1], [2, 3]]))