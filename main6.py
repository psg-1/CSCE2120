"""Module 6 zyLab starter: LU and Cholesky for a weather model."""

from math import sqrt


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
    if not isinstance(n, int) or n <= 0:
        raise ValueError("identity matrix size must be a positive integer")

    result = []
    for row in range(n):
        new_row = []
        for col in range(n):
            if row == col:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        result.append(new_row)
    return result


def transpose(matrix):
    """Return the transpose of a matrix."""
    rows, cols = matrix_size(matrix)
    result = []
    for col in range(cols):
        new_row = []
        for row in range(rows):
            new_row.append(matrix[row][col])
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
            total = 0.0
            for shared in range(cols_A):
                total += A[row][shared] * B[shared][col]
            new_row.append(total)
        result.append(new_row)
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


def vectors_close(u, v, tolerance=1e-9):
    """Return True when corresponding vector entries are within tolerance."""
    if len(u) != len(v):
        return False
    for index in range(len(u)):
        if abs(u[index] - v[index]) > tolerance:
            return False
    return True


def elimination_multiplier(target_entry, pivot_entry, tolerance=1e-12):
    """Return the multiplier used to eliminate one entry below a pivot."""
    if abs(pivot_entry) <= tolerance:
        raise ValueError("pivot entry must be nonzero")

    # Level 1: Fix the reversed division on this line.
    return target_entry / pivot_entry


def eliminate_row(target_row, pivot_row, multiplier):
    """Return target_row minus multiplier times pivot_row.

    This Module 3 review function is provided and is not a student TODO.
    """
    if len(target_row) == 0 or len(target_row) != len(pivot_row):
        raise ValueError("rows must have the same positive length")

    new_row = []
    for col in range(len(target_row)):
        updated_entry = target_row[col] - multiplier * pivot_row[col]
        new_row.append(updated_entry)
    return new_row


def eliminate_below_pivot(L, U, pivot_col, tolerance=1e-12):
    """Return copies of L and U after eliminating below one pivot."""
    rows_U, cols_U = matrix_size(U)
    rows_L, cols_L = matrix_size(L)

    if rows_U != cols_U or rows_L != rows_U or cols_L != cols_U:
        raise ValueError("L and U must be square matrices of the same size")
    if pivot_col < 0 or pivot_col >= rows_U - 1:
        raise ValueError("pivot column is out of range")

    new_U = [[float(value) for value in row] for row in U]
    new_L = [[float(value) for value in row] for row in L]
    pivot_entry = new_U[pivot_col][pivot_col]

    if abs(pivot_entry) <= tolerance:
        raise ValueError("basic LU cannot continue without a row swap")

    for target_row in range(pivot_col + 1, rows_U):
        # Level 2 TODO 1: Compute the multiplier with the Level 1 helper.
        multiplier = elimination_multiplier(new_U[target_row][pivot_col], pivot_entry)

        # Level 2 TODO 2: Store the multiplier in L.
        new_L[target_row][pivot_col] = multiplier

        # Level 2 TODO 3: Update this row with the provided row helper.
        new_U[target_row] = eliminate_row(new_U[target_row], new_U[pivot_col], multiplier)

        if abs(new_U[target_row][pivot_col]) <= tolerance:
            new_U[target_row][pivot_col] = 0.0

    return new_L, new_U


def lu_decomposition(matrix, tolerance=1e-12):
    """Return L and U for a square matrix that needs no row swaps."""
    rows, cols = matrix_size(matrix)
    if rows != cols:
        raise ValueError("LU decomposition requires a square matrix")

    U = [[float(value) for value in row] for row in matrix]
    L = identity_matrix(rows)

    for pivot_col in range(rows - 1):
        # Level 2 TODO 4: Process this pivot by calling the helper above.
        L, U = eliminate_below_pivot(L, U, pivot_col)

    if abs(U[rows - 1][rows - 1]) <= tolerance:
        raise ValueError("matrix is singular")

    return L, U


def forward_substitution(L, b, tolerance=1e-12):
    """Solve Ly=b for a lower-triangular matrix L."""
    rows, cols = matrix_size(L)
    if rows != cols or len(b) != rows:
        raise ValueError("matrix and vector dimensions must agree")

    y = [0.0] * rows
    for row in range(rows):
        known_sum = 0.0
        for col in range(row):
            known_sum += L[row][col] * y[col]
        if abs(L[row][row]) <= tolerance:
            raise ValueError("triangular matrix has a zero diagonal entry")
        y[row] = (b[row] - known_sum) / L[row][row]
    return y


def back_substitution(U, b, tolerance=1e-12):
    """Solve Ux=b for an upper-triangular matrix U."""
    rows, cols = matrix_size(U)
    if rows != cols or len(b) != rows:
        raise ValueError("matrix and vector dimensions must agree")

    x = [0.0] * rows
    for row in range(rows - 1, -1, -1):
        known_sum = 0.0
        for col in range(row + 1, cols):
            known_sum += U[row][col] * x[col]
        if abs(U[row][row]) <= tolerance:
            raise ValueError("triangular matrix has a zero diagonal entry")
        x[row] = (b[row] - known_sum) / U[row][row]
    return x


def cholesky_decomposition(matrix, tolerance=1e-12):
    """Return C such that matrix = C C^T for an SPD matrix.

    This instructor-provided implementation is not a student TODO.
    """
    rows, cols = matrix_size(matrix)
    if rows != cols:
        raise ValueError("Cholesky decomposition requires a square matrix")

    for row in range(rows):
        for col in range(rows):
            if abs(matrix[row][col] - matrix[col][row]) > tolerance:
                raise ValueError("Cholesky decomposition requires symmetry")

    C = []
    for row in range(rows):
        C.append([0.0] * rows)

    for row in range(rows):
        for col in range(row + 1):
            product_sum = 0.0
            for shared in range(col):
                product_sum += C[row][shared] * C[col][shared]

            if row == col:
                diagonal_value = matrix[row][row] - product_sum
                if diagonal_value <= tolerance:
                    raise ValueError(
                        "Cholesky decomposition requires positive definiteness"
                    )
                C[row][col] = sqrt(diagonal_value)
            else:
                C[row][col] = (
                    matrix[row][col] - product_sum
                ) / C[col][col]

    return C


def solve_with_lu(matrix, b):
    """Solve matrix x=b using the student-completed LU functions."""
    L, U = lu_decomposition(matrix)
    y = forward_substitution(L, b)
    return back_substitution(U, y)


def solve_with_cholesky(matrix, b):
    """Solve matrix x=b using the provided Cholesky function."""
    C = cholesky_decomposition(matrix)
    y = forward_substitution(C, b)
    return back_substitution(transpose(C), y)


def graph_to_weighted_adjacency_matrix(node_count, weighted_edges):
    """Return a symmetric weighted adjacency matrix."""
    if not isinstance(node_count, int) or node_count <= 0:
        raise ValueError("node_count must be a positive integer")

    adjacency = []
    for row in range(node_count):
        adjacency.append([0.0] * node_count)

    for edge in weighted_edges:
        if len(edge) != 3:
            raise ValueError("each edge must contain two nodes and one weight")
        first, second, weight = edge
        if not isinstance(first, int) or not isinstance(second, int):
            raise ValueError("node indices must be integers")
        if first < 0 or first >= node_count or second < 0 or second >= node_count:
            raise ValueError("node index is out of range")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("edge weights must be nonnegative numbers")

        adjacency[first][second] = float(weight)
        if first != second:
            # Level 1: Fix this line so an undirected edge has equal weights
            # in both symmetric positions.
            adjacency[second][first] = float(weight)

    return adjacency


def weather_transition_matrix():
    """Return the transition matrix for sunny, rainy, and cloudy states."""
    SUNNY = 0
    RAINY = 1
    CLOUDY = 2

    # Level 3 TODO 1: Replace the six placeholder weights using the weather
    # probabilities in Level3.md. List each symmetric weight only once.
    weighted_edges = [
        (SUNNY, SUNNY, 0.6),
        (RAINY, RAINY, 0.5),
        (CLOUDY, CLOUDY, 0.5),
        (SUNNY, RAINY, 0.2),
        (SUNNY, CLOUDY, 0.2),
        (RAINY, CLOUDY, 0.3),
    ]
    return graph_to_weighted_adjacency_matrix(3, weighted_edges)


def solve_weather_problem(next_distribution):
    """Recover the previous weather distribution using two factorizations."""
    if len(next_distribution) != 3:
        raise ValueError("weather distribution must contain three values")
    if any(value < 0 for value in next_distribution):
        raise ValueError("weather probabilities must be nonnegative")
    if abs(sum(next_distribution) - 1.0) > 1e-9:
        raise ValueError("weather probabilities must sum to 1")

    transition = weather_transition_matrix()

    # Level 3 TODO 2: Recover the previous distribution with LU.
    lu_previous = solve_with_lu(transition, next_distribution)

    # Level 3 TODO 3: Recover the previous distribution with Cholesky.
    cholesky_previous = solve_with_cholesky(transition, next_distribution)

    return transition, lu_previous, cholesky_previous


def main():
    print("Level 1")
    print("Multiplier:", elimination_multiplier(4, 2))
    sample_edges = [(0, 0, 0.6), (1, 1, 0.5), (0, 1, 0.2)]
    print(
        "Weighted adjacency:",
        graph_to_weighted_adjacency_matrix(2, sample_edges),
    )

    print()
    print("Level 2")
    sample = [
        [2, 1, 1],
        [4, -6, 0],
        [-2, 7, 2],
    ]
    L, U = lu_decomposition(sample)
    print("L:", L)
    print("U:", U)
    print("LU equals A:", matrices_close(matrix_multiply(L, U), sample))

    print()
    print("Level 3")
    tomorrow = [0.40, 0.31, 0.29]
    transition, lu_today, cholesky_today = solve_weather_problem(tomorrow)
    print("Weather transition matrix:", transition)
    print("Previous distribution from LU:", lu_today)
    print("Previous distribution from Cholesky:", cholesky_today)


if __name__ == "__main__":
    main()