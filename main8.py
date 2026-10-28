"""Module 8 zyLab starter: coordinate changes and four subspaces."""


def validate_matrix(matrix):
    """Return (rows, columns) for a nonempty real rectangular matrix."""
    if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list) or not matrix[0]:
        raise ValueError("matrix must be a nonempty list of nonempty rows")
    width = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("matrix rows must have the same length")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")
    return len(matrix), width


def validate_vector(vector, dimension):
    """Validate that vector is a list of real numbers of the given dimension."""
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError("vector has the wrong dimension")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("vector entries must be real numbers")


def linear_combination(first, second):
    """Return a linear combination of vectors (provided Module 7 helper)."""
    if not isinstance(first, list) or not isinstance(second, list):
        raise ValueError("scalars and vectors must be lists")
    if not first or not second or len(first) != len(second):
        raise ValueError("scalars and vectors must have the same positive length")

    if isinstance(first[0], list) and not isinstance(second[0], list):
        vectors, scalars = first, second
    else:
        scalars, vectors = first, second

    dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
    if dimension == 0:
        raise ValueError("vectors must be nonempty lists")
    for vector in vectors:
        validate_vector(vector, dimension)
    for scalar in scalars:
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise ValueError("scalars must be real numbers")

    result = [0.0] * dimension
    for i in range(len(scalars)):
        for j in range(dimension):
            result[j] += float(scalars[i]) * float(vectors[i][j])
    return result


def select_vectors(vectors, indices):
    """Return copies of the vectors at the given zero-based indices."""
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("at least one vector is required")
    dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
    if dimension == 0:
        raise ValueError("vectors must be nonempty lists")
    for vector in vectors:
        validate_vector(vector, dimension)
    if not isinstance(indices, list):
        raise ValueError("indices must be a list")
    selected = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(vectors):
            raise ValueError("index out of range")
        selected.append([float(value) for value in vectors[index]])
    return selected


def matrix_from_columns(vectors):
    """Place each input vector in one matrix column (Module 7 helper)."""
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("at least one vector is required")
    dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
    if dimension == 0:
        raise ValueError("vectors must be nonempty lists")
    for vector in vectors:
        validate_vector(vector, dimension)
    return [[float(vectors[col][row]) for col in range(len(vectors))]
            for row in range(dimension)]


def rref(matrix, tolerance=1e-10):
    """Return RREF without changing the input (provided Module 3 helper)."""
    rows, cols = validate_matrix(matrix)
    reduced = [[float(value) for value in row] for row in matrix]
    pivot_row = 0
    for pivot_col in range(cols):
        candidate = next((r for r in range(pivot_row, rows)
                          if abs(reduced[r][pivot_col]) > tolerance), None)
        if candidate is None:
            continue
        reduced[pivot_row], reduced[candidate] = reduced[candidate], reduced[pivot_row]
        pivot = reduced[pivot_row][pivot_col]
        reduced[pivot_row] = [value / pivot for value in reduced[pivot_row]]
        for row in range(rows):
            if row != pivot_row:
                factor = reduced[row][pivot_col]
                reduced[row] = [reduced[row][col] - factor * reduced[pivot_row][col]
                                for col in range(cols)]
        pivot_row += 1
        if pivot_row == rows:
            break
    return [[0.0 if abs(value) <= tolerance else value for value in row]
            for row in reduced]


def pivot_column_indices(reduced, tolerance=1e-10):
    """Return zero-based pivot-column indices (provided Module 7 helper)."""
    validate_matrix(reduced)
    pivots = []
    for row in reduced:
        for col, value in enumerate(row):
            if abs(value) > tolerance:
                pivots.append(col)
                break
    return pivots


def coordinates_in_basis(basis, vector):
    """Return coordinates of vector relative to an ordered basis."""
    basis_matrix = matrix_from_columns(basis)
    rows, cols = validate_matrix(basis_matrix)
    if rows != cols:
        raise ValueError("basis must contain one vector per dimension")
    validate_vector(vector, rows)
    augmented = [basis_matrix[row][:] + [float(vector[row])]
                 for row in range(rows)]
    reduced = rref(augmented)
    if any(abs(reduced[row][col] - float(row == col)) > 1e-10
           for row in range(rows) for col in range(cols)):
        raise ValueError("basis vectors must be linearly independent")

    # Level 1: Debug which RREF column holds the coordinate coefficients.
    return [reduced[row][-1] for row in range(rows)]


def vector_from_coordinates(basis, coordinates):
    """Rebuild a standard-coordinate vector from basis coordinates (provided)."""
    basis_matrix = matrix_from_columns(basis)
    rows, cols = validate_matrix(basis_matrix)
    if rows != cols or pivot_column_indices(rref(basis_matrix)) != list(range(cols)):
        raise ValueError("basis vectors must form a basis")
    validate_vector(coordinates, cols)
    return linear_combination(coordinates, basis)


def change_coordinates(from_basis, to_basis, coordinates):
    """Convert coordinates from one ordered basis to another."""
    # Level 1, Part B: Rebuild the vector, then find its new coordinates.
    standard_vector = vector_from_coordinates(from_basis, coordinates)
    return coordinates_in_basis(to_basis, standard_vector)


def null_space_basis(matrix):
    """Return one null-space basis vector for each free variable."""
    rows, cols = validate_matrix(matrix)
    reduced = rref(matrix)
    pivots = pivot_column_indices(reduced)
    free_columns = [col for col in range(cols) if col not in pivots]
    basis = []

    for free_col in free_columns:
        vector = [0.0] * cols
        vector[free_col] = 1.0
        # Level 2: Set pivot components using the RREF equations.
        # Each pivot row gives x_pivot = -RREF[pivot_row][free_col].
        for pivot_row, pivot_col in enumerate(pivots):
            vector[pivot_col] = -reduced[pivot_row][free_col]
        basis.append(vector)

    return basis


def transpose(matrix):
    """Return the transpose of a nonempty rectangular matrix (provided)."""
    rows, cols = validate_matrix(matrix)
    return [[float(matrix[row][col]) for row in range(rows)]
            for col in range(cols)]


def four_subspace_bases(matrix):
    """Return bases for the column, null, row, and left null spaces."""
    rows, cols = validate_matrix(matrix)
    reduced = rref(matrix)
    pivots = pivot_column_indices(reduced)

    column_basis = []
    for pivot_col in pivots:
        # Level 3: Append the matching column from the original matrix.
        column_basis.append([float(matrix[row][pivot_col]) for row in range(rows)])

    row_basis = []
    for row in reduced:
        # Level 3: Append each nonzero RREF row.
        if any(abs(val) > 1e-10 for val in row):
            row_basis.append(row[:])

    # Level 3: Reuse Level 2 for both null spaces.
    null_basis = null_space_basis(matrix)
    left_null_basis = null_space_basis(transpose(matrix))

    return {"column": column_basis, "null": null_basis,
            "row": row_basis, "left_null": left_null_basis}


def main():
    basis = [[1, 1], [1, -1]]
    new_basis = [[1, 0], [1, 1]]
    print("Level 1A:", coordinates_in_basis(basis, [4, 2]))
    print("Level 1B:", change_coordinates(basis, new_basis, [3, 1]))
    matrix = [[1, 2, 0], [0, 1, 1], [1, 3, 1]]
    print("Level 2:", null_space_basis(matrix))
    print("Level 3:", four_subspace_bases(matrix))


if __name__ == "__main__":
    main()