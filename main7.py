"""Module 7 zyLab starter: span, independence, basis, and dimension."""


def validate_vector_set(vectors):
    """Return the common dimension of a nonempty set of numeric vectors."""
    if len(vectors) == 0:
        raise ValueError("vector set must contain at least one vector")

    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("vectors must contain at least one component")

    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("all vectors must have the same dimension")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("vector components must be real numbers")

    return dimension


def validate_vector(vector, expected_dimension, name="vector"):
    """Raise ValueError unless vector has the expected numeric components."""
    if len(vector) != expected_dimension:
        raise ValueError(name + " has the wrong dimension")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " components must be real numbers")


def validate_matrix(matrix):
    """Return matrix size after validating a nonempty numeric rectangle."""
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must have positive dimensions")

    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError("all matrix rows must have the same length")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("matrix entries must be real numbers")

    return len(matrix), width


def linear_combination(vectors, weights):
    """Return the weighted sum of a set of vectors."""
    dimension = validate_vector_set(vectors)
    if len(weights) != len(vectors):
        raise ValueError("one weight is required for each vector")
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weights must be real numbers")

    result = [0.0] * dimension
    for vector_index in range(len(vectors)):
        weight = weights[vector_index]
        vector = vectors[vector_index]
        for component in range(dimension):
            # Level 1: Fix the arithmetic operation on this line.
            result[component] += weight * vector[component]
    return result


def matrix_from_columns(vectors):
    """Return a matrix whose columns are the given vectors."""
    dimension = validate_vector_set(vectors)
    matrix = []
    for row in range(dimension):
        new_row = []
        for col in range(len(vectors)):
            # Level 1: Fix the reversed vector and component indices.
            new_row.append(float(vectors[col][row]))
        matrix.append(new_row)
    return matrix


def select_vectors(vectors, indices):
    """Return copies of the vectors at the selected zero-based indices."""
    validate_vector_set(vectors)
    selected = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("vector indices must be integers")
        if index < 0 or index >= len(vectors):
            raise ValueError("vector index is out of range")

        # Level 1: Append a copy of the vector at this index.
        selected.append(list(vectors[index]))

    return selected


def rref(matrix, tolerance=1e-10):
    """Return the reduced row echelon form of a matrix.

    This Module 3 helper is provided and is not a student TODO.
    """
    rows, cols = validate_matrix(matrix)
    reduced = [[float(value) for value in row] for row in matrix]
    pivot_row = 0

    for pivot_col in range(cols):
        candidate = None
        for row in range(pivot_row, rows):
            if abs(reduced[row][pivot_col]) > tolerance:
                candidate = row
                break

        if candidate is None:
            continue

        reduced[pivot_row], reduced[candidate] = (
            reduced[candidate],
            reduced[pivot_row],
        )

        pivot_value = reduced[pivot_row][pivot_col]
        for col in range(cols):
            reduced[pivot_row][col] /= pivot_value

        for row in range(rows):
            if row == pivot_row:
                continue
            factor = reduced[row][pivot_col]
            if abs(factor) <= tolerance:
                continue
            for col in range(cols):
                reduced[row][col] -= factor * reduced[pivot_row][col]

        for row in range(rows):
            for col in range(cols):
                if abs(reduced[row][col]) <= tolerance:
                    reduced[row][col] = 0.0

        pivot_row += 1
        if pivot_row == rows:
            break

    return reduced


def pivot_column_indices(reduced_matrix, tolerance=1e-10):
    """Return zero-based pivot-column indices from an echelon matrix."""
    rows, cols = validate_matrix(reduced_matrix)
    pivots = []
    for row in range(rows):
        for col in range(cols):
            if abs(reduced_matrix[row][col]) > tolerance:
                pivots.append(col)
                break
    return pivots


def is_in_span(vectors, target, tolerance=1e-10):
    """Return True when target belongs to the span of vectors."""
    dimension = validate_vector_set(vectors)
    validate_vector(target, dimension, "target")

    # Level 2 TODO 1: Build the coefficient matrix with vectors as columns.
    coefficient_matrix = matrix_from_columns(vectors)

    # Level 2 TODO 2: Append target as the augmented column.
    augmented = []
    for i in range(dimension):
        augmented.append(coefficient_matrix[i] + [float(target[i])])

    # Level 2 TODO 3: Row reduce and reject an inconsistent row.
    reduced = rref(augmented, tolerance)
    cols = len(reduced[0])
    
    for row in reduced:
        all_zeros = True
        # Check coefficients only (excluding the augmented column)
        for j in range(cols - 1):
            if abs(row[j]) > tolerance:
                all_zeros = False
                break
        
        # If coefficients are all 0 but the augmented value is non-zero, it's inconsistent
        if all_zeros and abs(row[-1]) > tolerance:
            return False

    return True


def is_linearly_independent(vectors):
    """Return True when the vector set is linearly independent."""
    validate_vector_set(vectors)

    # Level 2 TODO 4: Compare the number of pivot columns with the number
    # of vectors.
    matrix = matrix_from_columns(vectors)
    reduced = rref(matrix)
    pivots = pivot_column_indices(reduced)
    
    return len(pivots) == len(vectors)


def basis_indices(vectors):
    """Return the original vector indices that form a basis for the span."""
    validate_vector_set(vectors)

    # Level 2 TODO 5: Build the column matrix and reduce it.
    matrix = matrix_from_columns(vectors)
    reduced = rref(matrix)

    # Level 2 TODO 6: Return the pivot-column indices.
    return pivot_column_indices(reduced)


def basis_from_spanning_set(vectors):
    """Return original vectors at the pivot-column positions."""
    validate_vector_set(vectors)
    return select_vectors(vectors, basis_indices(vectors))


def subspace_dimension(vectors):
    """Return the dimension of the subspace spanned by vectors."""
    return len(basis_indices(vectors))


def main():
    print("Level 1")
    print(
        "Linear combination:",
        linear_combination([[5, 1], [3, 4]], [-3, 7]),
    )
    sample_vectors = [[1, 0, -2], [0, 2, 1], [-1, 2, 3]]
    print("Column matrix:", matrix_from_columns(sample_vectors))
    print("Selected vectors:", select_vectors(sample_vectors, [0, 2]))

    print()
    print("Level 2")
    print("Target in span:", is_in_span([[1, -1], [2, 2]], [4, 8]))
    print("Standard pair independent:", is_linearly_independent([[1, 0], [0, 1]]))
    print("Scalar multiples independent:", is_linearly_independent([[1, 2], [2, 4]]))
    feature_vectors = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 2, 0],
        [0, 0, 0, 1],
        [2, -1, 1, 1],
    ]
    print("Basis indices:", basis_indices(feature_vectors))
    print("Basis vectors:", basis_from_spanning_set(feature_vectors))
    print("Dimension:", subspace_dimension(feature_vectors))


if __name__ == "__main__":
    main()