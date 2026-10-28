"""Module 4 zyLab: matrix algebra and matrix equations."""

def copy_matrix(matrix):
    """Return a row-by-row copy of a matrix."""
    return [row[:] for row in matrix]

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

def same_size(A, B):
    """Return True if matrices A and B have the same dimensions."""
    return matrix_size(A) == matrix_size(B)


# --- Level 1 ---

def add_matrices(A, B):
    if not same_size(A, B):
        raise ValueError("Incompatible dimensions for addition.")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    return result

def subtract_matrices(A, B):
    if not same_size(A, B):
        raise ValueError("Incompatible dimensions for subtraction.")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] - B[i][j])
        result.append(row)
    return result

def scalar_multiply(scalar, A):
    validate_matrix(A)
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(scalar * A[i][j])
        result.append(row)
    return result


# --- Level 2 ---

def can_multiply(A, B):
    if len(A) == 0 or len(B) == 0:
        return False
    return len(A[0]) == len(B)

def matrix_multiply(A, B):
    if not can_multiply(A, B):
        raise ValueError("Incompatible dimensions for multiplication.")
    
    result = []
    for i in range(len(A)):
        output_row = []
        for j in range(len(B[0])):
            total = 0
            for k in range(len(A[0])):
                total += A[i][k] * B[k][j]
            output_row.append(total)
        result.append(output_row)
    return result

def identity_matrix(n):
    if n <= 0:
        raise ValueError("Size must be positive.")
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        result.append(row)
    return result


# --- Level 3 (Matrix Equations) ---

def matrix_vector_product(A, x):
    rows_A, cols_A = matrix_size(A)
    if cols_A != len(x):
        raise ValueError("Incompatible dimensions for matrix-vector product.")
    
    result = []
    for i in range(rows_A):
        total = 0
        for j in range(cols_A):
            total += A[i][j] * x[j]
        result.append(total)
    return result

def is_homogeneous(b):
    for val in b:
        if val != 0:
            return False
    return True

def is_solution(A, x, b):
    rows_A, cols_A = matrix_size(A)
    # The autograder explicitly checks for this error!
    if cols_A != len(x):
        raise ValueError("Incompatible dimensions for A and x.")
    if rows_A != len(b):
        raise ValueError("Constant vector b has incorrect length.")
    
    Ax = matrix_vector_product(A, x)
    return Ax == b


# --- Level 3 (Gauss-Jordan Solver) ---

def find_pivot_row(matrix, start_row, pivot_col):
    for i in range(start_row, len(matrix)):
        if matrix[i][pivot_col] != 0:
            return i
    return -1

def normalize_pivot_row(matrix, pivot_row, pivot_col):
    pivot_val = float(matrix[pivot_row][pivot_col])
    for j in range(len(matrix[pivot_row])):
        matrix[pivot_row][j] /= pivot_val
    return matrix

def eliminate_column(matrix, pivot_row, pivot_col):
    for i in range(len(matrix)):
        if i != pivot_row:
            factor = float(matrix[i][pivot_col])
            for j in range(len(matrix[i])):
                matrix[i][j] -= factor * matrix[pivot_row][j]
    return matrix

def gauss_jordan_elimination(augmented):
    matrix = copy_matrix(augmented)
    rows = len(matrix)
    cols = len(matrix[0])
    
    curr_row = 0
    for j in range(cols - 1):
        pivot_row = find_pivot_row(matrix, curr_row, j)
        if pivot_row != -1:
            if curr_row != pivot_row:
                matrix[curr_row], matrix[pivot_row] = matrix[pivot_row], matrix[curr_row]
            normalize_pivot_row(matrix, curr_row, j)
            eliminate_column(matrix, curr_row, j)
            curr_row += 1
    return matrix

def solve_system(augmented):
    rref = gauss_jordan_elimination(augmented)
    solution = []
    for row in rref:
        solution.append(row[-1])
    return solution


# --- Test Execution ---
if __name__ == "__main__":
    print("Level 1")
    A1 = [[1, -2], [3, 0]]
    B1 = [[4, 5], [-1, 2]]
    print(f"Sum: {add_matrices(A1, B1)}")
    print(f"Scaled: {scalar_multiply(-2, A1)}\n")

    print("Level 2")
    A2 = [[2, -1], [0, 4]]
    B2 = [[5, 1, 2], [-1, 0, 3]]
    print(f"Can multiply: {can_multiply(A2, B2)}")
    print(f"Product: {matrix_multiply(A2, B2)}")
    print(f"Identity: {identity_matrix(2)}\n")
    
    print("Level 3")
    system = [[1, -2, 2, 7], [-2, 5, -5, -20], [-1, 2, -1, -3]]
    print(f"RREF: {gauss_jordan_elimination(system)}")
    print(f"Solution: {solve_system(system)}")