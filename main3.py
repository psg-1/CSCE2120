import copy

# ==========================================
# Level 1: Row Operations
# ==========================================

def scale_row(matrix, row_index, scale):
    """Scales a specific row by a scalar factor. Returns a new matrix."""
    if not isinstance(scale, (int, float)):
        raise TypeError("Scale must be a number")
    if scale == 0:
        raise ValueError("Scale cannot be zero")
    if row_index < 0 or row_index >= len(matrix):
        raise IndexError("Row index out of range")
        
    new_matrix = copy.deepcopy(matrix)
    new_matrix[row_index] = [entry * scale for entry in new_matrix[row_index]]
    return new_matrix

def replace_row(matrix, source_row, target_row, scale):
    """Adds a scaled version of the source row to the target row. Returns a new matrix."""
    if not isinstance(scale, (int, float)):
        raise TypeError("Scale must be a number")
    if source_row < 0 or source_row >= len(matrix) or target_row < 0 or target_row >= len(matrix):
        raise IndexError("Row index out of range")
        
    new_matrix = copy.deepcopy(matrix)
    new_matrix[target_row] = [
        t_val + scale * s_val 
        for t_val, s_val in zip(new_matrix[target_row], new_matrix[source_row])
    ]
    return new_matrix

# ==========================================
# Level 2: Echelon Structure & Variables
# ==========================================

def is_echelon_form(matrix):
    """Determines if a matrix is in row echelon form."""
    previous_leading_index = -1
    zero_row_already_seen = False
    
    for row in matrix:
        # Find leading index
        leading_index = -1
        for j, val in enumerate(row):
            if val != 0:
                leading_index = j
                break
        
        is_zero_row = (leading_index == -1)
        
        if is_zero_row:
            zero_row_already_seen = True
        elif zero_row_already_seen:
            return False
        elif leading_index <= previous_leading_index:
            return False
        else:
            previous_leading_index = leading_index
            
    return True

def pivot_columns(matrix):
    """Returns a list of the column indices containing pivot (leading) entries."""
    pivots = []
    for row in matrix:
        for j, val in enumerate(row):
            if val != 0:
                pivots.append(j)
                break
    return pivots

def basic_and_free_variables(echelon, num_variables):
    """Returns a tuple of two lists: (basic_variables, free_variables)."""
    pivots = pivot_columns(echelon)
    basic_vars = []
    free_vars = []
    
    for i in range(num_variables):
        var_name = f"x{i+1}"
        if i in pivots:
            basic_vars.append(var_name)
        else:
            free_vars.append(var_name)
            
    return (basic_vars, free_vars)

# ==========================================
# Level 3: Guided Gauss-Jordan Solver
# ==========================================

def find_pivot_row(matrix, start_row, pivot_col):
    """Finds the first row at or below start_row with a nonzero entry in pivot_col."""
    for i in range(start_row, len(matrix)):
        if matrix[i][pivot_col] != 0:
            return i
    return None

def normalize_pivot_row(matrix, pivot_row, pivot_col):
    """Divides the pivot row by the pivot value so the leading entry becomes 1."""
    pivot_value = matrix[pivot_row][pivot_col]
    return scale_row(matrix, pivot_row, 1.0 / pivot_value)

def eliminate_column(matrix, pivot_row, pivot_col):
    """Uses row replacement to make all other entries in the pivot column zero."""
    new_matrix = copy.deepcopy(matrix)
    for i in range(len(new_matrix)):
        if i != pivot_row and new_matrix[i][pivot_col] != 0:
            scale_factor = -new_matrix[i][pivot_col]
            new_matrix = replace_row(new_matrix, pivot_row, i, scale_factor)
    return new_matrix

def gauss_jordan_elimination(augmented):
    """Transforms an augmented matrix into reduced row echelon form."""
    matrix = copy.deepcopy(augmented)
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    
    current_row = 0
    # Process each coefficient column (excluding the augmented column)
    for pivot_col in range(num_cols - 1): 
        pivot_row = find_pivot_row(matrix, current_row, pivot_col)
        
        if pivot_row is not None:
            # Swap rows if the pivot is not in the current row
            if pivot_row != current_row:
                matrix[current_row], matrix[pivot_row] = matrix[pivot_row], matrix[current_row]
                pivot_row = current_row
            
            # Normalize the pivot row to make the leading entry 1
            matrix = normalize_pivot_row(matrix, pivot_row, pivot_col)
            
            # Eliminate all other entries in the current column
            matrix = eliminate_column(matrix, pivot_row, pivot_col)
            
            current_row += 1
            
    return matrix

def solve_system(augmented):
    """Uses Gauss-Jordan elimination to solve the system and returns the solution vector."""
    rref = gauss_jordan_elimination(augmented)
    
    # Read the solution from the last column
    solution = []
    for row in rref:
        # Assuming a unique solution system as per the lab requirements
        if any(val != 0 for val in row[:-1]): 
            solution.append(row[-1])
        
    return solution

# ==========================================
# Main Execution (Testing the code)
# ==========================================
if __name__ == "__main__":
    A = [
        [1, 3, -6],
        [-2, 5, 4],
    ]
    print("Level 1")
    print(f"Scaled row: {scale_row(A, 0, 3)}")
    print(f"Replaced row: {replace_row(A, 0, 1, 1)}\n")
    
    echelon = [
        [3, -2, 6, 0, 1],
        [0, 0, 2, 7, 5],
        [0, 0, 0, -3, 5],
        [0, 0, 0, 0, 0],
    ]
    
    rref_example = [
        [1, 3, 0, -3, 0, 4],
        [0, 0, 1, -2, 0, 2],
        [0, 0, 0, 0, 1, -3],
    ]
    
    print("Level 2")
    print(f"Is echelon: {is_echelon_form(echelon)}")
    print(f"Pivot columns: {pivot_columns(echelon)}")
    print(f"Basic/free: {basic_and_free_variables(rref_example, 5)}\n")
    
    system = [
        [1, -2, 2, 7],
        [-2, 5, -5, -20],
        [-1, 2, -1, -3],
    ]
    
    print("Level 3")
    print(f"Solution: {solve_system(system)}")