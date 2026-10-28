import math


def same_dimension(u, v):
    """Return True when u and v have the same number of components."""
    return len(u) == len(v)


def add_vectors(u, v):
    """Return the component-wise sum of u and v."""
    if not same_dimension(u, v):
        raise ValueError("vectors must have the same dimension")

    result = []
    for i in range(len(u)):
        result.append(u[i] + v[i])
    return result


def position_vector(initial, terminal):
    """Return the vector from initial to terminal."""
    if not same_dimension(initial, terminal):
        raise ValueError("points must have the same dimension")

    result = []
    for i in range(len(initial)):
        result.append(terminal[i] - initial[i])
    return result


def direction_r2(initial, terminal):
    """Return the direction angle in degrees in the interval [0, 360)."""
    if len(initial) != 2 or len(terminal) != 2:
        raise ValueError("points must be in R^2")

    v = position_vector(initial, terminal)

    if v[0] == 0 and v[1] == 0:
        raise ValueError("zero vector does not have a direction")

    angle = math.degrees(math.atan2(v[1], v[0]))

    if angle < 0:
        angle += 360

    return angle


def scalar_multiply(c, v):
    """Return the vector obtained by multiplying every component of v by c."""
    result = []
    for component in v:
        result.append(c * component)
    return result


def magnitude(v):
    """Return the magnitude of v."""
    total = 0
    for component in v:
        total += component * component
    return total ** 0.5


def feature_vector_results(student_a, student_b):
    """Return the combined, weighted, and magnitude results for Level 3."""
    combined = add_vectors(student_a, student_b)
    weighted_a = scalar_multiply(2, student_a)
    magnitude_a = magnitude(student_a)
    
    return combined, weighted_a, magnitude_a


def main():
    u = [1, 2]
    v = [5, -3]

    print("Level 1")
    print("Vector sum:", add_vectors(u, v))

    initial = [-8, 1]
    terminal = [-2, -5]

    print()
    print("Level 2")
    print("Position vector:", position_vector(initial, terminal))
    direction = direction_r2(initial, terminal)
    print("Direction:", direction, "degrees")

    student_a = [6, 8, 10]
    student_b = [5, 9, 7]
    combined, weighted_a, magnitude_a = feature_vector_results(
        student_a, student_b
    )

    print()
    print("Level 3")
    print("Combined feature vector:", combined)
    print("Weighted student_a:", weighted_a)
    if magnitude_a is None:
        print("Magnitude of student_a:", magnitude_a)
    else:
        print("Magnitude of student_a:", round(magnitude_a, 4))


if __name__ == "__main__":
    main()