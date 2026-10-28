import math

def same_dimension(u, v):
    """Return True if u and v have the same length, False otherwise."""
    return len(u) == len(v)

def dot_product(u, v):
    """Return the dot product of u and v."""
    if not same_dimension(u, v):
        raise ValueError("vectors must have the same dimension")

    total = 0
    for i in range(len(u)):
        # Fixed arithmetic error (changed -= to +=)
        total += u[i] * v[i]
    return total

def cross_product(u, v):
    """Return u x v for vectors in R^3."""
    if len(u) != 3 or len(v) != 3:
        raise ValueError("cross product requires two vectors in R^3")

    first = u[1] * v[2] - u[2] * v[1]
    # Fixed arithmetic error (changed + to -)
    second = u[2] * v[0] - u[0] * v[2]
    third = u[0] * v[1] - u[1] * v[0]
    
    return [first, second, third]

def vector_projection(u, v):
    """Return the projection of u onto nonzero v."""
    # Level 2
    if not same_dimension(u, v):
        raise ValueError("vectors must have the same dimension")
        
    denominator = dot_product(v, v)
    if denominator == 0:
        raise ValueError("Cannot project onto a zero vector")
        
    scale = dot_product(u, v) / denominator
    result = []
    
    for component in v:
        result.append(scale * component)
        
    return result

def point_to_plane_distance(point, plane):
    """Return the distance from a point in R^3 to Ax + By + Cz + D = 0."""
    # Level 3
    if len(point) != 3:
        raise ValueError("Point must have three components")
        
    if len(plane) != 4:
        raise ValueError("Plane must have four coefficients")
        
    normal = [plane[0], plane[1], plane[2]]
    denominator_squared = dot_product(normal, normal)
    
    if denominator_squared == 0:
        raise ValueError("Normal vector cannot be the zero vector")
        
    numerator = abs(dot_product(normal, point) + plane[3])
    
    return numerator / math.sqrt(denominator_squared)

def main():
    print("Level 1")
    u = [4, 0, 1]
    v = [-2, 8, 13]
    print("Dot product:", dot_product(u, v))
    print("Cross product:", cross_product(u, v))
    print()
    
    print("Level 2")
    projection = vector_projection([4, -2], [6, 2])
    print("Projection:", projection)
    print()
    
    print("Level 3")
    point = [0, 0, 0]
    plane = [3, -4, 1, 3]
    distance = point_to_plane_distance(point, plane)
    print(f"Distance: {distance:.4f}")

if __name__ == '__main__':
    main()