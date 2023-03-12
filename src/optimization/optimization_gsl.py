import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def column_sum_constraint(size: int) -> LinearConstraint:
    """Creates a linear constraint for all columns to sum to 1.

    Args:
        size (int): _description_

    Returns:
        LinearConstraint: _description_
    """
    constraints = []

    for i in range(size):
        matrix = np.zeros((size, size))
        matrix[:,i] = 1
        constraints.append(matrix.reshape(-1))

    return LinearConstraint(np.stack(constraints), lb=np.zeros(size) + 1, ub=np.zeros(size) + 1)

def positive_constraint(size: int) -> LinearConstraint:
    """Creates a linear constraint on every values having to be positive.
    Args:
        size (int): size of the matrix.
    Returns:
        LinearConstraint: Linear constraint.
    """
    constraints = []
    for i in range(size):
        for j in range(size):
            matrix = np.zeros((size, size))
            matrix[i,j] = 1
            constraints.append(matrix.reshape(-1))

    return LinearConstraint(np.stack(constraints), lb=np.zeros(size*size))

def diagonal_constraint(size: int) -> LinearConstraint:
    """
    Creates a linear constraint on the diagonal of the matrix to be null.
    Args:
        size (int): size of the matrix.
    Returns:
        LinearConstraint: Linear constraint.
    """
    constraints = []
    # Constraint on diagonal is 0
    for i in range(size):
        matrix = np.zeros((size, size))
        matrix[i,i] = 1
        constraints.append(matrix.reshape(-1))

    return LinearConstraint(np.stack(constraints), lb=0, ub=0)

def fitness_function(size: int, signal: np.array):
    def fitness(adj_array):
        adj_matrix = adj_array.reshape(size, size)
        external_sum = 0
        for i, x_i in enumerate(signal):
            internal_sum = 0
            for j, x_j in enumerate(signal):
                internal_sum += adj_matrix[i,j]*x_j
            external_sum += np.linalg.norm(x_i - internal_sum)
        return external_sum
    return fitness

def smoothness_function(size: int, signal: np.array):
    def smoothness(adj_array):
        adj_matrix = adj_array.reshape(size, size)
        acc = 0
        for i, x_i in enumerate(signal):
            for j, x_j in enumerate(signal):
                acc += 1/2 * adj_matrix[i,j]*np.linalg.norm(x_i - x_j)
        return acc
    return smoothness