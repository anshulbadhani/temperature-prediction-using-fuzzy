import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def define_fuzzy_variables():
    """
    Creates fuzzy input/output variables with 5 generic categories each:
    t1 (low) to t5 (high)
    """

    # Temperature and feature input ranges (you can tune these based on your dataset)
    temperature_range = np.arange(10, 50, 1)
    feature_range = np.arange(0, 100, 1)
    output_range = np.arange(10, 50, 1)

    # Create fuzzy variables
    temperature_input = ctrl.Antecedent(temperature_range, 'temperature_input')
    feature_input = ctrl.Antecedent(feature_range, 'feature_input')
    temperature_output = ctrl.Consequent(output_range, 'temperature_output')

    # Define 5 membership categories for each variable: t1 to t5
    labels = ['t1', 't2', 't3', 't4', 't5']

    # Function to evenly split a range into overlapping triangular categories
    def create_membership(var, universe, labels):
        step = (universe[-1] - universe[0]) // (len(labels) - 1)
        for i, label in enumerate(labels):
            if i == 0:
                var[label] = fuzz.trimf(universe, [universe[0], universe[0], universe[0] + step])
            elif i == len(labels) - 1:
                var[label] = fuzz.trimf(universe, [universe[-1] - step, universe[-1], universe[-1]])
            else:
                left = universe[0] + (i - 1) * step
                center = universe[0] + i * step
                right = universe[0] + (i + 1) * step
                var[label] = fuzz.trimf(universe, [left, center, right])

    create_membership(temperature_input, temperature_range, labels)
    create_membership(feature_input, feature_range, labels)
    create_membership(temperature_output, output_range, labels)

    return temperature_input, feature_input, temperature_output
