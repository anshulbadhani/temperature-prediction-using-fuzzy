import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def define_fuzzy_variables():
    """
    Creates fuzzy input/output variables with 5 generic categories each:
    t1 (low) to t5 (high)
    """

    # Temperature and feature input ranges
    temp_min = np.arange(-5, 55, 1)
    temp_max = np.arange(-5, 55, 1)
    output_range = np.arange(10, 50, 1)

    # Create fuzzy variables
    temp_min_input = ctrl.Antecedent(temp_min, 'temp_min_input')
    temp_max_input = ctrl.Antecedent(temp_max, 'temp_max_input')
    temp_output = ctrl.Consequent(output_range, 'temperature_output')

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

    create_membership(temp_min_input, temp_min, labels)
    create_membership(temp_max_input, temp_max, labels)
    create_membership(temp_output, output_range, labels)

    return temp_min_input, temp_max_input, temp_output

def classify_value(value, fuzzy_variable):
    """
    Assigns a fuzzy category label (e.g., t1–t5) to a given numeric value
    by finding which membership function it fits best.
    """
    max_membership = 0
    best_label = None
    for label in fuzzy_variable.terms:
        membership_func = fuzzy_variable[label].mf
        membership_value = fuzz.interp_membership(
            fuzzy_variable.universe, membership_func, value
        )
        if membership_value > max_membership:
            max_membership = membership_value
            best_label = label
    return best_label


def generate_rules_from_data(dataframe, temp_var, feat_var, output_var):
    """
    Automatically creates fuzzy rules by reading from labeled CSV data.
    """
    rules = []

    for index, row in dataframe.iterrows():
        temp_val = row['Temperature']
        feat_val = row['Feature']
        out_val = row['NextDayTemperature']

        temp_label = classify_value(temp_val, temp_var)
        feat_label = classify_value(feat_val, feat_var)
        out_label = classify_value(out_val, output_var)

        rule = ctrl.Rule(
            antecedent=(temp_var[temp_label] & feat_var[feat_label]),
            consequent=output_var[out_label]
        )

        rules.append(rule)

    return rules

def main():
    file_path = "data.csv"  # Your CSV file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    df = pd.read_csv(file_path)

    # Ensure necessary columns exist
    if not {"min_temp", "max_temp", "next_day_min_temp"}.issubset(df.columns):
        raise ValueError("CSV must have: min_temp, max_temp, next_day_min_temp")

    # Create fuzzy variables
    min_var, max_var, output_var = define_fuzzy_variables()

    # Generate fuzzy rules from existing data
    rules = generate_rules_from_data(df, min_var, max_var, output_var)

    # Create control system and simulator
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)

    # Predict and update CSV
    predictions = []
    for _, row in df.iterrows():
        sim.input['min_temp'] = row['min_temp']
        sim.input['max_temp'] = row['max_temp']
        sim.compute()
        predictions.append(sim.output['predicted_min_temp'])

    df['predicted_min_temp'] = np.round(predictions, 2)
    df.to_csv(file_path, index=False)

    print(f"Updated {file_path} with predictions.")

if __name__ == "__main__":
    main()
