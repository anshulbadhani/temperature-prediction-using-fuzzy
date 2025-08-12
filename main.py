import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Data Loading and Preparation
def load_and_prepare_data(filepath="data.csv"):
    """
    Loads data, calculates the average temperature, and prepares the target variable.

    Args:
        filepath (str): The path to the CSV data file.

    Returns:
        pandas.DataFrame: Prepared DataFrame with necessary columns.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

    # Ensure required columns exist
    required_cols = ["min_temp", "max_temp", "humidity", "pressure"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain the columns: {required_cols}")
        return None

    # Handle potential missing values (using forward fill as a simple strategy)
    df.ffill(inplace=True)

    # Calculate the average temperature for the current day
    df["avg_temp"] = (df["min_temp"] + df["max_temp"]) / 2

    # Prepare the target variable: the actual average temperature of the *next* day
    df["actual_next_avg_temp"] = df["avg_temp"].shift(-1)

    # Drop the last row as it has no "next day" to predict
    df.dropna(inplace=True)

    print("Data loaded and prepared successfully.")
    print(f"Data shape after preparation: {df.shape}")
    return df


# 2. Fuzzy System Definition
def define_fuzzy_system(df):
    """
    Defines the fuzzy antecedents (inputs) and consequent (output) based on data ranges.

    Args:
        df (pandas.DataFrame): The dataframe containing the training data.

    Returns:
        tuple: A tuple containing the defined fuzzy variables.
    """
    # Create dynamic ranges with a 5% buffer for robustness
    min_temp_range = (df["min_temp"].min() * 0.95, df["min_temp"].max() * 1.05)
    max_temp_range = (df["max_temp"].min() * 0.95, df["max_temp"].max() * 1.05)
    humidity_range = (df["humidity"].min() * 0.95, df["humidity"].max() * 1.05)
    pressure_range = (df["pressure"].min() * 0.99, df["pressure"].max() * 1.01)
    output_range = (
        df["actual_next_avg_temp"].min() * 0.95,
        df["actual_next_avg_temp"].max() * 1.05,
    )

    # Define Antecedents (Inputs)
    min_temp = ctrl.Antecedent(np.arange(*min_temp_range, 0.1), "min_temp")
    max_temp = ctrl.Antecedent(np.arange(*max_temp_range, 0.1), "max_temp")
    humidity = ctrl.Antecedent(np.arange(*humidity_range, 1), "humidity")
    pressure = ctrl.Antecedent(np.arange(*pressure_range, 0.1), "pressure")

    # Define Consequent (Output)
    next_avg_temp = ctrl.Consequent(np.arange(*output_range, 0.1), "next_avg_temp")

    # Automatically generate membership functions
    # Using 5 levels: 't1' (very less), 't2' (less), 't3' (average), 't4' (good), 't5' (very good)
    # The names are generic; they represent levels from low to high.
    var_names = ["t1", "t2", "t3", "t4", "t5"]
    min_temp.automf(names=var_names)
    max_temp.automf(names=var_names)
    humidity.automf(names=var_names)
    pressure.automf(names=var_names)
    next_avg_temp.automf(names=var_names)

    print("Fuzzy variables and membership functions defined.")
    return min_temp, max_temp, humidity, pressure, next_avg_temp


def get_fuzzy_labels(df, fuzzy_vars):
    """
    Determines the fuzzy label for each data point for every variable.

    Args:
        df (pandas.DataFrame): The input data.
        fuzzy_vars (dict): A dictionary of fuzzy variables.

    Returns:
        pandas.DataFrame: DataFrame with added columns for fuzzy labels.
    """
    fuzzified_df = df.copy()
    for name, var in fuzzy_vars.items():
        labels = []
        for val in df[name]:
            # Find the fuzzy set with the highest membership value for the given crisp input
            memberships = {
                label: fuzz.interp_membership(var.universe, var[label].mf, val)
                for label in var.terms
            }
            best_label = max(memberships, key=memberships.get)
            labels.append(best_label)
        fuzzified_df[f"{name}_fuzzy"] = labels
    return fuzzified_df


def generate_rules_from_data(df, fuzzy_vars):
    """
    Generates fuzzy rules by finding the most frequent patterns in the data.

    This is the core of the data-driven approach.

    Args:
        df (pandas.DataFrame): The prepared dataframe.
        fuzzy_vars (dict): A dictionary of the fuzzy variables.

    Returns:
        list: A list of generated fuzzy `ctrl.Rule` objects.
    """
    print("Generating rules from data...")

    # Fuzzify the entire dataset to get labels for each row
    fuzzy_inputs = {
        "min_temp": fuzzy_vars["min_temp"],
        "max_temp": fuzzy_vars["max_temp"],
        "humidity": fuzzy_vars["humidity"],
        "pressure": fuzzy_vars["pressure"],
    }
    fuzzy_output = {"actual_next_avg_temp": fuzzy_vars["next_avg_temp"]}

    df_fuzzified = get_fuzzy_labels(df, fuzzy_inputs)
    df_fuzzified = get_fuzzy_labels(df_fuzzified, fuzzy_output)

    # Group by the fuzzy input states and find the most common output state
    rule_antecedents = [
        "min_temp_fuzzy",
        "max_temp_fuzzy",
        "humidity_fuzzy",
        "pressure_fuzzy",
    ]

    # Use mode to find the most frequent consequent for each antecedent combination
    rule_map = (
        df_fuzzified.groupby(rule_antecedents)["actual_next_avg_temp_fuzzy"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    # Create the rules
    rules = []
    for _, row in rule_map.iterrows():
        antecedent = (
            fuzzy_vars["min_temp"][row["min_temp_fuzzy"]]
            & fuzzy_vars["max_temp"][row["max_temp_fuzzy"]]
            & fuzzy_vars["humidity"][row["humidity_fuzzy"]]
            & fuzzy_vars["pressure"][row["pressure_fuzzy"]]
        )
        consequent = fuzzy_vars["next_avg_temp"][row["actual_next_avg_temp_fuzzy"]]
        rules.append(ctrl.Rule(antecedent, consequent))

    print(f"Successfully generated {len(rules)} rules from data.")
    return rules


# 3. Fuzzy Simulation and Prediction


def run_fuzzy_simulation(df, rules, fuzzy_vars):
    """
    Runs the fuzzy control system simulation to predict next day's temperature.

    Args:
        df (pandas.DataFrame): The input data.
        rules (list): The list of fuzzy rules.
        fuzzy_vars (dict): A dictionary of the fuzzy variables.

    Returns:
        pandas.DataFrame: DataFrame with a new column for predictions.
    """
    print("Running fuzzy simulation...")
    # Create the control system and simulation
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)

    predictions = []
    for i, row in df.iterrows():
        try:
            # Set inputs for the simulation
            simulation.input["min_temp"] = row["min_temp"]
            simulation.input["max_temp"] = row["max_temp"]
            simulation.input["humidity"] = row["humidity"]
            simulation.input["pressure"] = row["pressure"]

            # Compute the result
            simulation.compute()
            predictions.append(simulation.output["next_avg_temp"])
        except Exception as e:
            # If a specific combination of inputs has no matching rule, it can fail.
            # We append NaN and handle it later.
            # print(f"Warning: Could not compute for row {i}. Error: {e}")
            predictions.append(np.nan)

    df["predicted_temp"] = predictions

    # A simple forward-fill for cases where no rule was activated
    df["predicted_temp"].ffill(inplace=True)
    df["predicted_temp"].bfill(inplace=True)  # Back-fill for any leading NaNs

    print("Simulation complete.")
    return df


# 4. Evaluation and Visualization


def evaluate_and_plot(df):
    """
    Calculates error metrics and plots the results for analysis.
    """
    if "predicted_temp" not in df.columns or df["predicted_temp"].isnull().all():
        print("Evaluation skipped: No valid predictions were made.")
        return

    # --- Metrics ---
    mae = np.mean(np.abs(df["actual_next_avg_temp"] - df["predicted_temp"]))
    rmse = np.sqrt(np.mean((df["actual_next_avg_temp"] - df["predicted_temp"]) ** 2))
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}°C")
    print("------------------------\n")

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Actual vs. Predicted Time Series
    plt.figure(figsize=(15, 6))
    plt.plot(
        df.index,
        df["actual_next_avg_temp"],
        label="Actual Next Day Temp",
        color="blue",
        alpha=0.8,
    )
    plt.plot(
        df.index,
        df["predicted_temp"],
        label="Predicted Temp",
        color="red",
        linestyle="--",
    )
    plt.title("Actual vs. Predicted Temperature", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Average Temperature (°C)")
    plt.legend()
    plt.show()

    # 2. Error Distribution
    errors = df["actual_next_avg_temp"] - df["predicted_temp"]
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, kde=True, bins=20)
    plt.title("Prediction Error Distribution", fontsize=16)
    plt.xlabel("Error (°C)")
    plt.ylabel("Frequency")
    plt.show()

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_df = df[
        [
            "min_temp",
            "max_temp",
            "humidity",
            "pressure",
            "actual_next_avg_temp",
            "predicted_temp",
        ]
    ]
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()


# Main Execution Block

if __name__ == "__main__":
    # 1. Load and process data
    data = load_and_prepare_data("data.csv")

    if data is not None:
        # Set date as index for better plotting if 'date' column exists
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data.set_index("date", inplace=True)

        # 2. Define the fuzzy system architecture
        min_temp_var, max_temp_var, humidity_var, pressure_var, next_avg_temp_var = (
            define_fuzzy_system(data)
        )

        fuzzy_variables = {
            "min_temp": min_temp_var,
            "max_temp": max_temp_var,
            "humidity": humidity_var,
            "pressure": pressure_var,
            "next_avg_temp": next_avg_temp_var,
        }

        # 3. Generate rules from the data
        fuzzy_rules = generate_rules_from_data(data, fuzzy_variables)

        # 4. Run the simulation to get predictions
        results_df = run_fuzzy_simulation(data, fuzzy_rules, fuzzy_variables)

        # 5. Evaluate the results and plot graphs
        evaluate_and_plot(results_df)

        # 6. Save results to CSV
        output_filepath = "temperature_predictions_output.csv"
        results_df.to_csv(output_filepath)
        print(f"Results saved to {output_filepath}")
