import numpy as np
import pandas as pd
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
temp range
"""


def define_fuzzy_variables():
    """
    Creates fuzzy input/output variables for AVERAGE temperature prediction:
    - Temperature: t1 (very_cold) to t5 (very_hot)
    - Humidity: h1 (very_dry) to h5 (very_humid)
    - Pressure: p1 (very_low) to p5 (very_high)
    - Output: Next day's AVERAGE temperature
    """
    # Temperature ranges - wider range for average temperatures
    temp_range = np.arange(-5, 50, 1)
    avg_temp_output_range = np.arange(0, 45, 1)  # Average temp output range

    # Weather parameter ranges (adjusted for your data)
    humidity_range = np.arange(0, 101, 1)  # 0-100% humidity
    pressure_range = np.arange(96.5, 99.5, 0.1)  # 96.5-99.5 kPa to match your data range

    # Create fuzzy variables
    temp_min_input = ctrl.Antecedent(temp_range, 'temp_min_input')
    temp_max_input = ctrl.Antecedent(temp_range, 'temp_max_input')
    humidity_input = ctrl.Antecedent(humidity_range, 'humidity_input')
    pressure_input = ctrl.Antecedent(pressure_range, 'pressure_input')
    avg_temp_output = ctrl.Consequent(avg_temp_output_range, 'avg_temp_output')

    # Define membership function labels
    temp_labels = ['t1', 't2', 't3', 't4', 't5']  # very_cold to very_hot
    humidity_labels = ['h1', 'h2', 'h3', 'h4', 'h5']  # very_dry to very_humid
    pressure_labels = ['p1', 'p2', 'p3', 'p4', 'p5']  # very_low to very_high

    def create_membership_functions(var, universe, labels):
        """Create overlapping triangular membership functions"""
        n_labels = len(labels)
        min_val = universe[0]
        max_val = universe[-1]
        range_size = max_val - min_val

        for i, label in enumerate(labels):
            if i == 0:
                # Left shoulder function (e.g., very_cold, very_dry, very_low)
                peak = min_val + range_size * 0.3
                var[label] = fuzz.trimf(universe, [min_val, min_val, peak])
            elif i == n_labels - 1:
                # Right shoulder function (e.g., very_hot, very_humid, very_high)
                peak = min_val + range_size * 0.7
                var[label] = fuzz.trimf(universe, [peak, max_val, max_val])
            else:
                # Middle triangular functions
                left_point = min_val + range_size * (i - 0.5) / (n_labels - 1)
                center_point = min_val + range_size * i / (n_labels - 1)
                right_point = min_val + range_size * (i + 0.5) / (n_labels - 1)
                var[label] = fuzz.trimf(universe, [left_point, center_point, right_point])

    # Create membership functions for all variables
    create_membership_functions(temp_min_input, temp_range, temp_labels)
    create_membership_functions(temp_max_input, temp_range, temp_labels)
    create_membership_functions(humidity_input, humidity_range, humidity_labels)
    create_membership_functions(pressure_input, pressure_range, pressure_labels)
    create_membership_functions(avg_temp_output, avg_temp_output_range, temp_labels)

    return temp_min_input, temp_max_input, humidity_input, pressure_input, avg_temp_output


def create_average_temp_rules(temp_min, temp_max, humidity, pressure, output):
    """
    Create fuzzy rules for AVERAGE temperature prediction

    For your pressure range (97-99 kPa over Jan-July):
    - p1 (very_low): ~97.0-97.4 kPa - Relatively low pressure, potential weather changes
    - p2 (low): ~97.4-97.8 kPa - Below average pressure, some instability
    - p3 (normal): ~97.8-98.2 kPa - Average pressure for your region/season
    - p4 (high): ~98.2-98.6 kPa - Above average pressure, more stable conditions
    - p5 (very_high): ~98.6-99.0+ kPa - High pressure, stable/clear weather

    Average temperature logic:
    - Next day's average ≈ (today's min + today's max) / 2, with adjustments
    - Pressure affects stability of this relationship
    - Humidity affects the temperature moderation
    """

    rules = []

    # Primary rule: Average temperature based on current min/max range
    # Rule logic: avg_temp ≈ (min_temp + max_temp) / 2 with weather adjustments

    # When min and max are in same category, average follows closely
    rules.extend([
        ctrl.Rule(temp_min['t1'] & temp_max['t1'], output['t1']),  # Cold day -> cold average
        ctrl.Rule(temp_min['t2'] & temp_max['t2'], output['t2']),  # Cool day -> cool average
        ctrl.Rule(temp_min['t3'] & temp_max['t3'], output['t3']),  # Moderate day -> moderate average
        ctrl.Rule(temp_min['t4'] & temp_max['t4'], output['t4']),  # Warm day -> warm average
        ctrl.Rule(temp_min['t5'] & temp_max['t5'], output['t5']),  # Hot day -> hot average
    ])

    # When min and max differ, average between them (most common case)
    rules.extend([
        ctrl.Rule(temp_min['t1'] & temp_max['t2'], output['t1']),  # Very cold min, cool max
        ctrl.Rule(temp_min['t1'] & temp_max['t3'], output['t2']),  # Cold min, moderate max
        ctrl.Rule(temp_min['t1'] & temp_max['t4'], output['t2']),  # Cold min, warm max
        ctrl.Rule(temp_min['t1'] & temp_max['t5'], output['t3']),  # Cold min, hot max

        ctrl.Rule(temp_min['t2'] & temp_max['t3'], output['t2']),  # Cool min, moderate max
        ctrl.Rule(temp_min['t2'] & temp_max['t4'], output['t3']),  # Cool min, warm max
        ctrl.Rule(temp_min['t2'] & temp_max['t5'], output['t3']),  # Cool min, hot max

        ctrl.Rule(temp_min['t3'] & temp_max['t4'], output['t3']),  # Moderate min, warm max
        ctrl.Rule(temp_min['t3'] & temp_max['t5'], output['t4']),  # Moderate min, hot max

        ctrl.Rule(temp_min['t4'] & temp_max['t5'], output['t4']),  # Warm min, hot max
    ])

    # Humidity influence on average temperature
    # High humidity tends to moderate average temperatures (thermal inertia)
    rules.extend([
        ctrl.Rule(temp_min['t1'] & humidity['h5'], output['t2']),  # High humidity warms cold days
        ctrl.Rule(temp_min['t5'] & humidity['h5'], output['t4']),  # High humidity cools hot days
        ctrl.Rule(temp_min['t3'] & humidity['h1'], output['t3']),  # Low humidity maintains moderate
        ctrl.Rule(temp_min['t2'] & humidity['h1'], output['t2']),  # Low humidity maintains cool
    ])

    # Pressure influence on average temperature persistence
    # High pressure = more stable averages, low pressure = potential changes
    rules.extend([
        ctrl.Rule(temp_min['t3'] & pressure['p5'], output['t3']),  # High pressure = stable moderate
        ctrl.Rule(temp_min['t4'] & pressure['p5'], output['t4']),  # High pressure = stable warm
        ctrl.Rule(temp_min['t2'] & pressure['p5'], output['t2']),  # High pressure = stable cool

        ctrl.Rule(temp_min['t3'] & pressure['p1'], output['t2']),  # Low pressure = cooling trend
        ctrl.Rule(temp_min['t4'] & pressure['p1'], output['t3']),  # Low pressure = cooling from warm
        ctrl.Rule(temp_min['t2'] & pressure['p1'], output['t2']),  # Low pressure = maintain cool
    ])

    # Combined weather effects on average temperature
    rules.extend([
        # High humidity + high pressure = very stable average temperatures
        ctrl.Rule(temp_min['t3'] & humidity['h4'] & pressure['p4'], output['t3']),
        ctrl.Rule(temp_min['t2'] & humidity['h4'] & pressure['p4'], output['t2']),
        ctrl.Rule(temp_min['t4'] & humidity['h4'] & pressure['p4'], output['t4']),

        # Low humidity + low pressure = more variable, tend toward cooler averages
        ctrl.Rule(temp_min['t3'] & humidity['h2'] & pressure['p2'], output['t3']),
        ctrl.Rule(temp_min['t4'] & humidity['h2'] & pressure['p2'], output['t3']),

        # Moderate conditions = predictable averages
        ctrl.Rule(temp_min['t2'] & temp_max['t4'] & humidity['h3'] & pressure['p3'], output['t3']),
        ctrl.Rule(temp_min['t3'] & temp_max['t5'] & humidity['h3'] & pressure['p3'], output['t4']),
    ])

    # Seasonal adjustment rules (assuming Jan-July progression)
    # Add more warming bias for spring/summer progression
    rules.extend([
        ctrl.Rule(temp_min['t2'] & temp_max['t3'] & humidity['h3'], output['t3']),  # Spring warming
        ctrl.Rule(temp_min['t3'] & temp_max['t4'] & pressure['p4'], output['t4']),  # Summer warming
    ])

    return rules


def calculate_current_average(df):
    """Calculate current day's average temperature for comparison"""
    df['current_avg_temp'] = (df['min_temp'] + df['max_temp']) / 2
    return df


def validate_enhanced_data(df):
    """Validate data with humidity and pressure parameters for average temp prediction"""

    # print("Validating dataset for average temperature prediction...")

    # Check required columns
    required_cols = ['min_temp', 'max_temp', 'humidity', 'pressure']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Found all required columns: {required_cols}")

    # Calculate current average temperature for reference
    df = calculate_current_average(df)

    # Check and clean data ranges
    df['humidity'] = df['humidity'].clip(0, 100)  # 0-100%
    df['pressure'] = df['pressure'].clip(96.5, 99.5)  # Match your actual pressure range

    # Check for missing values
    missing_count = int(df[required_cols].isnull().sum().sum())
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values found")
        # Fill missing values with reasonable defaults
        df['humidity'].fillna(50, inplace=True)  # Default moderate humidity
        df['pressure'].fillna(98.0, inplace=True)  # Default middle of your pressure range
        df['min_temp'].fillna(method='forward', inplace=True)
        df['max_temp'].fillna(method='forward', inplace=True)

    # Check for logical inconsistencies
    inconsistent = df['min_temp'] > df['max_temp']
    if inconsistent.any():
        print(f"Warning: {inconsistent.sum()} rows have min_temp > max_temp")
        df.loc[inconsistent, ['min_temp', 'max_temp']] = df.loc[inconsistent, ['max_temp', 'min_temp']].values

    # print(f"Data validation complete. {len(df)} rows ready for processing.")
    print(f"Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
    print(f"Current average temp range: {df['current_avg_temp'].min():.1f}°C to {df['current_avg_temp'].max():.1f}°C")
    print(f"Humidity range: {df['humidity'].min():.1f}% to {df['humidity'].max():.1f}%")
    print(f"Pressure range: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} kPa")

    return df


def main():
    file_path = "data.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    df = pd.read_csv(file_path)
    # print(f"Loaded {len(df)} rows from {file_path}")
    # print(f"Available columns: {list(df.columns)}")

    # Validate data with weather parameters
    df = validate_enhanced_data(df)

    # Create enhanced fuzzy variables for average temperature
    temp_min, temp_max, humidity, pressure, avg_temp_output = define_fuzzy_variables()

    # Create rules for average temperature prediction
    rules = create_average_temp_rules(temp_min, temp_max, humidity, pressure, avg_temp_output)
    print(f"Created {len(rules)} fuzzy rules for average temperature prediction")

    try:
        # Create control system
        system = ctrl.ControlSystem(rules)

        # Make predictions
        predictions = []
        errors = 0

        print("Making average temperature predictions...")
        for index, row in df.iterrows():
            try:
                sim = ctrl.ControlSystemSimulation(system)
                sim.input['temp_min_input'] = row['min_temp']
                sim.input['temp_max_input'] = row['max_temp']
                sim.input['humidity_input'] = row['humidity']
                sim.input['pressure_input'] = row['pressure']
                sim.compute()
                predictions.append(sim.output['avg_temp_output'])
            except Exception as e:
                predictions.append(np.nan)
                errors += 1
                if errors <= 3:
                    print(f"Error on row {index}: {e}")

        df['predicted_next_avg_temp'] = np.round(predictions, 2)

        # Show results
        valid_predictions = df['predicted_next_avg_temp'].dropna()
        print(f"\nAverage Temperature Prediction Results:")
        print(
            f"Successful predictions: {len(valid_predictions)}/{len(df)} ({len(valid_predictions) / len(df) * 100:.1f}%)")

        if len(valid_predictions) > 0:
            print(
                f"Predicted avg temperature range: {valid_predictions.min():.1f}°C to {valid_predictions.max():.1f}°C")
            print(f"Average predicted temperature: {valid_predictions.mean():.1f}°C")

        if errors > 0:
            print(f"Prediction errors: {errors}")

        # Save results
        output_file = file_path.replace('.csv', '_avg_temp_predictions.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

        # Display sample results with all parameters
        display_cols = ['min_temp', 'max_temp', 'current_avg_temp', 'humidity', 'pressure', 'predicted_next_avg_temp']
        print(f"\nSample average temperature predictions:")
        print(df[display_cols].head(10))

        # Show prediction analysis
        if len(valid_predictions) > 5:
            print(f"\nPrediction Analysis:")

            # Compare with simple average baseline
            simple_avg = (df['min_temp'] + df['max_temp']) / 2
            prediction_vs_simple = df['predicted_next_avg_temp'] - simple_avg
            prediction_vs_simple_clean = prediction_vs_simple.dropna()

            if len(prediction_vs_simple_clean) > 0:
                print(f"Fuzzy prediction vs simple average:")
                print(f"  Average difference: {prediction_vs_simple_clean.mean():.2f}°C")
                print(
                    f"  Difference range: {prediction_vs_simple_clean.min():.1f}°C to {prediction_vs_simple_clean.max():.1f}°C")

            # Show weather parameter correlations
            high_pressure_mask = df['pressure'] > df['pressure'].quantile(0.7)
            low_pressure_mask = df['pressure'] < df['pressure'].quantile(0.3)

            if high_pressure_mask.sum() > 0 and low_pressure_mask.sum() > 0:
                high_pressure_avg = df.loc[high_pressure_mask, 'predicted_next_avg_temp'].mean()
                low_pressure_avg = df.loc[low_pressure_mask, 'predicted_next_avg_temp'].mean()
                print(f"\nPressure influence on predictions:")
                print(f"  High pressure days avg prediction: {high_pressure_avg:.1f}°C")
                print(f"  Low pressure days avg prediction: {low_pressure_avg:.1f}°C")
                print(f"  Pressure effect: {high_pressure_avg - low_pressure_avg:.1f}°C difference")

    except Exception as e:
        print(f"Error creating fuzzy control system: {e}")
        print("This might be due to insufficient rule coverage or data range issues.")


if __name__ == "__main__":
    main()