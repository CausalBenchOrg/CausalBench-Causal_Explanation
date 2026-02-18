import os
import zipfile
import csv
import yaml
import pandas as pd
from rapidfuzz import process, fuzz
from causalbench.modules import Dataset
from causalbench.modules import Run
from causalbench.modules.context import Context

from common_constants import TEMP_DIR
os.environ['MPLCONFIGDIR'] = f'{TEMP_DIR}/mplconfig'

# Set working directory to parent dir
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# os.chdir(parent_dir)


def read_yaml(yaml_file):
    """Reads the YAML file and returns the parsed data."""
    try:
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {yaml_file} was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def replace_null_with_none(value, default=None):
    return default if value is None else value


def count_csv_rows(file_path):
    with open('dataset/' + file_path, 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader) - 1  # Subtract 1 for the header row
    return row_count


def extract_information(results, profiling):
    """Extracts the required data from results and profiling sections."""
    extracted_data = []
    hyperparameter_list = []

    for result in results:
        dataset = result.dataset
        model = result.model
        duration = 0

        # number_of_cols, number_of_rows = dataset_extract(dataset.id, dataset.version)

        hyperparameters = {}
        for param in model.hyperparameters:
            hyperparameters[param] = model.hyperparameters[param]

        # Loop through the metrics in each result
        gpu_key = None
        for metric in result.metrics:
            # Sum the duration from both model and metric
            duration += model.time.duration + metric.time.duration
            if metric.profiling.gpu:
                gpu_key = next(iter(metric.profiling.gpu))
            row = [
                dataset.id,  # DS.ID
                dataset.version,  # DS.Version
                dataset.name,  # DS.Name
                model.id,  # Model.ID
                model.version,  # Model.Version
                model.name,  # Model.Name
                model.profiling.memory,  # Model.Memory
                model.profiling.gpu[gpu_key].idle if gpu_key else None,  # Model.GPUMemoryIdle
                model.profiling.gpu[gpu_key].peak if gpu_key else None,  # Model.GPUMemoryPeak
                sum([disk_obj.read_bytes for disk_name, disk_obj in model.profiling.disk.items()]),  # Model.ReadBytes
                sum([disk_obj.write_bytes for disk_name, disk_obj in model.profiling.disk.items()]),  # Model.WriteBytes
                metric.id,  # Metric.ID
                metric.version,  # Metric.Version
                metric.name,  # Metric.Name
                metric.output.score,  # Metric.Score
                metric.profiling.memory,  # Metric.Memory
                metric.profiling.gpu[gpu_key].idle if gpu_key else None,  # Metric.GPUMemoryIdle
                metric.profiling.gpu[gpu_key].peak if gpu_key else None,  # Metric.GPUMemoryPeak
                sum([disk_obj.read_bytes for disk_name, disk_obj in metric.profiling.disk.items()]),  # Metric.ReadBytes
                sum([disk_obj.write_bytes for disk_name, disk_obj in metric.profiling.disk.items()]),  # Metric.WriteBytes
                profiling.cpu.name,  # CPU Name
                profiling.gpu[gpu_key].name if gpu_key else None,  # GPU Name
                profiling.gpu[gpu_key].memory_total if gpu_key else None,  # GPU.MemoryTotal
                profiling.memory_total,  # HW.MemoryTotal
                profiling.storage_total,  # HW.StorageTotal
                profiling.platform.name,  # SW.Platform
                duration,  # Time.Duration
                metric.profiling.python  # SW.PythonVersion
            ]
            extracted_data.append(row)
            hyperparameter_list.append(hyperparameters)

    return extracted_data, hyperparameter_list


def write_headers(headers):
    """
    Initializes an empty DataFrame with the given headers as column names.
    Returns:
        df (pd.DataFrame): Empty DataFrame with specified columns.
    """
    try:
        df = pd.DataFrame(columns=headers)
        print("Headers have been successfully stored in an in-memory DataFrame.")
        return df
    except Exception as e:
        print(f"Error creating DataFrame with headers: {e}")
        # Return an empty DataFrame in case of error to keep downstream code safe
        return pd.DataFrame()


def edit_csv_header(csv_file, new_headers):
    """Edits the header row of an existing CSV file."""
    try:
        # Read the entire content of the CSV file
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)  # Read all the rows into a list

        # Replace the first row with the new headers
        if len(data) > 0:
            data[0] = new_headers
        else:
            print("The CSV file is empty!")
            return

        # Write back the modified data (including the new headers) into the same file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print(f"Header has been successfully updated in {csv_file}")

    except IOError as e:
        print(f"Error editing header in CSV file: {e}")


def append_rows_to_df(extracted_data, hyperparameter_list, df):
    """
    Appends rows of data (with hyperparameters) into the in-memory DataFrame `df`.

    Parameters:
    - extracted_data: List[List], each inner list is a row of values matching df’s non-HP columns.
    - hyperparameter_list: List[Dict], each dict maps hyperparameter names to their values for the corresponding row.
    - df: pd.DataFrame, existing table (may be empty) whose columns include your data headers
          and any previously seen "HP.<name>" columns.

    Returns:
    - pd.DataFrame: the updated DataFrame with new rows and any new HP columns added.
    """
    # 1) Determine all needed HP columns
    needed_hp_cols = {f"HP.{name}" for hp in hyperparameter_list for name in hp}

    # 2) Add any missing HP columns to df (initialized to None)
    for col in needed_hp_cols:
        if col not in df.columns:
            df[col] = None

    # 3) Figure out which columns in df are the “base” (non-HP) columns
    base_cols = [c for c in df.columns if not c.startswith("HP.")]

    # 4) Build a list of new rows as dicts
    new_rows = []
    for row_vals, hp in zip(extracted_data, hyperparameter_list):
        row_dict = {}
        # map values to base columns
        for idx, col in enumerate(base_cols):
            row_dict[col] = row_vals[idx] if idx < len(row_vals) else None
        # inject hyperparameters
        for name, val in hp.items():
            row_dict[f"HP.{name}"] = val
        new_rows.append(row_dict)

    # 5) Concatenate the new rows into df
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)

    print(f"Appended {len(new_rows)} rows to in-memory DataFrame.")
    return df

def dataset_extract(id, version):
    """Extracts dataset information."""
    dataset_web = Dataset(module_id=id, version=version)
    fetched_web = dataset_web.fetch()

    with zipfile.ZipFile(fetched_web, 'r') as zip_ref:
        zip_ref.extractall("./dataset")

    dataset_yaml_file = './dataset/config.yaml'
    yaml_data = read_yaml(dataset_yaml_file)

    if yaml_data is None:
        return

    data_file = yaml_data['files']['file1']['path']
    number_of_rows = count_csv_rows(data_file)

    return len(yaml_data["files"]["file1"]["columns"]), number_of_rows


def process_yaml(yaml_file, df):
    """Process a single yaml file: parse YAML, and write data to CSV."""
    # Read and process the YAML file
    try:
        run: Run = Run(zip_file=yaml_file)
        # Extract data from the results and profiling sections
        results = run.results
        profiling = run.profiling
        extracted_data, hyperparameters = extract_information(results, profiling)

        # Append extracted data as rows to the CSV file
        df = append_rows_to_df(extracted_data, hyperparameters, df)
    except Exception as e:
        print(f"Error processing {yaml_file}: {e}")
    finally:
        return df

def process_multiple_yamls(yaml_directory, headers):
    # Write headers to the CSV file only once
    df = write_headers(headers)

    # Loop over each .zip file in the specified directory
    for filename in os.listdir(yaml_directory):
        if filename.endswith('.zip'):
            yaml_file_path = os.path.join(yaml_directory, filename)
            print(f"Processing {yaml_file_path}...")
            df = process_yaml(yaml_file_path, df)
    
    return df


def merge_benchmark_data(df, cpu_benchmark_df, gpu_benchmark_df):
    """
    Merges CPU and GPU benchmark data into your in-memory DataFrame.

    Parameters:
    - df: pd.DataFrame with at least 'CPU Name' and 'GPU Name' columns.
    - cpu_benchmark_df: pd.DataFrame of CPU benchmarks, with columns like 'CPU', 'SingleCore', 'MultiCore'.
    - gpu_benchmark_df: pd.DataFrame of GPU benchmarks, with columns like 'Device', 'Score'.

    Returns:
    - pd.DataFrame: the original df plus 'HW.CPUSingleCore', 'HW.CPUMultiCore', 'HW.GPUScore'.
    """
    # 1) Ensure the HW columns exist
    for col in ['SingleCore', 'MultiCore']:
        hw_col = f"HW.CPU{col}"
        if hw_col not in df.columns:
            df[hw_col] = None

    for col in ['Score']:
        hw_col = f"HW.GPU{col}"
        if hw_col not in df.columns:
            df[hw_col] = None

    # 2) For each row, fuzzy-match and fill in benchmark values
    for idx, row in df.iterrows():
        cpu_name = row.get('CPU Name')
        if pd.notna(cpu_name):
            cpu_match = fuzzy_match_device(cpu_name, cpu_benchmark_df, 'CPU')
            if cpu_match is not None:
                for col in ['SingleCore', 'MultiCore']:
                    df.at[idx, f"HW.CPU{col}"] = cpu_match.get(col)

        gpu_name = row.get('GPU Name')
        if pd.notna(gpu_name):
            gpu_match = fuzzy_match_device(gpu_name, gpu_benchmark_df, 'Device')
            if gpu_match is not None:
                df.at[idx, "HW.GPUScore"] = gpu_match.get('Score')

    # 3) Drop the old name columns if you no longer need them
    df = df.drop(columns=['CPU Name', 'GPU Name'], errors='ignore')

    return df

def fuzzy_match_device(device_name, benchmark_df, column_name, threshold=40):
    """
    Performs fuzzy matching to find the best match for a given device name in the benchmark DataFrame.

    Args:
    device_name (str): The device name to match.
    benchmark_df (DataFrame): The DataFrame containing benchmark data.
    column_name (str): The column in the benchmark DataFrame that contains device names.
    threshold (int): The minimum score to consider a valid match (default is 80).

    Returns:
    dict: The row from the benchmark_df that best matches the device name, or None if no match found.
    """
    # Use rapidfuzz's process.extractOne for fuzzy matching
    match = process.extractOne(device_name, benchmark_df[column_name], scorer=fuzz.ratio)

    # Check if the match meets the threshold
    if match and match[1] >= threshold:
        matched_device = benchmark_df[benchmark_df[column_name] == match[0]]
        return matched_device.iloc[0]  # Return the best match
    return None


def main(yaml_directory, headers):
    """Main function to process multiple zip files and write results to a CSV."""
    df = process_multiple_yamls(yaml_directory, headers)
    print(df)
    # Merge benchmark data with the final CSV
    cpu_benchmark_csv = os.path.join(os.path.dirname(__file__), 'HWBench', 'GeekbenchCPU.csv')
    gpu_benchmark_csv = os.path.join(os.path.dirname(__file__), 'HWBench', 'geekbenchopencl-gpu.csv')
    
    print(f"Loading CPU benchmark data from {cpu_benchmark_csv}")
    print(f"Loading GPU benchmark data from {gpu_benchmark_csv}")
    cpu_benchmark_df = pd.read_csv(cpu_benchmark_csv)
    gpu_benchmark_df = pd.read_csv(gpu_benchmark_csv)
    merged_df = merge_benchmark_data(df, cpu_benchmark_df, gpu_benchmark_df)
    return merged_df
    # output_path = "merged_output.csv"
    # merged_df.to_csv(output_path, index=False)


# Select and fetch the Context
#context1: Context = Context(module_id=2, version=1)

# Run selected Context

# Print Run execution results

# Publish the Run

# Headers for the DataFrame
headers = [
    "DS.ID", "DS.Version", "DS.Name",
    "Model.ID", "Model.Version", "Model.Name", "Model.Memory",
    "Model.GPUMemoryIdle", "Model.GPUMemoryPeak", "Model.ReadBytes",
    "Model.WriteBytes", "Metric.ID", "Metric.Version", "Metric.Name",
    "Metric.Score", "Metric.Memory", "Metric.GPUMemoryIdle",
    "Metric.GPUMemoryPeak", "Metric.ReadBytes",
    "Metric.WriteBytes", "CPU Name", "GPU Name", "GPU.MemoryTotal",
    "HW.MemoryTotal", "HW.StorageTotal", "SW.Platform", "Time.Duration",
    "SW.PythonVersion"
]

# Only run if this file is executed directly, not when imported
if __name__ == "__main__":
    # Input YAML folder path
    yaml_dir = 'zip_folder'
    
    # Run the main function
    main(yaml_dir, headers)
