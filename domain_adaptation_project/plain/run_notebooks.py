# import papermill as pm
# import traceback
# notebooks = ["fiction/notebook1.ipynb", "fiction/TRTE_union_macro.ipynb"]
# kernel_name = 'python3'  # Use the identified kernel name

# for notebook in notebooks:
#     try:
#         print(f"Running {notebook}...")
#         pm.execute_notebook(
#             input_path=notebook,
#             output_path=notebook,
#             kernel_name=kernel_name
#         )
#         print(f"Successfully ran {notebook}")
#     except Exception as e:
#         print(f"Error running {notebook}:")
#         traceback.print_exc()


import papermill as pm
import traceback
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the location of your notebooks relative to the script's location
notebooks_dir = os.path.join(script_dir, 'task')
os.chdir(notebooks_dir)
print(f"Current working directory: {os.getcwd()}")

# Add the directory containing the modules to the Python path
module_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'modules'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(f"Module path: {module_path}")
print(f"Python path: {sys.path}")

# Ensure the environment variables are set correctly
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

# Define the notebooks to run and the kernel name
notebooks = [ 'S_Lora.ipynb', 'S_Prefix.ipynb', 'TR_Lora.ipynb', 'TR_Prefix.ipynb']
kernel_name = 'python3'

for notebook in notebooks:
    try:
        print(f"Running {notebook}...")
        pm.execute_notebook(
            input_path=notebook,
            output_path=notebook,
            kernel_name=kernel_name
        )
        print(f"Successfully ran {notebook}")
    except Exception as e:
        print(f"Error running {notebook}:")
        traceback.print_exc()
