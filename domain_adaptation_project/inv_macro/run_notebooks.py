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

# Define the list of folders and their respective notebooks
folders_and_notebooks = {
    #  'fiction': ['SF_inv_macro.ipynb', 'GF_inv_macro.ipynb', 'TEF_inv_macro.ipynb', 'TRF_inv_macro.ipynb'],
    # 'slate': ['FS_inv_macro.ipynb', 'GS_inv_macro.ipynb', 'TES_inv_macro.ipynb', 'TRS_inv_macro.ipynb'],
    # 'government':  ['TEG_inv_macro_2.ipynb', 'TRG_inv_macro.ipynb'],
    # 'telephone':  ['FTE_inv_macro.ipynb', 'GTE_inv_macro.ipynb', 'STE_inv_macro.ipynb', 'TRTE_inv_macro.ipynb'],
    'travel': ['FTR_inv_macro.ipynb', 'STR_inv_macro.ipynb', 'GTR_inv_macro.ipynb', 'TETR_inv_macro.ipynb']
}

# Loop through each folder
for folder, notebooks in folders_and_notebooks.items():
    # Set the working directory to the location of your notebooks relative to the script's location
    notebooks_dir = os.path.join(script_dir, folder)
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

    # Define the kernel name
    kernel_name = 'python3'

    # Loop through and run each notebook
    for notebook in notebooks:
        try:
            print(f"Running {notebook} in folder {folder}...")
            pm.execute_notebook(
                input_path=notebook,
                output_path=notebook,
                kernel_name=kernel_name
            )
            print(f"Successfully ran {notebook}")
        except Exception as e:
            print(f"Error running {notebook} in folder {folder}:")
            traceback.print_exc()
