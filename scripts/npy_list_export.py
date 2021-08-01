import os
from LocalConfigs import *

if os.path.exists(VAL_NPY_LIST_FILE):
    os.remove(VAL_NPY_LIST_FILE)

npy_list_file = open(VAL_NPY_LIST_FILE, "a")

for root, dirs, files in os.walk(VALIDATION_DIR):
    for file in files:
        if file.endswith(".npy"):
            npy_list_file.write(file.split('.npy')[0] + '\n')
            print(file.split('.npy')[0])
    npy_list_file.close()
