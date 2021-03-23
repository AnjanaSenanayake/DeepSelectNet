import os
from LocalConfigs import *

if os.path.exists(NPY_LIST_FILE):
    os.remove(NPY_LIST_FILE)

npy_list_file = open(NPY_LIST_FILE, "a")

for root, dirs, files in os.walk(TRAIN_DIR):
    for file in files:
        if file.endswith(".npy"):
            npy_list_file.write(file.split('.npy')[0] + '\n')
            print(file.split('.npy')[0])
    npy_list_file.close()
