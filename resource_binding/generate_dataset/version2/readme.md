## Dataset generation

### 1. introduction

- The entry of generation is **run_multi_thread.py**. This script enables us to run multiple process simultaneously, so that we can accelerate the generation. For example, run in terminal:

  > nohup python run_multi_thread.py >>thread1.log &

  This command will run one of the thread in the backstage and redirect the log into thread1.log.

- Instructions to customize **run_multi_thread.py** for your own use is listed inside the script.

- In **run_one_case.py**, you can edit the variable *timeout* to control the generation failure time limit.   

### 2. Environment

In this work, we generate data with vitis2021.1, python3,  linux.