import Gen_all_cases
import Run_one_case
import os
import subprocess
#####################################################################################
#  """Run multiple vivado by changing the range of cases you wanna run."""
#  ---------------------------------------------------------------------------------
#  1. For example, run case_1 to case_100 on the first thread, set_start_case = 1.
#     similarly, run second thread from case_101 to case_200, set_start_case = 101
#  2. To change the step of the case range, change variable "step" (default 100)
#  3. to run in the shell cmd, run with: 
#            "nohup python application1/run_mulit-thread.py >>[re_deirecting file] &"
#     For each thread, modifiy the [re_directing file] for the output file name.
#  4. The timeout limit is set to 15 minutes. 
#  5. Don't forget to add vivado_hls or vitis_hls into to your env.
#####################################################################################
set_start_case = 1
step = 100
set_end_case = set_start_case + step

if __name__ == '__main__':
    gen_path = './gen_dfg'
    # gen_path = './gen_40x200'
    case_num = 300
    print(os.getcwd())
    # generate cases
    if not os.path.exists(gen_path):
        Gen_all_cases.gen_all_cases(gen_path, case_num=case_num)

    # run cases
    for case in range(set_start_case, set_end_case+1):
        # path = "/home/congh/HLS_ML/data_collect/HLS/case_" + str(case) + "/"
        path = gen_path + "/case_" + str(case) + "/"
        # solution_1 is the default HLS solution
        # case num set to 500, each has 50 instances
        Run_one_case.run_one_case(case, path, samp_num=50, sol=1)
