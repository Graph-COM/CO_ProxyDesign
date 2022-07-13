from Run_one_case import *


all_cases = 500
for case in range(1, 1 + all_cases):
	path = "/home/congh/HLS_ML/data_collect/HLS/case_" + str(case) + "/"
	# solution_1 is the default HLS solution
	run_one_case(case, path, samp_num = 50, sol = 1)
