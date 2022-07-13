from Gen_one_case import *
import os
import subprocess


def gen_all_cases(path, case_num=50):
    if not os.path.exists(path):
        subprocess.call(["mkdir", path])
    os.chdir(path)

    for case in range(1, 1 + case_num):
        Gen_one_case(case_id=case, max_prim_in=30, max_op_cnt=200)
        case_dir = "case_" + str(case)

        subprocess.call(["rm", "-rf", case_dir])
        subprocess.call(["mkdir", case_dir])

        subprocess.call(["mv", "DFG_" + case_dir + ".txt", case_dir])
        subprocess.call(["mv", case_dir + ".cc", case_dir])
        subprocess.call(["mv", "directive.tcl", case_dir])
        subprocess.call(["mv", "script.tcl", case_dir])

if __name__=='__main__':
    Gen_one_case(case_id=1, max_prim_in=30, max_op_cnt=200)