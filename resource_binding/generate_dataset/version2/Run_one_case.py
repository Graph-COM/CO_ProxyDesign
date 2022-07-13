import random
import itertools
from string import Template
import subprocess
import json
import math
import re
from time import sleep
import time
import os, signal

timeout = 900  # if one case is over 15 minutes, it's considered a failure 


def get_LUT_op_list(directives):
    LUT_op_list = []
    for direct in directives:
        if direct.startswith('#'):
            continue
        res = list(map(int, re.findall(r'\d+', direct)))
        if len(res) == 2:
            LUT_op_list.append(res[1])
    return LUT_op_list


def run_one_case(case_id, path, samp_num=100, sol=1):
    ## Get script

    print("Running case %d" % case_id)
    f_script = open(path + 'script.tcl', 'r')
    script_template = f_script.read()
    f_script.close()

    f_directive = open(path + 'directive.tcl', 'r')
    directive_all = f_directive.read()
    f_directive.close()

    directive_list = []
    for line in directive_all.splitlines():
        directive_list.append(line)

    all_lists = []
    ## THIS IS THE VIVADO HLS BASELINE
    if sol == 1:
        all_lists.append([])

    for i in range(1, samp_num):
        direct_list_sampled = []
        for directive in directive_list:
            ri = random.uniform(0, i % 17 + 4)
            if ri <= 3:
                directive = "# " + directive
            direct_list_sampled.append(directive)
        all_lists.append(direct_list_sampled)

    print("Generated %d combinations of directives" % len(all_lists))

    json_file = path + "case_" + str(case_id) + "_all_data.json"
    if sol == 1:
        all_solutions = {}
        f_json = open(json_file, "w")
        json.dump(all_solutions, f_json)
        f_json.close()
    
    for directives in all_lists:
        print("Generating vivado HLS directive files for solution_%d" %
                sol)
        f_direct_name = "./directive_tmp%d.tcl" % case_id
        f_direct = open(f_direct_name, "w")
        for ele in directives:
            f_direct.write(ele + "\n")
        f_direct.close()
        f_script_name = "./script_tmp%d.tcl" % case_id
        f_script = open(f_script_name, "w")
        #script_content = script_template.substitute(sol = str(sol), f_directive = f_direct_name)
        script_content = script_template.replace(
            'solution_',
            'solution_tmp').replace('directive',
                                    'directive_tmp%d' % case_id).replace(
                                        'case_', 'case_' + str(case_id))
        script_content = script_content.replace('PATH', path).replace(
            'project_', 'project_%d' % case_id)
        f_script.write(script_content)
        f_script.close()

        print("Running Vivado HLS for solution_%d" % sol)
        # subprocess.call(['/home/xilinx/Vitis/Vivado/2020.1/bin/vivado_hls', '-f', f_script_name])
        try:
            start_time = time.time()
            # replace your vitis/vivado environment
            p = subprocess.Popen([
                'vitis_hls', '-f',
                f_script_name
            ],
                                    close_fds=True,
                                    preexec_fn=os.setsid)
            while True:
                if p.poll() is not None:
                    break
                seconds_passed = time.time() - start_time
                if timeout and seconds_passed > timeout:
                    print(p.pid)
                    os.killpg(p.pid, signal.SIGUSR1)
                    raise TimeoutError
                time.sleep(2)
        except TimeoutError:
            print("Running timeout for process_%s (15 minutes), and we skipped it" % case_id)
            break
        except BaseException:
            print("Process_%s failed, and we skipped it" % case_id)
            break

        rpt_name = 'project_%d/solution_tmp/impl/report/verilog/case_%d_export.rpt' % (
            case_id, case_id)
        f_rpt = open(rpt_name, 'r')
        SLICE = LUT = FF = DSP = CP = 0
        for line in f_rpt.readlines():
            res = [i for i in line.split() if i.isdigit()]
            if line.startswith('SLICE'):
                SLICE = int(res[0])
            elif line.startswith('LUT'):
                LUT = int(res[0])
            elif line.startswith('FF'):
                FF = int(res[0])
            elif line.startswith('DSP'):
                DSP = int(res[0])
            elif line.startswith('CP achieved'):
                res = [i for i in line.split()]
                CP = float(res[3])
        print(SLICE, LUT, FF, DSP, CP)

        f_json = open(json_file, "r")
        all_solutions = json.load(f_json)
        f_json.close()

        all_solutions["solution_" + str(sol)] = {}
        sol_tb = all_solutions["solution_" + str(sol)]
        sol_tb['directives'] = directives
        sol_tb['LUT_op'] = get_LUT_op_list(directives)
        sol_tb['SLICE'] = int(SLICE)
        sol_tb['LUT'] = int(LUT)
        sol_tb['FF'] = int(FF)
        sol_tb['DSP'] = int(DSP)
        sol_tb['CP'] = float(CP)

        f_json = open(json_file, "w")
        json.dump(all_solutions, f_json)
        f_json.close()

        sol = sol + 1
    

def main():
    run_one_case(3,
                 "/home/congh/HLS_ML/data_collect/HLS/case_3/",
                 samp_num=7)


if __name__ == "__main__":
    main()
