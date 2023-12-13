#!/home/mitch/anaconda3/envs/Thesis/bin/python

import sys
import os

# Uncomment to allow lib package to be imported when running from command line
# cwd = os.getcwd()
# sys.path.append(os.path.dirname(cwd))


from Opt_Model_Defs import vanilla_IP
import yaml

with open('config.yaml',"r") as f:
    opt_params = yaml.safe_load(f)

model = vanilla_IP(opt_params)
model.solve_all_instances()
