import os

for i in range(100):
    os.system("python config_gather_GCAS.py --random_seed 2021%03d"%(i))