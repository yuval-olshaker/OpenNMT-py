import os
import sys

if len(sys.argv) < 3:
    exit(1)

exp_name = sys.argv[1]
best_model_num = sys.argv[2]

base_path = '/home/ubuntu/wasm_decompiler/Codenator/' + exp_name + '/models/'
saved_models = os.listdir(base_path)
for model in saved_models:
    if best_model_num not in model:
        os.system('rm ' + base_path + model)
