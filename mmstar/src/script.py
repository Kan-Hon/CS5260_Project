import subprocess
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
sh_path = os.path.join(dir_path, 'run_inference_answer.sh')
subprocess.call(['sh', sh_path])

