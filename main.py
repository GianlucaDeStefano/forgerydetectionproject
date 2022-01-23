import subprocess
import time

scripts_to_execute = ['test_NoiseprintDSOPristine.py', 'test_NoiseprintColumbiaPristine.py', 'test_ExifDSOPristine.py',
                      'test_ExifColumbiaPristine.py']

for script in scripts_to_execute:
    p1 = subprocess.call(f"python {script}",shell=True)