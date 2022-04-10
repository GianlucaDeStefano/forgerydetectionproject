import subprocess
import time

scripts_to_execute = ['TransferabilityNoiseprint2ExifColumbia.py','TransferabilityNoiseprint2ExifDSO.py','TransferabilityExif2NoiseprintColumbia.py','TransferabilityExif2NoiseprintDSO.py']

for script in scripts_to_execute:
    try:
        p1 = subprocess.call(f"python {script}",shell=True)
    except Exception as e:
        print(e)
        continue