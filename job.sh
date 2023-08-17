#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/tmp/job-%j.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=gianluca.de-stefano@cispa.de
#SBATCH --time 48:00:00

#SBATCH --partition=gpu --gres=gpu:A100:1 --cpus-per-gpu=32

JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "48:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

srun mkdir "$JOBTMPDIR"

srun --container-mounts="$JOBTMPDIR":/tmp --container-image=projects.cispa.saarland:5005#c01gide/tesi:tf2 python /home/c01gide/CISPA-home/tesi/attack-samples.py -d columbia -t exif -f /home/c01gide/CISPA-projects/llm_security_triggers-2023/tesi_archive/alpha_5/risultati/Data/Attacks/SingleAttacks/Attack-exif-columbia &

srun mv /tmp/job-"$SLURM_JOB_ID".out "$JOBDATADIR"/out.txt
srun mv "$JOBTMPDIR" "$JOBDATADIR"/data

