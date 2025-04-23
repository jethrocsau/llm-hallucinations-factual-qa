#intearctive job
srun --partition normal --account=mscbdt2024 --nodes=1 --gpus-per-node=1  --time 0-00:30:00 --pty bash

# batch job
sbatch job.sh --partition normal --account=mscbdt2024 --username csauac

# check queue
squeue -u csauac

#  cancel job
scancel <job_id>

# view .out
cat job-<job_id>.out

# watch .out
tail -f job-<job_id>.out
