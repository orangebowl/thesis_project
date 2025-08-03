#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general'
#SBATCH --qos=short # Request Quality of Service. Default is'short' (maximum run time: 4 hours)
#SBATCH --time=4:00:00 # Request run time (wall-clock). Default is 1minute
#SBATCH --ntasks=1 # Request number of parallel tasks per job.Default is 1
#SBATCH --cpus-per-task=1 # Request number of CPUs (threads) per task.Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=16384 # Request memory (MB) per node. Default is1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END # Set mail type to 'END' to receive a mailwhen the job finishes.
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm
jobId
#SBATCH --error=slurm_%j.err # Set name of error log. %j is the SlurmjobId
#SBATCH --gres=gpu:1
# Measure GPU usage of your job (initialization)
# previous=$(/usr/bin/nvidia-smi --query-accountedapps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
# /usr/bin/nvidia-smi # Check sbatch settings are working (it should showthe GPU that you requested)
module use /opt/insy/modulefiles
module load cuda/12.4 cudnn/12-8.9.1.23 miniconda/3.9
conda activate /tudelft.net/home/nfs/wanxinchen environment.yml
srun python /home/nfs/wanxinchen/thesis_project/main_pou_recursive.py
# /usr/bin/nvidia-smi --query-accountedapps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"