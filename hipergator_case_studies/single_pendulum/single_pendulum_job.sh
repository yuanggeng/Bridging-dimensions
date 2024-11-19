#!/bin/bash

#SBATCH --job-name=pendulum_parallel    # Job name
#SBATCH --output=batch_%A_%a_output.log # Output log for each array job (%A is job ID, %a is array task ID)
#SBATCH --error=batch_%A_%a_error.log   # Error log for each array job
#SBATCH --ntasks=10                     # 10 tasks per batch
#SBATCH --nodes=10                      # 1 task per node
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=hpg-default         # Partition to run the job on
#SBATCH --cpus-per-task=1               # 1 CPU per task
#SBATCH --mem=5gb                       # Memory per task
#SBATCH --array=0-9                     # Array job with 10 batches
# Email notifications
#SBATCH --mail-user=sundaran.sukanth@ufl.edu    
#SBATCH --mail-type=ALL               

# Load necessary modules
module load python/3.10
module load libgmp/6.1.2 gsl/2.6 libmpfr/3.1.1 glpk
export LD_LIBRARY_PATH=/apps/gsl/2.6/lib:$LD_LIBRARY_PATH

cd /home/sundaran.sukanth/single_pendulum

# Generate sub-square coordinates using the Python script
sub_squares=$(python generate_subsquares.py)
IFS=$'\n' read -d '' -r -a sub_squares_array <<< "$sub_squares"

# Calculate the starting and ending index for the current batch
batch_size=10
start_index=$((SLURM_ARRAY_TASK_ID * batch_size))
end_index=$((start_index + batch_size - 1))

echo "Running batch ${SLURM_ARRAY_TASK_ID}: Tasks from $start_index to $end_index"

# Run each sub-square task in the current batch
for i in $(seq $start_index $end_index); do
    square="${sub_squares_array[$i]}"
    echo "Launching task $i with parameters: $square"
    srun --ntasks=1 --nodes=1 ./single_pendulum $square > "task_${i}.log" 2>&1 &
done

# Wait for all tasks in the batch to complete
wait

echo "Batch ${SLURM_ARRAY_TASK_ID} completed at $(date)"
