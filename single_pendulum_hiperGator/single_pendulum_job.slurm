#!/bin/bash
#SBATCH --job-name=pendulum_job        # Job name
#SBATCH --output=pendulum_output.log   # Standard output log
#SBATCH --error=pendulum_error.log     # Error log for standard error
#SBATCH --ntasks=1                     # Number of CPU tasks (1 CPU)
#SBATCH --time=01:00:00                # Time limit hrs:min:sec (1 hour)
#SBATCH --mem=2gb                      # Memory required
#SBATCH --partition=standard           # Partition to run the job on

# Email notifications
#SBATCH --mail-user=sundaran.sukanth@ufl.edu    # Your email address
#SBATCH --mail-type=ALL                         # Send email on all job events (start, end, and fail)

# Load necessary modules (ensure correct versions are loaded)
module load libgmp/6.1.2
module load gsl
module load mpfr/3.1.1
module load glpk

# Change to the directory where your executable is located
cd /home/sundaran.sukanth/single_pendulum

# Run the executable with its arguments
./single_pendulum 1 1.2 -1.2 -1

