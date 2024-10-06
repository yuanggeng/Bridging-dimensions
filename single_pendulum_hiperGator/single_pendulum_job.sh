#!/bin/bash
#SBATCH --job-name=pendulum_job        # Job name
#SBATCH --output=pendulum_output.log   # Standard output log
#SBATCH --error=pendulum_error.log     # Error log for standard error
#SBATCH --ntasks=1                     # Number of CPU tasks (1 CPU)
#SBATCH --time=01:00:00                # Time limit hrs:min:sec (1 hour)
#SBATCH --mem=2gb                      # Memory required
#SBATCH --partition=hpg-default           # Partition to run the job on


# Email notifications
#SBATCH --mail-user=sundaran.sukanth@ufl.edu    # Your email address
#SBATCH --mail-type=ALL                         # Send email on all job events (start, end, and fail)



# Load necessary modules (ensure correct versions are loaded)
module load libgmp/6.1.2 gsl/2.6 libmpfr/3.1.1 glpk

#module load libgmp/6.1.2
#module load mpfr/4.2.1
#module load gsl
#module load mpfr/3.1.6
#module load glpk

export LD_LIBRARY_PATH=/apps/gsl/2.6/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/apps/intel/2020.0.166/gsl/2.7/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/apps/intel/2020.0.166/gsl/2.7/lib:$LD_LIBRARY_PATH


# Change to the directory where your executable is located
cd /home/sundaran.sukanth/single_pendulum

# Run the executable with its arguments
./single_pendulum 1 1.2 -1.2 -1

