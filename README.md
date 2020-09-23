# JSTOR Import

Used to import JSTOR files to elasticsearch instance

### JSTOR import with Embeddings

Current implementation creates BERT Embeddings for sentences with specific words in it.

#### Use with Agave Cluster
To run the script in Agave Cluster:
1. Create an account in Agave: https://cores.research.asu.edu/research-computing/get-started/create-an-account
2. Once you have a login on rc, upload the JSTOR directories (zipped) to Scratch directory
3. Create an environment
```
conda create -n pytorch-jstor -c anaconda -c conda-forge pytorch==1.6.0 transformers
module load anaconda3/5.3.0
source activate pytorch-jstor
```
4. Create the following sbatch file along with the code, change variable as necessary

**SBATCH Script:**
```
#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -c 16  # number of "tasks" (cores)
#SBATCH --mem=16G        # GigaBytes of memory required (per node)
#SBATCH -t 0-05:00:00   # time in d-hh:mm:ss
#SBATCH -p gpu       # partition
#SBATCH -q wildfire       # QOS
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=avtale@asu.edu # Mail-to address

# Always purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
module load anaconda3/5.3.0
source activate /home/avtale/.conda/envs/pytorch-jstor #change to your path

cd ~/jstor/code #cd to your directory
python JSTOR-Elasticsearch-with-embeddings.py

mail -s "SBATCH Completed" avtale@asu.edu << EOF
Hello me,
The JSTOR Upload job completed.
EOF
```
5. Run the sbatch with command sbatch followed by your sbatch file name