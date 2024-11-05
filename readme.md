# RADICAL-Pilot (RP) applications for LUCID pipelines

## Cell Painting Pipeline

Example of RP app for the current implementation of the Cell Painting Pipeline
([run_cell_outline.sh](https://github.com/abestroka/JUMP_vision_model/blob/main/rad_pipeline/run_cell_outline.sh))

```shell
module use /soft/modulefiles; module load conda
eval "$(conda shell.posix hook)"
conda create -y -n ve.rp python=3.9 radical.pilot
```

```shell
# workspace
mkdir lucid; cd lucid
git clone https://github.com/abestroka/JUMP_vision_model.git 
git clone https://github.com/mtitov/lucid-radical-pipelines.git wfms

cd wfms/src
# ensure that virtual environment is active
#   module use /soft/modulefiles; module load conda
#   eval "$(conda shell.posix hook)"
#   conda activate ve.rp
nohup python3 cell.rp.py --work_dir "../../" > OUTPUT 2>&1 </dev/null &
```

