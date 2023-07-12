## Installation
### Create singularity file
For this project we are using singularity. In the following you can find the necessary information to set up your `.sif` file.
The `.txt` file (e.g. `sing_def.txt`) for creating the `.sif` file:
```
BootStrap: library
From: ubuntu:20.04

%environment

%files
    /path-on-host /path-in-container

%post
    apt -y update
    apt -y upgrade
    DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common build-essential cmake wget nano
    add-apt-repository universe
    apt -y update
    apt -u install python3.8
    ln -s /usr/bin/python3.8 /usr/bin/python
    apt-get update -y
    apt-get -y install python3-pip
    apt-get update
    apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
    apt install rustc -y
    apt-get install libomp-dev -y
    apt update
    apt install hwloc -y
    apt-get -y install cmake
    pip install -r /path2requirements/requirements.txt
```
Then run the following command: 
```shell
sudo singularity build sing_file.sif sing_def.txt
```

### Run a job
To use the created singularity file, you shoule run a command similar to: 
```shell
singularity exec --nv --bind path-on-host/VLFAT:path-in-container/VLFAT sing_file.sif /bin/bash /VLFAT/run_CMDs/ViT.sh
```
On a SLURM enabled server, you can use the `run_CMDs/ViT.job` with the following command :
```shell
sbatch ViT.job
```