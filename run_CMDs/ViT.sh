#!/usr/bin/env bash

%runscript
  exec "$@"

cd /VLFAT
echo "Where am I ..."
ls
nvidia-smi


python main/Smain.py --config_path config/YML_files/VLFAT.yaml