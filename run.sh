#!/bin/bash

isPackageNotInstalled() {
  $1 --version &> /dev/null
  if [ $? -eq 0 ]; then
    echo "$1: Already installed"
  else
    install_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    bash Anaconda3-2022.05-Linux-x86_64.sh -b -p  $install_dir
    export PATH=$install_dir/bin:$PATH
  fi
}


if [ $1 == "--cloud" ]; then
    isPackageNotInstalled conda
    if ! { conda env list | grep 'RUN_ENV'; } >/dev/null 2>&1; then
        conda create --name nf python=3.9
    fi
    conda activate nf
    install --upgrade pip
    pip install -r requirements.txt
fi

python -m experiments.train
