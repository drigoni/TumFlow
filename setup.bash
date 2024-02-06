#!/usr/bin/env bash

s1=$1
s2="install"
s3="download"
s4="uninstall"

if [[ $s1 == $s2 ]]; then
    echo "------------------------------------------------------------------------------------------------------------"
    echo "---------------------------------------------------  TumFlow  -----------------------------------------------"
    echo "------------------------------------------------------------------------------------------------------------"
    conda env create -f tumflow_env.yml
    conda activate tumflow
    # pip install -r tumflow_env.txt
    # conda deactivate
elif [[ $s1 == $s3 ]]; then
          echo "-------------------------------------------------------------------------------------------------------------"
          echo "---------------------------------------------------  DATA --------------------------------------------------"
          echo "-------------------------------------------------------------------------------------------------------------"
          cd data
          pip install gdown
          gdown 'https://docs.google.com/uc?export=download&id=1cXZnCMD5NId8QPFjWwkMynLzzwZ1zJ6T' -O 'melanoma_skmel28.zip'
          unzip -a melanoma_skmel28.zip
          rm melanoma_skmel28.zip
          cd ..
elif [[ $s1 == $s4 ]]; then
               conda deactivate
               conda remove -n tumflow --all
else
     echo Use "install" or "remove"
fi