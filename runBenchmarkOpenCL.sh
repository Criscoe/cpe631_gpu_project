#!/usr/bin/env bash

for i in $(seq 1000 1 1000);
do
    python3 ./spitballScripts/cudaCpuVsGpuNumItrOpenCL.py --min-iteration $i --max-iterations $i --full-image-path /home/seeshadmin/repos/cpe631_gpu_project/images/sample_6000x4000.jpg
    mv outData.txt tmp/outData_$i.txt 
    # python3 ./spitballScripts/cudaCpuVsGpuNumItrOpenCL.py --min-iteration $i --max-iterations $i --gauss
    # mv outData.txt tmp/outDataG_$i.txt 
    echo $i
done

python3 ./spitballScripts/cudaReader.py