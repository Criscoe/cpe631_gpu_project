#!/usr/bin/env bash

for i in $(seq 1 100 1001);
do
    python3 ./spitballScripts/cudaCpuVsGpuNumItr.py --min-iteration $i --max-iterations $i
    mv outData.txt tmp/outData_$i.txt 
    python3 ./spitballScripts/cudaCpuVsGpuNumItr.py --min-iteration $i --max-iterations $i --gauss
    mv outData.txt tmp/outDataG_$i.txt 
    echo $i
done

python3 ./spitballScripts/cudaReader.py