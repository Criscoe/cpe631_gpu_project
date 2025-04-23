#!/usr/bin/env bash
array_name=(images/sample_80x45.jpg images/sample_1920x1080.jpg images/sample.jpg images/sample_6000x4000.jpg)
for element in "${array_name[@]}";
do
    python3 ./spitballScripts/cudaCpuVsGpuNumItr.py --min-iteration 100 --max-iterations 100 --full-image-path $element
    mv outData.txt tmp2/outData_$i.txt 
    python3 ./spitballScripts/cudaCpuVsGpuNumItr.py --min-iteration 100 --max-iterations 100 --gauss --full-image-path $element
    mv outData.txt tmp2/outDataG_$i.txt 
    echo $i
done

python3 ./spitballScripts/cudaReader2.py