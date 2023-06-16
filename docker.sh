#!/bin/bash

if [ $1 = "run-local" ]
then
        docker run --name "egemen-bevfusion" \
                -v $(pwd):/root/mmdet3d \
                -v $(pwd)/data:/dataset \
                -v /mnt/c/data:/mnt/c/data/ \
                --env="DISPLAY" \
                --gpus all \
                -d -it "egemen/bevfusion:egemen";
elif [ $1 = "run-tum" ]
then
        nvidia-docker run --name "egemen-bevfusion" \
                -v $(pwd):/root/mmdet3d \
                -v $(pwd)/data:/dataset \
                -v /mnt/ssd_1tb_samsung/datasets/:/mnt/ssd_1tb_samsung/datasets/ \
                --env="DISPLAY" \
                --shm-size 16g \
                -d -it "egemen/bevfusion:egemen";
elif [ $1 = "run-setlabs" ]
then
        docker run --name "egemen-bevfusion" \
                -v $(pwd):/root/mmdet3d \
                -v $(pwd)/data:/dataset \
                -v /mnt/Drive/datasets/:/mnt/Drive/datasets/ \
                --env="DISPLAY" \
                --shm-size 16g \
                --gpus all \
                -d -it "egemen/bevfusion:egemen";
elif [ $1 = "exec" ]
then
        docker exec -it egemen-bevfusion bash
elif [ $1 = "start" ]
then
        docker start egemen-bevfusion
elif [ $1 = "stop" ]
then
        docker container stop egemen-bevfusion
elif [ $1 = "remove" ]
then
        docker container rm egemen-bevfusion
elif [ $1 = "build"  ]
then
        docker build -t "egemen/bevfusion:egemen" .
else
        echo "Invalid argument"
fi
