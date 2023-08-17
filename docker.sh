#!/bin/bash

if [ -z "$2" ]
then
        echo "Please provide either 'dev' or 'prod' as the second argument" && exit 1 || return 1
fi

if [ $2 = "dev" ]
then
        if [ -z "$3" ]
        then
                echo "Container prefix is not provided, using plain 'bevfusion' as the prefix"
                CONTAINER_NAME="bevfusion-dev"
        else
                CONTAINER_NAME="$3-bevfusion-dev"
        fi
        IMAGE_NAME="bevfusion:dev"
        DOCKERFILE="Dockerfile.dev"

        # add or remove arguments according to your needs
        RUN_COMMAND_ARGS="--name $CONTAINER_NAME \
                -v $(pwd):/root/mmdet3d \
                -v $(pwd)/data:/dataset \
                --env=DISPLAY \
                --gpus all \
                --shm-size 8g"

        if [ $1 = "run-tum" ]
        then
        # add additional volumes for TUM
        RUN_COMMAND_ARGS+=" -v /mnt/ssd_4tb_samsung/datasets/:/mnt/ssd_4tb_samsung/datasets/ \
                -v /mnt/ssd_4tb_samsung/egemen/:/mnt/ssd_4tb_samsung/egemen/"
        elif [ $1 = "run-setlabs" ]
        then
        # add additional volumes for SetLabs
        RUN_COMMAND_ARGS+=" -v /mnt/Drive/datasets/:/mnt/Drive/datasets/ \
                -v /mnt/Drive/egemen/:/mnt/Drive/egemen/"
        fi
        RUN_COMMAND_ARGS+=" -d -it $IMAGE_NAME"
elif [ $2 = "prod" ]
then
        if [ -z "$3" ]
        then
                echo "Container prefix is not provided, using plain 'bevfusion' as the prefix"
                CONTAINER_NAME="bevfusion-prod"
        else
                CONTAINER_NAME="$3-bevfusion-prod"
        fi
        IMAGE_NAME="bevfusion:prod"
        DOCKERFILE="Dockerfile.prod"

        # add or remove arguments according to your needs
        RUN_COMMAND_ARGS="--name $CONTAINER_NAME \
                -v $(pwd)/data:/dataset \
                --env=DISPLAY \
                --gpus all \
                --shm-size 8g \
                -d -it $IMAGE_NAME"
else
        echo "Invalid argument for second argument (dev or prod)" && exit 1 || return 1
fi

# running the commands

if  [ $1 == "run" ] || [ $1 == "run-tum" ] || [ $1 == "run-setlabs" ]
then
        echo "Running docker container with the following arguments:"
        echo $RUN_COMMAND_ARGS
        docker run $RUN_COMMAND_ARGS
elif [ $1 = "build"  ]
then
        docker build -f $DOCKERFILE -t $IMAGE_NAME .
elif [ $1 = "access" ] || [ $1 == "exec" ]
then
        docker exec -it $CONTAINER_NAME bash
elif [ $1 = "start" ]
then
        docker start $CONTAINER_NAME
elif [ $1 = "stop" ]
then
        docker container stop $CONTAINER_NAME
elif [ $1 = "remove-container" ]
then
        if [ "$( docker container inspect -f '{{.State.Running}}' $CONTAINER_NAME )" = "true" ];
        then
                read -p "Container '$CONTAINER_NAME' is running, would you like to stop it first (y/n)? " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]
                then
                        docker container stop $CONTAINER_NAME
                else
                        exit 1
                fi
        fi

        docker container rm $CONTAINER_NAME
elif [ $1 = "remove-image" ]
then

        docker image rm $IMAGE_NAME
elif [ $1 = "remove-all" ]
then
        if [ "$( docker container inspect -f '{{.State.Status}}' $CONTAINER_NAME )" = "running" ]; then
                read -p "Container '$CONTAINER_NAME' is running, would you like to stop it first (y/n)? " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]
                then
                        docker container stop $CONTAINER_NAME
                else
                        exit 1
                fi
        fi

        docker container rm $CONTAINER_NAME
        docker image rm $IMAGE_NAME
else
        echo "Invalid argument for first argument (build, run, run-tum, run-setlabs, access, exec, start, stop, remove-container, remove-image, remove-all)"
fi
