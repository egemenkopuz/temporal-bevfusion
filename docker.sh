#!/bin/bash

if [ -z "$2" ]; then
	echo "Please provide either 'dev' or 'prod' as the second argument" && exit 1 || return 1
fi

if [ $2 = "dev" ]; then
	if [ -z "$3" ]; then
		echo "Container prefix is not provided, using plain 'bevfusion' as the prefix"
		CONTAINER_NAME="bevfusion-dev"
	else
		CONTAINER_NAME="$3-bevfusion-dev"
	fi
	IMAGE_NAME="bevfusion:dev"
	DOCKERFILE="Dockerfile.dev"
	DOCKER_COMMAND="docker"

	# add or remove arguments according to your needs
	RUN_COMMAND_ARGS="--name $CONTAINER_NAME \
                -v $(pwd):/root/mmdet3d \
                -v $(pwd)/data:/dataset \
                --env=DISPLAY \
                --gpus all \
				--shm-size 16g"

	RUN_COMMAND_ARGS+=" -d -it $IMAGE_NAME"

elif [ $2 = "prod" ]; then
	if [ -z "$3" ]; then
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

if [ $1 == "run" ]; then
	echo "Running docker container with the following arguments:"
	echo $RUN_COMMAND_ARGS
	$DOCKER_COMMAND run $RUN_COMMAND_ARGS
elif [ $1 = "build" ]; then
	$DOCKER_COMMAND build -f $DOCKERFILE -t $IMAGE_NAME .
elif [ $1 = "access" ] || [ $1 == "exec" ]; then
	$DOCKER_COMMAND exec -it $CONTAINER_NAME bash
elif [ $1 = "start" ]; then
	$DOCKER_COMMAND start $CONTAINER_NAME
elif [ $1 = "stop" ]; then
	$DOCKER_COMMAND container stop $CONTAINER_NAME
elif [ $1 = "remove-container" ]; then
	if [ "$($DOCKER_COMMAND container inspect -f '{{.State.Running}}' $CONTAINER_NAME)" = "true" ]; then
		read -p "Container '$CONTAINER_NAME' is running, would you like to stop it first (y/n)? " -n 1 -r
		echo
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			$DOCKER_COMMAND container stop $CONTAINER_NAME
		else
			exit 1
		fi
	fi

	$DOCKER_COMMAND container rm $CONTAINER_NAME
elif [ $1 = "remove-image" ]; then

	$DOCKER_COMMAND image rm $IMAGE_NAME
elif [ $1 = "remove-all" ]; then
	if [ "$($DOCKER_COMMAND container inspect -f '{{.State.Status}}' $CONTAINER_NAME)" = "running" ]; then
		read -p "Container '$CONTAINER_NAME' is running, would you like to stop it first (y/n)? " -n 1 -r
		echo
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			$DOCKER_COMMAND container stop $CONTAINER_NAME
		else
			exit 1
		fi
	fi

	$DOCKER_COMMAND container rm $CONTAINER_NAME
	$DOCKER_COMMAND image rm $IMAGE_NAME
else
	echo "Invalid argument for first argument (build, run, access, exec, start, stop, remove-container, remove-image, remove-all)"
fi
