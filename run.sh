#!/bin/bash
# DEV/NONDEV DEVICE DATASET EXPERIMENT

COMMIT=$(git rev-parse --short HEAD)

MODE=$1

if [ $MODE == "DEV" ]
then 
    COMMAND="cd ../NeuralODE && pip install -r requirements.txt && wandb login 7aedec3427d399660c93a0ac0d61da08f861325b && python run.py -device $2 -dataset $3 -experiment $4 --dev"
else 
    COMMAND="cd ../NeuralODE && pip install -r requirements.txt && wandb login 7aedec3427d399660c93a0ac0d61da08f861325b && python run.py -device $2 -dataset $3 -experiment $4"
fi

nvidia-docker run --rm -v /raid/data/gmezentsev/mycode/NeuralODE:/NeuralODE \
                  --name gmezentsev_$COMMIT \
                  nvcr.io/nvidia/pytorch:21.07-py3 \
                  /bin/bash -c "$COMMAND"

echo "Finished"
