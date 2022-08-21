#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --models unet=insurance_v2.mar --ts-config /home/model-serve/docker/serve/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
