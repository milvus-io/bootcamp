#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' c5154606b258)


if [[ "${RUNNING_STATUS}" != "true" ]];then
    exit 1
fi

