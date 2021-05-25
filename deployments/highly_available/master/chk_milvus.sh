#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <master-docker-id>)


if [[ "${RUNNING_STATUS}" != "true" ]];then
    exit 1
fi

