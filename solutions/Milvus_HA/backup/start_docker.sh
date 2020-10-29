#!/bin/bash
RUNNING_STATUS=$(docker inspect --format '{{.State.Running}}' <docker-id>)


if [[ "${RUNNING_STATUS}" != "true" ]];then
docker start <docker-id>
fi

