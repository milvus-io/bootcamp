FROM milvusbootcamp/one-step-img-search:server-2.1.0 AS Server
FROM milvusbootcamp/one-step-img-search:client-2.1.0 AS Client

FROM milvusbootcamp/one-step-img-search:milvus-2.1.0

RUN apt update && apt install supervisor -y && \
    apt-get remove --purge -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*
COPY --from=Client /web                                     /app/client/
COPY --from=Server /app                                     /app/server
COPY --from=Server /usr/local/lib/python3.9/site-packages   /usr/local/lib/python3.9/
COPY --from=Server /root/.cache/torch                       /root/.cache/torch
COPY --from=Server /root/.towhee                            /root/.towhee

SHELL ["/bin/bash", "-c"]

RUN echo $' \n\
[unix_http_server] \n\
file=/var/run/supervisor.sock \n\
chmod=0700 \n\

[supervisord] \n\
nodaemon=true \n\
logfile=/var/log/supervisor/supervisord.log \n\
pidfile=/var/run/supervisord.pid \n\
childlogdir=/var/log/supervisor \n\

[rpcinterface:supervisor] \n\
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface \n\

[supervisorctl] \n\
serverurl=unix:///var/run/supervisor.sock \n\

[program:milvus] \n\
command=/entrypoint.sh \n\

[program:server] \n\
directory=/app/server \n\
command=python main.py \n\

[program:client] \n\
command=/app/client/web \n\

'> /etc/supervisor/supervisord.conf

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]
