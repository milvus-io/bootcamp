FROM milvusbootcamp/one-step-img-search:server-2.1.0 AS Server
FROM milvusbootcamp/one-step-img-search:client-2.1.0 AS Client

FROM tsl0922/ttyd:latest AS Console

FROM milvusbootcamp/one-step-img-search:milvus-2.1.0
RUN apt update && apt install supervisor -y && \
    apt-get remove --purge -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*
COPY --from=Client  /web                                     /app/client/
COPY --from=Server  /app                                     /app/server
COPY --from=Console /usr/bin/ttyd                            /app/console/ttyd
COPY --from=Server  /usr/local/lib/python3.9/site-packages   /usr/local/lib/python3.9/
COPY --from=Server  /root/.cache/torch                       /root/.cache/torch
COPY --from=Server  /root/.towhee                            /root/.towhee

SHELL ["/bin/bash", "-c"]

RUN echo $' \n\
[unix_http_server] \n\
file=/var/run/supervisor.sock \n\
chmod=0700 \n\

[inet_http_server] \n\
port=0.0.0.0:8080 \n\

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

[program:console] \n\
command=/app/console/ttyd --port=8090 bash \n\

'> /etc/supervisor/supervisord.conf

COPY client/src/containers/Logo.svg /usr/lib/python2.7/dist-packages/supervisor/ui/images/logo.svg
RUN sed -i -E "s/\"images\/supervisor.gif\"/\"images\/logo.svg\"/g" /usr/lib/python2.7/dist-packages/supervisor/ui/status.html
RUN sed -i -E "s/alt=\"Supervisor status\"/style=\"width: 120px;\"/g" /usr/lib/python2.7/dist-packages/supervisor/ui/status.html
RUN sed -i -E "s/<body>/<body style=\"background: #212121;\">/g" /usr/lib/python2.7/dist-packages/supervisor/ui/status.html
RUN sed -i -E "s/tailbody.content(tail)/tail.decode(encoding=\"utf8\",errors=\"ignore\")/g" /usr/lib/python2.7/dist-packages/supervisor/web.py

CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]
