FROM node:18-alpine as Assets
ENV NODE_ENV=production
ENV NODE_OPTIONS="--openssl-legacy-provider"
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM golang:1.19.0-buster AS GoBuilder
RUN sed -i -E "s/\w+.debian.org/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y upx
WORKDIR /app
COPY --from=Assets /app/build  /app/assets
COPY docker/standalone/*  ./
ENV GO111MODULE=on
ENV CGO_ENABLED=0
ENV GOPROXY=https://goproxy.cn
RUN go build -ldflags "-w -s" -o web main.go && \
    upx -9 -o web.minify web

FROM scratch
COPY --from=GoBuilder /app/web.minify /web