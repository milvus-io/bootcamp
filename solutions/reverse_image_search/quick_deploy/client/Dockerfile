# => Build container
FROM node:18-alpine as builder
ENV NODE_ENV=production
ENV NODE_OPTIONS="--openssl-legacy-provider"
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# => Run container
FROM nginx:1.23.1-alpine

# Nginx config
RUN rm -rf /etc/nginx/conf.d
COPY conf /etc/nginx

# Static build
COPY --from=builder /app/build /usr/share/nginx/html/

# Default port exposure
EXPOSE 80

# Copy .env file and shell script to container
WORKDIR /usr/share/nginx/html
COPY ./env.sh .
COPY .env .

# Add bash
RUN apk add --no-cache bash

# Make our shell script executable
RUN chmod +x env.sh

# Start Nginx server
CMD ["/bin/bash", "-c", "/usr/share/nginx/html/env.sh && nginx -g \"daemon off;\""]