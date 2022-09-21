FROM node:18-alpine as Assets
ENV NODE_ENV=production
ENV NODE_OPTIONS="--openssl-legacy-provider"
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.23.1-alpine
RUN rm -rf /etc/nginx/conf.d
COPY nginx/conf /etc/nginx
COPY --from=Assets /app/build /usr/share/nginx/html/
EXPOSE 80
WORKDIR /usr/share/nginx/html
COPY ./env.sh .
COPY .env .
RUN apk add --no-cache bash
RUN chmod +x env.sh
CMD ["/bin/bash", "-c", "/usr/share/nginx/html/env.sh && nginx -g \"daemon off;\""]