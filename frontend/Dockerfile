FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE=http://localhost:8800
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# Runs as non-root on 8080 by default - matches this repo's security posture
# without extra Dockerfile config.
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
