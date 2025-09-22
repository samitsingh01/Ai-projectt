FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json public/ src/ ./

ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL

RUN npm ci --only=production && \
    npm cache clean --force && \
    npm run build

FROM nginx:1.25-alpine

COPY --from=builder /app/build /usr/share/nginx/html
RUN echo 'server { \
    listen 3000; \
    server_name localhost; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
    location /api/ { \
        proxy_pass http://api-gateway:8000/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    location /health { \
        access_log off; \
        return 200 "healthy\\n"; \
        add_header Content-Type text/plain; \
    } \
}' > /etc/nginx/conf.d/default.conf && \
    rm -f /etc/nginx/conf.d/default.conf.template && \
    addgroup -g 1001 -S nginxuser && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginxuser nginxuser && \
    chown -R nginxuser:nginxuser /usr/share/nginx/html /var/cache/nginx /var/log/nginx /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown nginxuser:nginxuser /var/run/nginx.pid

USER nginxuser
EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
