FROM node:20-alpine AS webbuild
WORKDIR /src
COPY web/package.json web/package.json
COPY web/vite.config.js web/vite.config.js
COPY web/index.html web/index.html
COPY web/src web/src
RUN cd web && npm install && npm run build

FROM node:20-alpine
WORKDIR /app

COPY gateway/package.json ./package.json
RUN npm install --omit=dev

COPY gateway/server.js ./server.js
COPY --from=webbuild /src/web/dist ./web/dist

EXPOSE 3000
CMD ["node", "server.js"]
