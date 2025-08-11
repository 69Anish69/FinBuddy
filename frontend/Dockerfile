# ---------------- Build Stage ----------------
FROM node:18-bullseye AS build

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

# ---------------- Production Stage ----------------
FROM nginx:stable

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d

# ✅ Fix: Vite outputs to 'dist', not 'build'
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
