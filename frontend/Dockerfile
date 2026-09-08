# build จากภายใน frontend/ (docker compose ตั้ง context ไว้ที่ frontend/ อยู่แล้ว):
#     docker build -t sunseg-frontend .
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# เสิร์ฟ static bundle ด้วย nginx — เบากว่ารัน Node ใน production มาก และเป็นตัวที่
# proxy คำขอ /api ไปยัง backend container ให้ browser เห็นเป็น origin เดียวกัน (ไม่ต้อง
# ตั้ง CORS ฝั่ง FastAPI เลย เหมือนกับที่ vite dev server proxy ให้ตอนพัฒนา)
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
