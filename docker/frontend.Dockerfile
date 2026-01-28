FROM node:18-alpine

WORKDIR /app

# -------------------------
# Install dependencies
# -------------------------
COPY frontend/package*.json ./
RUN npm install

# -------------------------
# Copy source
# -------------------------
COPY frontend .

# -------------------------
# Build frontend
# -------------------------
RUN npm run build

# -------------------------
# Serve with preview
# -------------------------
EXPOSE 4173

CMD ["npm", "run", "preview", "--", "--host"]
