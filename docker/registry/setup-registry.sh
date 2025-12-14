#!/bin/bash
# Setup HTTPS Docker Registry with Let's Encrypt
#
# Usage:
#   ./setup-registry.sh <domain> <email>
#
# Example:
#   ./setup-registry.sh registry.example.com admin@example.com

set -e

DOMAIN="${1:?Usage: $0 <domain> <email>}"
EMAIL="${2:?Usage: $0 <domain> <email>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================"
echo "Docker Registry HTTPS Setup"
echo "========================================"
echo "Domain: ${DOMAIN}"
echo "Email:  ${EMAIL}"
echo "========================================"
echo ""

# Create certs directory
mkdir -p certs

# Update nginx.conf with domain
sed -i "s/\${REGISTRY_DOMAIN}/${DOMAIN}/g" nginx.conf

# Step 1: Start with temporary self-signed cert for initial nginx start
echo "[1/4] Creating temporary certificate..."
openssl req -x509 -nodes -days 1 -newkey rsa:2048 \
    -keyout certs/privkey.pem \
    -out certs/fullchain.pem \
    -subj "/CN=${DOMAIN}"

# Create directory structure for Let's Encrypt
mkdir -p "certs/live/${DOMAIN}"
cp certs/privkey.pem "certs/live/${DOMAIN}/"
cp certs/fullchain.pem "certs/live/${DOMAIN}/"

# Step 2: Start nginx for ACME challenge
echo "[2/4] Starting nginx for certificate challenge..."
docker-compose -f docker-compose.registry.yml up -d nginx

sleep 5

# Step 3: Get real certificate from Let's Encrypt
echo "[3/4] Obtaining Let's Encrypt certificate..."
docker run --rm \
    -v "${SCRIPT_DIR}/certs:/etc/letsencrypt" \
    -v "${SCRIPT_DIR}/certbot-webroot:/var/www/certbot" \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

# Step 4: Restart all services with real certificate
echo "[4/4] Restarting services with valid certificate..."
docker-compose -f docker-compose.registry.yml down
docker-compose -f docker-compose.registry.yml up -d

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Registry URL: https://${DOMAIN}"
echo ""
echo "To login:"
echo "  docker login ${DOMAIN}"
echo ""
echo "To push images:"
echo "  docker tag myimage:latest ${DOMAIN}/myimage:latest"
echo "  docker push ${DOMAIN}/myimage:latest"
echo ""
echo "For Kubernetes, create image pull secret:"
echo "  kubectl create secret docker-registry regcred \\"
echo "    --docker-server=${DOMAIN} \\"
echo "    --docker-username=<username> \\"
echo "    --docker-password=<password>"
