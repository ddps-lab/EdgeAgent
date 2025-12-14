#!/bin/bash
# Setup HTTPS for existing Docker Registry using Let's Encrypt
#
# Prerequisites:
#   - Domain (srv2.ddps.cloud) pointing to this server
#   - Port 80 open for ACME challenge
#   - Existing registry running on port 5000
#
# Usage:
#   ./setup-https-registry.sh srv2.ddps.cloud your@email.com

set -e

DOMAIN="${1:-srv2.ddps.cloud}"
EMAIL="${2:-admin@ddps.cloud}"

echo "========================================"
echo "HTTPS Registry Setup with Let's Encrypt"
echo "========================================"
echo "Domain: ${DOMAIN}"
echo "Email:  ${EMAIL}"
echo "========================================"

# Step 1: Install certbot if not present
echo ""
echo "[1/5] Checking certbot..."
if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot
fi

# Step 2: Stop nginx if running (to free port 80)
echo ""
echo "[2/5] Preparing for certificate..."
sudo systemctl stop nginx 2>/dev/null || true

# Step 3: Get certificate using standalone mode
echo ""
echo "[3/5] Obtaining Let's Encrypt certificate..."
sudo certbot certonly \
    --standalone \
    --preferred-challenges http \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

# Step 4: Create registry TLS config
echo ""
echo "[4/5] Configuring registry for TLS..."

CERT_PATH="/etc/letsencrypt/live/${DOMAIN}"
REGISTRY_CONFIG="/home/container-registry/config.yml"

# Backup existing config
sudo cp "${REGISTRY_CONFIG}" "${REGISTRY_CONFIG}.bak"

# Update registry config with TLS
sudo tee "${REGISTRY_CONFIG}" > /dev/null << EOF
version: 0.1
log:
  fields:
    service: registry
storage:
  cache:
    blobdescriptor: inmemory
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true
http:
  addr: :5000
  tls:
    certificate: /certs/fullchain.pem
    key: /certs/privkey.pem
  headers:
    X-Content-Type-Options: [nosniff]
EOF

# Step 5: Restart registry with TLS
echo ""
echo "[5/5] Restarting registry with HTTPS..."

# Stop existing registry
docker stop registry 2>/dev/null || true
docker rm registry 2>/dev/null || true

# Start registry with TLS certificates mounted
docker run -d \
    --name registry \
    --restart=always \
    -p 443:5000 \
    -v /home/container-registry:/var/lib/registry \
    -v "${REGISTRY_CONFIG}":/etc/docker/registry/config.yml \
    -v "${CERT_PATH}/fullchain.pem:/certs/fullchain.pem:ro" \
    -v "${CERT_PATH}/privkey.pem:/certs/privkey.pem:ro" \
    registry:2

# Also keep HTTP registry for local access (optional)
echo ""
read -p "Keep HTTP registry on port 5000 for local access? (y/N): " keep_http
if [[ "${keep_http}" =~ ^[Yy]$ ]]; then
    docker run -d \
        --name registry-http \
        --restart=always \
        -p 5000:5000 \
        -v /home/container-registry:/var/lib/registry \
        registry:2
    echo "HTTP registry available at localhost:5000"
fi

# Setup auto-renewal
echo ""
echo "[Bonus] Setting up certificate auto-renewal..."
sudo tee /etc/cron.d/certbot-registry > /dev/null << 'EOF'
0 0 1 * * root certbot renew --quiet --post-hook "docker restart registry"
EOF

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "HTTPS Registry: https://${DOMAIN}"
echo ""
echo "Test connection:"
echo "  curl -v https://${DOMAIN}/v2/"
echo ""
echo "Login:"
echo "  docker login ${DOMAIN}"
echo ""
echo "Push images:"
echo "  docker tag myimage:latest ${DOMAIN}/myimage:latest"
echo "  docker push ${DOMAIN}/myimage:latest"
echo ""
echo "Update push scripts:"
echo "  ./scripts/push-to-registry.sh ${DOMAIN} edgeagent"
