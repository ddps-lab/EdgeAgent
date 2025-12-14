#!/bin/bash
# Setup HTTPS Registry using Let's Encrypt DNS-01 Challenge (Route53)
#
# No port 80 required! Uses Route53 API for domain validation.
#
# Prerequisites:
#   - AWS CLI configured with Route53 access
#   - Domain managed in Route53
#
# Usage:
#   ./setup-https-dns.sh srv2.ddps.cloud mhsong@kookmin.ac.kr

set -e

DOMAIN="${1:-srv2.ddps.cloud}"
EMAIL="${2:-mhsong@kookmin.ac.kr}"

echo "========================================"
echo "HTTPS Registry Setup (DNS-01 Challenge)"
echo "========================================"
echo "Domain: ${DOMAIN}"
echo "Email:  ${EMAIL}"
echo "========================================"
echo ""

# Step 1: Install certbot with Route53 plugin
echo "[1/5] Installing certbot with Route53 plugin..."
if ! command -v certbot &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y certbot
fi

# Install Route53 plugin
sudo apt-get install -y python3-certbot-dns-route53 2>/dev/null || \
    pip3 install certbot-dns-route53

# Step 2: Check AWS credentials
echo ""
echo "[2/5] Checking AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured."
    echo "Run: aws configure"
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: ${AWS_ACCOUNT}"

# Check Route53 hosted zone
HOSTED_ZONE=$(aws route53 list-hosted-zones --query "HostedZones[?contains(Name, '${DOMAIN#*.}')].Id" --output text | head -1)
if [[ -z "${HOSTED_ZONE}" ]]; then
    echo "WARNING: No hosted zone found for ${DOMAIN}"
    echo "Make sure the domain is managed in Route53"
fi

# Step 3: Get certificate using DNS-01 challenge
echo ""
echo "[3/5] Obtaining Let's Encrypt certificate via DNS-01..."
echo "This will automatically create/delete TXT records in Route53"
echo ""

sudo certbot certonly \
    --dns-route53 \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

CERT_PATH="/etc/letsencrypt/live/${DOMAIN}"

# Verify certificate
echo ""
echo "Certificate obtained:"
sudo ls -la "${CERT_PATH}/"

# Step 4: Configure registry with TLS
echo ""
echo "[4/5] Configuring registry for TLS..."

REGISTRY_CONFIG="/home/container-registry/config.yml"

# Backup existing config
if [[ -f "${REGISTRY_CONFIG}" ]]; then
    sudo cp "${REGISTRY_CONFIG}" "${REGISTRY_CONFIG}.bak.$(date +%Y%m%d)"
fi

# Create new config with TLS
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

# Step 5: Restart registry with HTTPS
echo ""
echo "[5/5] Restarting registry with HTTPS..."

# Stop existing containers
docker stop registry registry-web 2>/dev/null || true
docker rm registry registry-web 2>/dev/null || true

# Start HTTPS registry on port 443
docker run -d \
    --name registry \
    --restart=always \
    -p 443:5000 \
    -v /home/container-registry:/var/lib/registry \
    -v "${REGISTRY_CONFIG}":/etc/docker/registry/config.yml:ro \
    -v "${CERT_PATH}/fullchain.pem:/certs/fullchain.pem:ro" \
    -v "${CERT_PATH}/privkey.pem:/certs/privkey.pem:ro" \
    registry:2

echo "Waiting for registry to start..."
sleep 3

# Test HTTPS
echo ""
echo "Testing HTTPS connection..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" "https://${DOMAIN}/v2/" || echo "Note: May need to wait for DNS propagation"

# Setup auto-renewal cron
echo ""
echo "Setting up certificate auto-renewal..."
sudo tee /etc/cron.d/certbot-registry > /dev/null << EOF
# Renew certificate monthly and restart registry
0 0 1 * * root certbot renew --dns-route53 --quiet --post-hook "docker restart registry"
EOF

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "HTTPS Registry: https://${DOMAIN}"
echo ""
echo "Test:"
echo "  curl https://${DOMAIN}/v2/"
echo ""
echo "Login:"
echo "  docker login ${DOMAIN}"
echo ""
echo "Push EdgeAgent images:"
echo "  cd /home/mhsong/edgeagent"
echo "  ./scripts/push-to-registry.sh ${DOMAIN} edgeagent"
