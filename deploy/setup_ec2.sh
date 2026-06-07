#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Provisioning & Deployment Script for npz_generator 🕺
# Supported OS : Ubuntu 20.04 / 22.04 / 24.04 LTS
# Supported Arch: ARM64 (m7g / Graviton) and x86_64
#
# NOTE: The React/Three.js frontend lives in a separate repository
#       (npz_gen_front) and is deployed independently.  This script
#       only sets up the Python backend (FastAPI + WebSocket server)
#       and the Nginx reverse proxy that co-exists alongside the
#       existing "livestock" site on the same machine.
#
# Run as: sudo bash deploy/setup_ec2.sh
# ==============================================================================

set -euo pipefail          # Exit on error, unbound variable, or pipe failure

# ── Text Styling ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'               # Reset
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================================"
echo -e "🕺  NPZ GENERATOR — REAL-TIME AVATAR STREAMING — AWS EC2 BOOTSTRAP 🚀"
echo -e "======================================================================${NC}\n"

# ── 0. Root & OS Validation ───────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}${BOLD}Error:${NC} This script must be run as root."
    echo -e "  ${CYAN}sudo bash deploy/setup_ec2.sh${NC}"
    exit 1
fi

# Detect actual non-root user that ran sudo
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(eval echo ~"$ACTUAL_USER")
if [ "$ACTUAL_USER" = "root" ]; then
    ACTUAL_USER="ubuntu"
    ACTUAL_HOME="/home/ubuntu"
fi

# Detect CPU architecture (m7g.large = aarch64)
ARCH=$(uname -m)
echo -e "${CYAN}Running as root — target user: ${BOLD}$ACTUAL_USER${NC} (home: $ACTUAL_HOME)"
echo -e "${CYAN}CPU architecture detected: ${BOLD}$ARCH${NC}"

# ── 1. Collect Environment Inputs ─────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📝  Server Configuration Inputs ---${NC}"

# Domain Name for THIS service (separate from the livestock site)
read -rp "Enter the subdomain for the NPZ avatar service (e.g. avatar.universeumr.com): " DOMAIN_NAME
if [ -z "$DOMAIN_NAME" ]; then
    DOMAIN_NAME="localhost"
    echo -e "${YELLOW}No domain provided — Nginx will bind to the local IP.${NC}"
fi

# Gemini API Key — stored ONLY in the server env file, never exported globally
read -rsp "Enter your Google Gemini API Key (input hidden): " GEMINI_KEY
echo ""
if [ -z "$GEMINI_KEY" ]; then
    echo -e "${YELLOW}Warning: No Gemini API Key provided. Add it manually to server/.env.local later.${NC}"
fi

# ── 2. System Packages ────────────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📦  Step 1: Installing System Dependencies ---${NC}"
apt-get update -y

# libasound2-dev package name changed in Ubuntu 24.04
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "22.04")
LIBASOUND_PKG="libasound2 libasound2-dev"
if dpkg --compare-versions "$UBUNTU_VERSION" ge "24.04"; then
    LIBASOUND_PKG="libasound2t64"
fi

apt-get install -y \
    git \
    curl \
    nginx \
    ffmpeg \
    $LIBASOUND_PKG \
    python3-pip \
    python3-venv \
    python3-dev \
    certbot \
    python3-certbot-nginx \
    build-essential

echo -e "${GREEN}✓ Core system packages installed.${NC}"

# ── 3. Production Directory ────────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📂  Step 2: Preparing Production Directory ---${NC}"
PROD_DIR="/var/www/npz-generator"

if [ -d "$PROD_DIR" ]; then
    BACKUP_DIR="${PROD_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${CYAN}$PROD_DIR already exists — backing up to $BACKUP_DIR${NC}"
    mv "$PROD_DIR" "$BACKUP_DIR"
fi

mkdir -p "$PROD_DIR"

# Copy the backend repository (this repo) — frontend is NOT included
echo -e "${CYAN}Copying backend codebase → $PROD_DIR${NC}"
cp -R . "$PROD_DIR"
chown -R "$ACTUAL_USER":"$ACTUAL_USER" "$PROD_DIR"

echo -e "${GREEN}✓ Backend deployed to $PROD_DIR.${NC}"

# ── 4. Python Virtual Environment & Dependencies ──────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🐍  Step 3: Setting Up Python Virtual Environment ---${NC}"
cd "$PROD_DIR"

sudo -u "$ACTUAL_USER" python3 -m venv venv
VENV_PYTHON="$PROD_DIR/venv/bin/python"

sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install --upgrade pip

# ── PyTorch wheel selection ────────────────────────────────────────────────────
# m7g.large uses ARM64 (aarch64/Graviton3) — no CUDA, CPU-only PyTorch.
# x86_64 instances: install CUDA build only when nvidia-smi is present.
GPU_AVAILABLE=false
if [ "$ARCH" = "x86_64" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
    fi
fi

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}${BOLD}NVIDIA GPU detected — installing CUDA-optimised PyTorch (cu121)...${NC}"
    sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
elif [ "$ARCH" = "aarch64" ]; then
    echo -e "${YELLOW}${BOLD}ARM64 / Graviton detected (m7g) — installing CPU PyTorch for aarch64...${NC}"
    # PyTorch ships aarch64 wheels on PyPI; do NOT use the /whl/cpu index-url
    # (that index only contains x86_64 wheels and will fail on ARM).
    sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install torch torchvision torchaudio
else
    echo -e "${YELLOW}${BOLD}x86_64 CPU-only — installing standard CPU PyTorch...${NC}"
    sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${CYAN}Installing remaining backend dependencies from requirements.txt...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install -r requirements.txt

echo -e "${GREEN}✓ Python environment ready.${NC}"

# ── 5. Environment Configuration & Secure API Key Storage ────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🔐  Step 4: Injecting Environment Configurations ---${NC}"
ENV_FILE="$PROD_DIR/server/.env.local"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$PROD_DIR/server/.env.local.example" ]; then
        cp "$PROD_DIR/server/.env.local.example" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
fi

# ── GEMINI_API_KEY: written exclusively to the env file — NEVER exported ──────
# This is the safest approach: the key is only readable by the service user
# and never leaks into shell history, /proc/environ, or nginx logs.
if [ -n "$GEMINI_KEY" ]; then
    sed -i '/GEMINI_API_KEY/d' "$ENV_FILE"
    sed -i '/GOOGLE_API_KEY/d'  "$ENV_FILE"
    # Write without surrounding quotes so python-dotenv / os.environ picks it up cleanly
    echo "GEMINI_API_KEY=$GEMINI_KEY"  >> "$ENV_FILE"
    echo "GOOGLE_API_KEY=$GEMINI_KEY"  >> "$ENV_FILE"
    echo -e "${GREEN}✓ GEMINI_API_KEY saved securely to $ENV_FILE (not exported to shell).${NC}"
fi

# Ensure production FPS is set
sed -i '/^STREAM_FPS/d'    "$ENV_FILE"
echo "STREAM_FPS=30"       >> "$ENV_FILE"

# VITE_WS_HOST is consumed by the frontend build (separate repo) — not needed here.
# Remove any stale VITE_WS_HOST line to keep the file clean.
sed -i '/^VITE_WS_HOST/d'  "$ENV_FILE"

# Restrict file permissions: only owner can read (protects the API key)
chown "$ACTUAL_USER":"$ACTUAL_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo -e "${GREEN}✓ Environment file configured and locked (chmod 600).${NC}"

# ── 6. Export Mesh Faces (no frontend build — that lives in npz_gen_front) ───
echo -e "\n${YELLOW}${BOLD}--- 🎭  Step 5: Exporting SMPL-X Mesh Faces ---${NC}"
echo -e "${CYAN}Running export_faces.py to generate the avatar index file...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PYTHON" scripts/export_faces.py

echo -e "${GREEN}✓ Mesh faces exported.${NC}"
echo -e "${YELLOW}ℹ  Frontend (npz_gen_front) must be built & deployed separately.${NC}"

# ── 7. Systemd Daemon Services ────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- ⚙️   Step 6: Creating Systemd Background Services ---${NC}"

cp "$PROD_DIR/deploy/backend.service" /etc/systemd/system/npz-backend.service
cp "$PROD_DIR/deploy/admin.service"   /etc/systemd/system/npz-admin.service

# Patch username if different from the default 'ubuntu'
if [ "$ACTUAL_USER" != "ubuntu" ]; then
    sed -i "s/User=ubuntu/User=$ACTUAL_USER/g"   /etc/systemd/system/npz-backend.service
    sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-backend.service
    sed -i "s/User=ubuntu/User=$ACTUAL_USER/g"   /etc/systemd/system/npz-admin.service
    sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-admin.service
fi

touch /var/log/npz-backend.log /var/log/npz-admin.log
chown "$ACTUAL_USER":"$ACTUAL_USER" /var/log/npz-backend.log /var/log/npz-admin.log

systemctl daemon-reload
systemctl enable npz-backend npz-admin
systemctl start  npz-backend npz-admin

echo -e "${GREEN}✓ npz-backend and npz-admin services registered and started.${NC}"

# ── 8. Nginx Configuration (coexist with the livestock site) ─────────────────
echo -e "\n${YELLOW}${BOLD}--- 🌐  Step 7: Configuring Nginx Reverse Proxy ---${NC}"
echo -e "${CYAN}The existing 'livestock' Nginx site will NOT be touched.${NC}"
echo -e "${CYAN}A new 'npz-generator' config block will be added alongside it.${NC}"

NGINX_CONF="/etc/nginx/sites-available/npz-generator"

cp "$PROD_DIR/deploy/nginx.conf" "$NGINX_CONF"

# Substitute the placeholder domain
sed -i "s/YOUR_DOMAIN_NAME_OR_IP/$DOMAIN_NAME/g" "$NGINX_CONF"
sed -i "s/YOUR_DOMAIN_NAME/$DOMAIN_NAME/g"        "$NGINX_CONF"

# Enable the new site (do NOT remove the livestock symlink or the default symlink used by livestock)
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/npz-generator

# Validate and reload — do NOT restart (avoids dropping active WebSocket sessions on livestock)
nginx -t
systemctl reload nginx

echo -e "${GREEN}✓ Nginx updated — livestock site unaffected.${NC}"

# ── 9. Instance Sizing Note ───────────────────────────────────────────────────
echo -e "\n${BLUE}${BOLD}--- 💡  AWS m7g.large Instance Sizing Assessment ---${NC}"
cat <<'EOF'
  Instance : m7g.large (Graviton3, ARM64)
  vCPU     : 2
  RAM      : 8 GB

  For the NPZ backend alone this is acceptable for low-to-medium traffic:
    ✅  Python FastAPI/uvicorn WebSocket server  — minimal CPU overhead
    ✅  PyTorch CPU inference (small batches)    — fits in 8 GB RAM
    ⚠️   High-concurrency sessions (8+)           — may saturate 2 vCPUs
    ❌  Real-time GPU inference / large models   — no GPU on m7g

  Recommendation:
    • Start with m7g.large; monitor CPU/RAM with 'htop' and CloudWatch.
    • If CPU stays above 70% under normal load → upgrade to m7g.xlarge (4 vCPU / 16 GB).
    • For GPU-accelerated inference → switch to a g4dn / g5 instance.
EOF

# ── 10. Summary ───────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}======================================================================"
echo -e "🎉  BACKEND DEPLOYED SUCCESSFULLY!"
echo -e "======================================================================${NC}"
echo -e "Backend WebSocket API is listening on ${BOLD}127.0.0.1:8000${NC} (proxied via Nginx)"
echo -e "Admin / RAG service is listening on   ${BOLD}127.0.0.1:8001${NC} (localhost only)"
echo -e ""
echo -e "${YELLOW}${BOLD}🔐  NEXT STEPS${NC}"
echo -e ""
echo -e "  1. Point DNS: ${CYAN}$DOMAIN_NAME → this server's public IP${NC}"
echo -e ""
echo -e "  2. Issue TLS certificate (required for WSS + mic access in browsers):"
echo -e "     ${CYAN}sudo certbot --nginx -d $DOMAIN_NAME${NC}"
echo -e ""
echo -e "  3. Deploy the frontend separately from the npz_gen_front repo:"
echo -e "     • Set VITE_WS_HOST=wss://$DOMAIN_NAME in its .env"
echo -e "     • Build: npm run build"
echo -e "     • Serve the dist/ folder via a CDN, S3 static site, or a"
echo -e "       separate nginx server block on this or another host."
echo -e ""
echo -e "${BLUE}${BOLD}📋  Handy Management Commands:${NC}"
echo -e "  View backend logs : ${CYAN}tail -f /var/log/npz-backend.log${NC}"
echo -e "  View admin logs   : ${CYAN}tail -f /var/log/npz-admin.log${NC}"
echo -e "  Restart backend   : ${CYAN}sudo systemctl restart npz-backend${NC}"
echo -e "  Restart admin     : ${CYAN}sudo systemctl restart npz-admin${NC}"
echo -e "  Nginx status      : ${CYAN}sudo nginx -t && sudo systemctl reload nginx${NC}"
echo -e "  Livestock site    : ${CYAN}sudo cat /etc/nginx/sites-available/livestock${NC}  (untouched)"
echo -e "======================================================================"
