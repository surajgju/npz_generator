#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Provisioning & Deployment Script for npz_generator 🕺
# Supported OS: Ubuntu 20.04 / 22.04 / 24.04 LTS (x86_64)
# Run as: sudo bash deploy/setup_ec2.sh
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# Text Styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0;37m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================================"
echo -e "🕺 NPZ GENERATOR & REAL-TIME AVATAR STREAMING - AWS EC2 BOOTSTRAP 🚀"
echo -e "======================================================================${NC}\n"

# 1. Root & OS Validation
if [ "$EUID" -ne 0 ]; then
   echo -e "${RED}${BOLD}Error: This script must be run as root.${NC} Please execute with: ${CYAN}sudo bash deploy/setup_ec2.sh${NC}"
   exit 1
fi

# Detect actual non-root user that ran sudo
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)
if [ "$ACTUAL_USER" = "root" ]; then
    ACTUAL_USER="ubuntu"
    ACTUAL_HOME="/home/ubuntu"
fi

echo -e "${CYAN}Running installation as root, targeting host user: ${BOLD}$ACTUAL_USER${NC} (Home: $ACTUAL_HOME)"

# 2. Collect Environment Inputs
echo -e "\n${YELLOW}${BOLD}--- 📝 Server Configuration Inputs ---${NC}"

# Domain Name
read -rp "Enter your public Domain Name (e.g. avatar.example.com, or press enter to use EC2 IP): " DOMAIN_NAME
if [ -z "$DOMAIN_NAME" ]; then
    DOMAIN_NAME="localhost"
    echo -e "${YELLOW}No domain provided. Nginx will default to local IP binding.${NC}"
fi

# Gemini API Key
read -rsp "Enter your Google/Gemini API Key (hidden): " GEMINI_KEY
echo ""
if [ -z "$GEMINI_KEY" ]; then
    echo -e "${RED}Warning: No Gemini API Key provided. You will need to add it manually to server/.env.local later.${NC}"
fi

# 3. System Packages Installation
echo -e "\n${YELLOW}${BOLD}--- 📦 Step 1: Installing System Dependencies ---${NC}"
apt-get update -y
apt-get install -y \
    git \
    curl \
    nginx \
    ffmpeg \
    libasound2 \
    libasound2-dev \
    python3-pip \
    python3-venv \
    python3-dev \
    certbot \
    python3-certbot-nginx \
    build-essential

# Install Node.js (Vite frontend compilation)
if ! command -v node &> /dev/null; then
    echo -e "${CYAN}Installing Node.js (LTS)...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo -e "${GREEN}✓ Core system packages installed successfully.${NC}"

# 4. Repository Setup in /var/www/npz-generator
echo -e "\n${YELLOW}${BOLD}--- 📂 Step 2: Preparing Production Directory ---${NC}"
PROD_DIR="/var/www/npz-generator"

if [ -d "$PROD_DIR" ]; then
    echo -e "${CYAN}Target directory $PROD_DIR already exists. Backing up and updating...${NC}"
    # Backup existing
    mv "$PROD_DIR" "${PROD_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$PROD_DIR"

# Copy repository files from current folder (where script is executed)
echo -e "${CYAN}Copying codebase to production directory: $PROD_DIR${NC}"
cp -R . "$PROD_DIR"
chown -R "$ACTUAL_USER":"$ACTUAL_USER" "$PROD_DIR"

echo -e "${GREEN}✓ Production folder deployed to $PROD_DIR.${NC}"

# 5. Virtual Environment & PyTorch Setup
echo -e "\n${YELLOW}${BOLD}--- 🐍 Step 3: Setting Up Python Virtual Environment ---${NC}"
cd "$PROD_DIR"

# Create venv as non-root user to avoid permission locking
sudo -u "$ACTUAL_USER" python3 -m venv venv
VENV_PYTHON="$PROD_DIR/venv/bin/python"

# Upgrade pip
sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install --upgrade pip

# Check for GPU (NVIDIA) presence
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
    fi
fi

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}${BOLD}NVIDIA GPU Detected! Installing CUDA-optimized PyTorch...${NC}"
    sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${YELLOW}${BOLD}No NVIDIA GPU detected. Installing standard CPU PyTorch...${NC}"
    sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install general Python requirements
echo -e "${CYAN}Installing remaining backend dependencies...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -m pip install -r requirements.txt

echo -e "${GREEN}✓ Python dependencies configured successfully.${NC}"

# 6. Apply Environment Config & API Keys
echo -e "\n${YELLOW}${BOLD}--- 🔧 Step 4: Injecting Environment Configurations ---${NC}"
ENV_FILE="$PROD_DIR/server/.env.local"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$PROD_DIR/server/.env.local.example" ]; then
        cp "$PROD_DIR/server/.env.local.example" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
fi

# Inject Gemini API Key
if [ -n "$GEMINI_KEY" ]; then
    # Delete existing keys if any
    sed -i '/GEMINI_API_KEY/d' "$ENV_FILE"
    sed -i '/GOOGLE_API_KEY/d' "$ENV_FILE"
    echo "GEMINI_API_KEY=\"$GEMINI_KEY\"" >> "$ENV_FILE"
    echo "GOOGLE_API_KEY=\"$GEMINI_KEY\"" >> "$ENV_FILE"
fi

# Configure local production-ready variables
sed -i '/STREAM_FPS/d' "$ENV_FILE"
echo "STREAM_FPS=30" >> "$ENV_FILE"

sed -i '/VITE_WS_HOST/d' "$ENV_FILE"
# Map host to empty string so frontend dynamically infers it from browser URL
echo "VITE_WS_HOST=\"\"" >> "$ENV_FILE"

chown "$ACTUAL_USER":"$ACTUAL_USER" "$ENV_FILE"
echo -e "${GREEN}✓ Production environment parameters injected.${NC}"

# 7. Compile Static Frontend (Vite) & Export 3D Mesh faces
echo -e "\n${YELLOW}${BOLD}--- 🎭 Step 5: Exporting SMPL-X Mesh Faces & Building Frontend ---${NC}"

# Export Faces
echo -e "${CYAN}Running export_faces.py to create 3D avatar index file...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PYTHON" scripts/export_faces.py

# Install Node modules & build Vite
echo -e "${CYAN}Compiling React + Three.js static bundle...${NC}"
cd "$PROD_DIR/frontend"
sudo -u "$ACTUAL_USER" npm install
sudo -u "$ACTUAL_USER" npm run build

echo -e "${GREEN}✓ Three.js assets and React bundle generated at frontend/dist/.${NC}"

# 8. Configure Daemon Services (Systemd)
echo -e "\n${YELLOW}${BOLD}--- ⚙️ Step 6: Creating Systemd Background Daemons ---${NC}"

# Copy backend daemon
cp "$PROD_DIR/deploy/backend.service" /etc/systemd/system/npz-backend.service
# Copy admin daemon
cp "$PROD_DIR/deploy/admin.service" /etc/systemd/system/npz-admin.service

# Inject actual user instead of hardcoded 'ubuntu' if username differs
if [ "$ACTUAL_USER" != "ubuntu" ]; then
    sed -i "s/User=ubuntu/User=$ACTUAL_USER/g" /etc/systemd/system/npz-backend.service
    sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-backend.service
    sed -i "s/User=ubuntu/User=$ACTUAL_USER/g" /etc/systemd/system/npz-admin.service
    sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-admin.service
fi

# Ensure log files exist with correct permissions
touch /var/log/npz-backend.log /var/log/npz-admin.log
chown "$ACTUAL_USER":"$ACTUAL_USER" /var/log/npz-backend.log /var/log/npz-admin.log

# Reload systemd and launch
systemctl daemon-reload
systemctl enable npz-backend npz-admin
systemctl start npz-backend npz-admin

echo -e "${GREEN}✓ Core & Admin systemd services registered and started.${NC}"

# 9. Configure Web Proxy (Nginx)
echo -e "\n${YELLOW}${BOLD}--- 🌐 Step 7: Configuring Nginx Reverse Proxy ---${NC}"

NGINX_CONF="/etc/nginx/sites-available/npz-generator"

cp "$PROD_DIR/deploy/nginx.conf" "$NGINX_CONF"

# Inject Domain Name / Host
sed -i "s/YOUR_DOMAIN_NAME_OR_IP/$DOMAIN_NAME/g" "$NGINX_CONF"
sed -i "s/YOUR_DOMAIN_NAME/$DOMAIN_NAME/g" "$NGINX_CONF"

# Link configuration
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/

# Remove default configuration to avoid port conflicts
rm -f /etc/nginx/sites-enabled/default

# Test Nginx syntax and restart
nginx -t
systemctl restart nginx

echo -e "${GREEN}✓ Nginx reverse proxy mapped and restarted.${NC}"

# 10. Summary & SSL/Certbot Guide
echo -e "\n${GREEN}${BOLD}======================================================================"
echo -e "🎉 CONGRATULATIONS! AVATAR PORTAL DEPLOYED SUCCESSFULLY!"
echo -e "======================================================================${NC}"
echo -e "The server is running locally at Ports 8000 (Core API) & 8001 (Admin) and proxied on Port 80."
echo -e "\n${YELLOW}${BOLD}🔐 CRITICAL NEXT STEP: SECURE HTTPS / WSS CERTIFICATE${NC}"
echo -e "Modern browsers require HTTPS to trigger microphone recordings."
echo -e "To configure free, automated SSL certificates for your domain, execute:"
echo -e "\n    ${CYAN}${BOLD}sudo certbot --nginx -d $DOMAIN_NAME${NC}"
echo -e "\nThis will automatically modify Nginx to listen over Port 443 with secure HTTPS/WSS!"
echo -e "\n${BLUE}${BOLD}📋 Handy Management Commands:${NC}"
echo -e "  - View Backend Logs:    ${CYAN}tail -f /var/log/npz-backend.log${NC}"
echo -e "  - View Admin Logs:      ${CYAN}tail -f /var/log/npz-admin.log${NC}"
echo -e "  - Restart Backend:      ${CYAN}sudo systemctl restart npz-backend${NC}"
echo -e "  - Restart Admin:        ${CYAN}sudo systemctl restart npz-admin${NC}"
echo -e "======================================================================"
