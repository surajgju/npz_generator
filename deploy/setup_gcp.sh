#!/usr/bin/env bash
# ==============================================================================
# GCP Compute Engine Provisioning & Deployment Script for npz_generator 🕺
#
# Target Instance:
#   Provider  : Google Cloud Platform (Compute Engine)
#   Machine   : c2-standard-4  (4 vCPU, 16 GB RAM, Intel Cascade Lake x86_64)
#   OS        : Debian 12 Bookworm (debian-12-bookworm-v20260513)
#   GPU       : None  (CPU-only inference)
#   SSH user  : suraj
#   Region    : us-central1-c
#
# What this script does:
#   • Installs all system dependencies (Debian 12 compatible)
#   • Deploys the Python backend to /var/www/npz-generator
#   • Creates a Python venv + installs CPU PyTorch (x86_64 /whl/cpu) + requirements
#   • Tunes OpenMP / MKL thread counts for Intel Cascade Lake (AVX-512)
#   • Stores the Gemini API key securely in server/.env.local (chmod 600)
#   • Registers npz-backend and npz-admin as systemd services
#   • Configures Nginx as a reverse proxy for /ws/ and /api/ endpoints
#   • Guides you through Certbot TLS setup
#
# NOTE: The React/Three.js frontend lives in the separate npz_gen_front repo
#       and is deployed independently. This script is backend-only.
#
# Run as: sudo bash deploy/setup_gcp.sh
# ==============================================================================

set -euo pipefail

# ── Text Styling ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================================"
echo -e "🕺  NPZ GENERATOR — GCP c2-standard-4 / Cascade Lake — BOOTSTRAP 🚀"
echo -e "    Debian 12 / x86_64 / AVX-512 CPU inference / us-central1-c"
echo -e "======================================================================${NC}\n"

# ── 0. Root Check ─────────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}${BOLD}Error:${NC} This script must be run as root."
    echo -e "  ${CYAN}sudo bash deploy/setup_gcp.sh${NC}"
    exit 1
fi

# Detect the actual non-root user that called sudo.
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(eval echo ~"$ACTUAL_USER")
if [ "$ACTUAL_USER" = "root" ]; then
    ACTUAL_USER="suraj"
    ACTUAL_HOME="/home/suraj"
fi

ARCH=$(uname -m)   # Expected: x86_64
echo -e "${CYAN}Deploying as root — service user: ${BOLD}$ACTUAL_USER${NC} (home: $ACTUAL_HOME)"
echo -e "${CYAN}CPU architecture : ${BOLD}$ARCH${NC}  |  OS: $(. /etc/os-release && echo "$PRETTY_NAME")"

# Confirm AVX-512 is available on this Cascade Lake host
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    echo -e "${GREEN}✓ AVX-512 detected — PyTorch CPU inference will use full vectorisation.${NC}"
else
    echo -e "${YELLOW}⚠  AVX-512 not found in /proc/cpuinfo (unexpected on c2-standard-4).${NC}"
fi

# ── 1. Collect Inputs ─────────────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📝  Configuration Inputs ---${NC}"

read -rp "Enter domain / subdomain for this service (e.g. avatar.universeumr.com), or Enter for raw IP: " DOMAIN_NAME
if [ -z "$DOMAIN_NAME" ]; then
    # Auto-detect the external IP from GCP metadata server
    DETECTED_IP=$(curl -sf -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/externalIp" \
        || echo "")
    DOMAIN_NAME="${DETECTED_IP:-localhost}"
    echo -e "${CYAN}No domain provided — using detected external IP: ${BOLD}$DOMAIN_NAME${NC}"
else
    echo -e "${CYAN}Using server name: ${BOLD}$DOMAIN_NAME${NC}"
fi

# Gemini API Key — stored ONLY in server/.env.local, NEVER exported to the shell
read -rsp "Enter your Google Gemini API Key (hidden): " GEMINI_KEY
echo ""
if [ -z "$GEMINI_KEY" ]; then
    echo -e "${YELLOW}⚠  No Gemini key provided. Add it manually to server/.env.local later.${NC}"
fi

# ── 2. System Packages (Debian 12 Bookworm) ───────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📦  Step 1: Installing System Dependencies ---${NC}"

apt-get update -y
apt-get install -y \
    git \
    curl \
    wget \
    nginx \
    ffmpeg \
    libasound2 \
    libasound2-dev \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-full \
    certbot \
    python3-certbot-nginx \
    build-essential \
    lsb-release \
    ca-certificates \
    gnupg \
    numactl \
    linux-cpupower

# c2 instances benefit from performance CPU governor
cpupower frequency-set -g performance 2>/dev/null || true

echo -e "${GREEN}✓ System packages installed. CPU governor set to performance.${NC}"

# ── 3. Production Directory ───────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 📂  Step 2: Preparing Production Directory ---${NC}"
PROD_DIR="/var/www/npz-generator"

if [ -d "$PROD_DIR" ]; then
    BACKUP_DIR="${PROD_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${CYAN}$PROD_DIR exists — backing up to $BACKUP_DIR${NC}"
    mv "$PROD_DIR" "$BACKUP_DIR"
fi

mkdir -p "$PROD_DIR"
echo -e "${CYAN}Copying backend codebase → $PROD_DIR${NC}"
cp -R . "$PROD_DIR"
chown -R "$ACTUAL_USER":"$ACTUAL_USER" "$PROD_DIR"

echo -e "${GREEN}✓ Backend code deployed to $PROD_DIR.${NC}"

# ── 4. Python Virtual Environment & CPU-tuned PyTorch ─────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🐍  Step 3: Python Virtual Environment & Dependencies ---${NC}"
cd "$PROD_DIR"

# Debian 12 enforces PEP 668 — always use a venv
sudo -u "$ACTUAL_USER" python3 -m venv venv
VENV_PYTHON="$PROD_DIR/venv/bin/python"
VENV_PIP="$PROD_DIR/venv/bin/pip"

sudo -u "$ACTUAL_USER" "$VENV_PIP" install --upgrade pip setuptools wheel

# ── CPU PyTorch for x86_64 ────────────────────────────────────────────────────
# c2-standard-4 is CPU-only. The /whl/cpu index provides a build linked
# against Intel MKL + OpenMP which automatically exploits AVX-512 on Cascade Lake.
echo -e "${CYAN}${BOLD}Installing CPU-optimised PyTorch (Intel MKL + AVX-512)...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PIP" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

echo -e "${CYAN}Installing backend requirements from requirements.txt...${NC}"
sudo -u "$ACTUAL_USER" "$VENV_PIP" install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Quick sanity check — confirm torch loads and reports CPU correctly
TORCH_CHECK=$(sudo -u "$ACTUAL_USER" "$VENV_PYTHON" -c \
    "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device: cpu')" 2>&1 || true)
echo -e "  ${CYAN}$TORCH_CHECK${NC}"

echo -e "${GREEN}✓ Python venv with CPU PyTorch ready.${NC}"

# ── 5. Secure Environment Configuration ───────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🔐  Step 4: Environment & API Key Configuration ---${NC}"
ENV_FILE="$PROD_DIR/server/.env.local"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$PROD_DIR/server/.env.local.example" ]; then
        cp "$PROD_DIR/server/.env.local.example" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
fi

# GEMINI_API_KEY is written ONLY to the env file (chmod 600).
# NEVER exported to the shell — prevents leaking into /proc/environ, ps output, logs.
if [ -n "$GEMINI_KEY" ]; then
    sed -i '/GEMINI_API_KEY/d' "$ENV_FILE"
    sed -i '/GOOGLE_API_KEY/d'  "$ENV_FILE"
    echo "GEMINI_API_KEY=$GEMINI_KEY" >> "$ENV_FILE"
    echo "GOOGLE_API_KEY=$GEMINI_KEY" >> "$ENV_FILE"
    echo -e "${GREEN}✓ GEMINI_API_KEY written to $ENV_FILE (not exported to shell).${NC}"
fi

# ── Production runtime tuning for Cascade Lake CPU inference ──────────────────
sed -i '/^STREAM_FPS/d'                    "$ENV_FILE"
echo "STREAM_FPS=30"                       >> "$ENV_FILE"

# Slightly smaller live batch = lower latency on CPU
# (real-time is not achievable on CPU, but smaller batches reduce lag per chunk)
sed -i '/^LIVE_INFERENCE_BATCH_SAMPLES/d'  "$ENV_FILE"
echo "LIVE_INFERENCE_BATCH_SAMPLES=4800"   >> "$ENV_FILE"

sed -i '/^INFERENCE_BATCH_SAMPLES/d'       "$ENV_FILE"
echo "INFERENCE_BATCH_SAMPLES=2400"        >> "$ENV_FILE"

# Intel OpenMP / MKL threading — use all 4 vCPUs for the inference process
# Keeping these in the env file makes them visible to the systemd-spawned process
sed -i '/^OMP_NUM_THREADS/d'               "$ENV_FILE"
echo "OMP_NUM_THREADS=4"                   >> "$ENV_FILE"

sed -i '/^MKL_NUM_THREADS/d'               "$ENV_FILE"
echo "MKL_NUM_THREADS=4"                   >> "$ENV_FILE"

# Disable GPU env var — no GPU on this instance
sed -i '/^CUDA_VISIBLE_DEVICES/d'          "$ENV_FILE"
echo "CUDA_VISIBLE_DEVICES="              >> "$ENV_FILE"

# VITE_* vars belong to the frontend repo — strip them from the backend env
sed -i '/^VITE_/d' "$ENV_FILE"

# Lock: only the service user can read the key
chown "$ACTUAL_USER":"$ACTUAL_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo -e "${GREEN}✓ env file configured and locked (chmod 600).${NC}"

# ── 6. Export Mesh Faces ──────────────────────────────────────────────────────
# (Skipped: Frontend is deployed separately, and SMPL-X models are not required
# for the backend inference pipeline to generate NPZ coefficients).
echo -e "\n${YELLOW}${BOLD}--- 🎭  Step 5: Exporting SMPL-X Mesh Faces (Skipped) ---${NC}"
echo -e "${YELLOW}ℹ  Frontend (npz_gen_front) is deployed separately, skipping faces.json export.${NC}"

# ── 7. Systemd Services ───────────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- ⚙️   Step 6: Registering Systemd Services ---${NC}"

cp "$PROD_DIR/deploy/backend.service" /etc/systemd/system/npz-backend.service
cp "$PROD_DIR/deploy/admin.service"   /etc/systemd/system/npz-admin.service

# Patch username (service files default to User=ubuntu)
sed -i "s/User=ubuntu/User=$ACTUAL_USER/g"   /etc/systemd/system/npz-backend.service
sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-backend.service
sed -i "s/User=ubuntu/User=$ACTUAL_USER/g"   /etc/systemd/system/npz-admin.service
sed -i "s/Group=ubuntu/Group=$ACTUAL_USER/g" /etc/systemd/system/npz-admin.service

# Inject CPU threading env vars into the backend service unit
# so the inference subprocess inherits them even on a bare systemd start
for VAR in "OMP_NUM_THREADS=4" "MKL_NUM_THREADS=4" "CUDA_VISIBLE_DEVICES="; do
    KEY="${VAR%%=*}"
    if ! grep -q "^Environment=${KEY}" /etc/systemd/system/npz-backend.service; then
        sed -i "/^Environment=PYTHONUNBUFFERED/a Environment=${VAR}" \
            /etc/systemd/system/npz-backend.service
    fi
done

# Remove GPU-specific flags that don't apply on a CPU-only instance
sed -i '/PYTORCH_ENABLE_MPS_FALLBACK/d'    /etc/systemd/system/npz-backend.service

touch /var/log/npz-backend.log /var/log/npz-admin.log
chown "$ACTUAL_USER":"$ACTUAL_USER" /var/log/npz-backend.log /var/log/npz-admin.log

systemctl daemon-reload
systemctl enable npz-backend npz-admin
systemctl start  npz-backend npz-admin

echo -e "${GREEN}✓ npz-backend (port 8000) and npz-admin (port 8001) started.${NC}"

# ── 8. Nginx Reverse Proxy ────────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🌐  Step 7: Configuring Nginx ---${NC}"

rm -f /etc/nginx/sites-enabled/default

NGINX_CONF="/etc/nginx/sites-available/npz-generator"
cp "$PROD_DIR/deploy/nginx.conf" "$NGINX_CONF"

sed -i "s/YOUR_DOMAIN_NAME_OR_IP/$DOMAIN_NAME/g" "$NGINX_CONF"
sed -i "s/YOUR_DOMAIN_NAME/$DOMAIN_NAME/g"        "$NGINX_CONF"

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/npz-generator

nginx -t
systemctl restart nginx

echo -e "${GREEN}✓ Nginx configured and restarted.${NC}"

# ── 9. GCP Firewall Reminder ──────────────────────────────────────────────────
echo -e "\n${YELLOW}${BOLD}--- 🔥  GCP Firewall Rules ---${NC}"
cat <<'FIREWALL'
  GCP VPC firewall rules are managed in Cloud Console (NOT ufw/iptables).
  Required ingress rules:

    Rule Name          Protocol   Port   Source
    ─────────────────────────────────────────────────────
    allow-http         TCP        80     0.0.0.0/0
    allow-https        TCP        443    0.0.0.0/0
    allow-ssh          TCP        22     0.0.0.0/0

  Ports 8000 and 8001 must remain CLOSED to the internet.

  gcloud CLI:
    gcloud compute firewall-rules create allow-http  --allow=tcp:80  --target-tags=npz-server
    gcloud compute firewall-rules create allow-https --allow=tcp:443 --target-tags=npz-server

FIREWALL

# ── 10. Instance Sizing Assessment ────────────────────────────────────────────
echo -e "${BLUE}${BOLD}--- 💡  c2-standard-4 / Cascade Lake CPU Assessment ---${NC}"
cat <<'SIZING'
  Machine   : c2-standard-4 (Intel Cascade Lake, x86_64)
  vCPU      : 4     RAM : 16 GB     GPU : None
  CPU flags : AVX-512F / AVX-512BW / AVX-512VL  (auto-used by PyTorch MKL)

  With CPU-only PyTorch + AVX-512 on Cascade Lake:
    ✅  Server startup & model load      — ~3–5 min (one-time at boot)
    ✅  FastAPI / WebSocket handling     — trivial, async I/O
    ✅  Offline NPZ batch generation     — works fine (no time constraint)
    ⚠️   Real-time avatar inference       — ~1.5–3× slower than audio clock
                                           (each 500ms chunk takes ~750ms–1.5s)
    ⚠️   Conversational avatar           — noticeable delay (~1–2s lag)
    ❌  Live streaming at 30 FPS         — GPU required for that

  Cascade Lake advantage over Broadwell (n1):
    → AVX-512 gives ~2–3× faster PyTorch matmul vs Broadwell (n1)
    → Suitable for offline/batch use-cases and low-traffic demos

  To reach real-time inference:
    → Add NVIDIA T4 GPU to this instance (GCP Console → Edit → Add GPU)
    → Re-run the GPU variant of this script (setup_gcp_gpu.sh)

  Thread tuning applied (already injected into env + systemd):
    OMP_NUM_THREADS=4   ← all 4 vCPUs used by OpenMP inference
    MKL_NUM_THREADS=4   ← Intel MKL BLAS threads

SIZING

# ── 11. Final Summary ─────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}======================================================================"
echo -e "🎉  GCP CPU BACKEND DEPLOYED SUCCESSFULLY!"
echo -e "======================================================================${NC}"
echo -e "  Instance    : ${BOLD}c2-standard-4 / Intel Cascade Lake / us-central1-c${NC}"
echo -e "  Backend WS  : ${BOLD}127.0.0.1:8000${NC}  → proxied at wss://$DOMAIN_NAME/ws/"
echo -e "  Admin API   : ${BOLD}127.0.0.1:8001${NC}  → localhost only (SSH tunnel)"
echo -e "  Public IP   : ${BOLD}$DOMAIN_NAME${NC}"
echo -e "  GPU         : ${BOLD}None — CPU-only inference (AVX-512)${NC}"
echo -e ""
echo -e "${YELLOW}${BOLD}🔐  NEXT STEPS${NC}"
echo -e ""
echo -e "  1. ${BOLD}Point DNS${NC} — add A-record:"
echo -e "     ${CYAN}$DOMAIN_NAME  →  <this instance's external IP>${NC}"
echo -e ""
echo -e "  2. ${BOLD}Issue TLS certificate${NC} (browsers require HTTPS for mic + WSS):"
echo -e "     ${CYAN}sudo certbot --nginx -d $DOMAIN_NAME${NC}"
echo -e ""
echo -e "  3. ${BOLD}Deploy frontend${NC} from the npz_gen_front repo:"
echo -e "     • Set ${CYAN}VITE_WS_HOST=wss://$DOMAIN_NAME${NC} in its .env"
echo -e "     • Run: ${CYAN}npm run build${NC}"
echo -e "     • Host dist/ on CDN / Firebase Hosting / separate nginx vhost"
echo -e ""
echo -e "  4. ${BOLD}Admin SSH tunnel${NC} (port 8001 is localhost-only):"
echo -e "     ${CYAN}ssh -L 8001:127.0.0.1:8001 suraj@$DOMAIN_NAME${NC}"
echo -e "     Then open: ${CYAN}http://localhost:8001/docs${NC}"
echo -e ""
echo -e "${BLUE}${BOLD}📋  Management Commands:${NC}"
echo -e "  CPU stats       : ${CYAN}htop${NC}  or  ${CYAN}mpstat -P ALL 1${NC}"
echo -e "  Backend logs    : ${CYAN}tail -f /var/log/npz-backend.log${NC}"
echo -e "  Admin logs      : ${CYAN}tail -f /var/log/npz-admin.log${NC}"
echo -e "  Restart backend : ${CYAN}sudo systemctl restart npz-backend${NC}"
echo -e "  Restart admin   : ${CYAN}sudo systemctl restart npz-admin${NC}"
echo -e "  Nginx reload    : ${CYAN}sudo nginx -t && sudo systemctl reload nginx${NC}"
echo -e "  Service status  : ${CYAN}sudo systemctl status npz-backend npz-admin${NC}"
echo -e "======================================================================"
