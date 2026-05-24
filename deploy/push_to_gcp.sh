#!/usr/bin/env bash
# ==============================================================================
# Local -> GCP Fast Deployment Script
# 
# This script uses 'tar' over 'ssh' to push your local code to the GCP instance.
# It automatically ignores huge folders like 'venv', '__pycache__', '.git', etc.
# ==============================================================================

set -euo pipefail

# Text Styling
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${CYAN}${BOLD}🚀 Fast Push to GCP (Excluding local dependencies)${NC}\n"

# Define target
GCP_USER="suraj"
GCP_IP="34.132.155.227"
TARGET_DIR="/tmp/npz_generator"

read -rp "Target IP [$GCP_IP]: " input_ip
GCP_IP="${input_ip:-$GCP_IP}"

echo -e "\n${CYAN}Compressing and syncing codebase to ${GCP_USER}@${GCP_IP}:${TARGET_DIR}...${NC}"

# Ensure target directory exists on remote server
ssh "${GCP_USER}@${GCP_IP}" "mkdir -p ${TARGET_DIR}"

# Compress locally (excluding heavy folders) and extract directly on the server
tar --exclude="venv" \
    --exclude=".venv" \
    --exclude="__pycache__" \
    --exclude=".git" \
    --exclude="node_modules" \
    --exclude=".DS_Store" \
    -czf - . | ssh "${GCP_USER}@${GCP_IP}" "tar -xzf - -C ${TARGET_DIR}"

echo -e "\n${GREEN}${BOLD}✓ Code pushed successfully without local dependencies!${NC}"
echo -e "\n${CYAN}Next steps:${NC}"
echo -e "1. SSH into your server:"
echo -e "   ssh ${GCP_USER}@${GCP_IP}"
echo -e "2. Run the setup script on the server:"
echo -e "   cd ${TARGET_DIR}"
echo -e "   sudo bash deploy/setup_gcp.sh"
