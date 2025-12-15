#!/usr/bin/env bash
set -e

source .venv/bin/activate
pm2 stop bitmind-generative-miner || true
pm2 delete bitmind-generative-miner || true
pm2 start gen_miner.config.js
gascli generator logs --follow