# Titan runtime secrets

This directory is for Titan-local, mode-0600 runtime credentials only. Secret
files are ignored and must never be packaged, uploaded, copied to the local
workspace, or written to manifests and logs.

P11-T expects the Kimi API key in `moonshot-api-key`. Load it with:

```bash
source titan/scripts/activate_p11_kimi_api.sh
```
