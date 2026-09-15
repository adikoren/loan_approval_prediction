# Deploying LoanSight to DigitalOcean App Platform

This repo deploys to **DigitalOcean App Platform**, not Kubernetes. The app
(`starfish-app`) is configured to auto-deploy from the `main` branch — every
push to `main` triggers a new build and rollout, no CI workflow required.

Live URL: `https://starfish-app-eyadx.ondigitalocean.app`

## One-time setup

Create the app once from the DigitalOcean control panel (or `doctl apps
create`) pointing at this GitHub repo, branch `main`, with:

- **Source**: this repo, Dockerfile build (`/Dockerfile`)
- **HTTP port**: `8000`
- **Deploy on push**: enabled

### Environment variable

Set under the app's **Settings → App-Level Environment Variables**:

| Variable | Type | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Encrypted (SECRET)** | Used by `rag/generator.py` for compliance explanations. Must be a **workspace-scoped** key — see `ANTHROPIC_WORKSPACE_ID` below if your Console org issues unscoped keys. |
| `ANTHROPIC_WORKSPACE_ID` | Optional, plaintext | Only needed if the API key isn't already scoped to a workspace (surfaces as a 400 error naming the fix if it's missing and required). |

Never set `ANTHROPIC_API_KEY` as a plaintext/`GENERAL` variable — always use
the encrypted `SECRET` type.

## What happens on push

1. DigitalOcean detects the push to `main` and builds the image from
   `Dockerfile`.
2. `rag_db.tar.gz` (a pre-built ChromaDB vector store, ~1,640 chunks) is
   baked into the image at build time. `docker-entrypoint.sh` only re-runs
   `rag/ingest.py` if `/app/rag_db` is empty at container start — so a
   change to `rag/ingest.py` or `docs/` alone does **not** take effect
   until `rag_db.tar.gz` is regenerated and committed (see below).
3. The container starts, `/health` becomes ready, and App Platform routes
   traffic to the new deployment.

Poll deployment status with `mcp__digitalocean__apps-get-deployment-status`
(or the DigitalOcean dashboard) — it moves through `PENDING_BUILD` →
`BUILDING` → `DEPLOYING` → `ACTIVE`.

## Regenerating the RAG knowledge base

If you change `docs/*` or `rag/ingest.py`, rebuild and re-commit the archive
so the deployed image actually picks it up:

```bash
python rag/ingest.py          # rebuilds ./rag_db/ from docs/
tar -czf rag_db.tar.gz rag_db/
git add rag_db.tar.gz
```

## Troubleshooting

- **Fallback compliance explanation** ("...temporarily unavailable"): check
  the app logs (`mcp__digitalocean__apps-get-logs` or the dashboard) for a
  `[rag.generator]`-prefixed line — it names the exact cause (missing key,
  invalid key, or workspace-scoping error) rather than a generic failure.
- **Retrieval returns nothing for a known-good query**: usually means
  `rag_db.tar.gz` is stale relative to `rag/ingest.py`/`docs/` — see above.
