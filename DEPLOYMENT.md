# Deploying LoanSight to DigitalOcean Kubernetes

This repo deploys itself via GitHub Actions (`.github/workflows/deploy.yml`).
Every push to `main` that touches app/model/deployment code builds a Docker
image, pushes it to DigitalOcean Container Registry (DOCR), and rolls it out
to a DigitalOcean Kubernetes (DOKS) cluster — creating the registry and
cluster on the first run if they don't exist yet.

## One-time setup

### 1. Add two repository secrets

GitHub repo → **Settings → Secrets and variables → Actions → Secrets**:

| Secret | Value |
|---|---|
| `DIGITALOCEAN_ACCESS_TOKEN` | A DO API token with **read/write** scope. Generate one at https://cloud.digitalocean.com/account/api/tokens |
| `ANTHROPIC_API_KEY` | Your Anthropic API key, from https://console.anthropic.com/settings/keys — used by `rag/generator.py` for the compliance explanations |

### 2. (Optional) Override infra defaults

Same screen, **Variables** tab, only if you want something other than the
defaults baked into the workflow:

| Variable | Default | Notes |
|---|---|---|
| `DO_REGISTRY_NAME` | `loansight` | DOCR only allows one registry per DO account — if you already have one, set this to its existing name instead of creating a second. |
| `DO_CLUSTER_NAME` | `loansight` | |
| `DO_REGION` | `nyc1` | Any DOKS region slug (`doctl kubernetes options regions`). |
| `DO_NODE_SIZE` | `s-2vcpu-4gb` | The app holds ~300k training rows in memory per pod plus ChromaDB/onnxruntime; `s-1vcpu-2gb` will likely OOM. |
| `DO_NODE_COUNT` | `1` | Single node keeps cost down for a portfolio project; bump for HA. |

### 3. Push to `main`

The workflow runs automatically. Or trigger it manually from the
**Actions** tab → "Build and deploy to DigitalOcean Kubernetes" → **Run workflow**.

## What it does, step by step

1. Installs `doctl`, authenticated with `DIGITALOCEAN_ACCESS_TOKEN`.
2. Creates the DOCR registry and DOKS cluster if they don't already exist
   (idempotent — safe to run on every push).
3. Links the registry to the cluster (`doctl kubernetes cluster registry add`)
   so nodes can pull the private image without a manually managed pull secret.
4. Builds the image from the repo's `Dockerfile` and pushes
   `registry.digitalocean.com/<registry>/loansight:<git-sha>` and `:latest`.
5. Applies `k8s/namespace.yaml`, syncs `ANTHROPIC_API_KEY` into a `Secret`
   (`loansight-secrets`), and applies `k8s/deployment.yaml` /
   `k8s/service.yaml` with the freshly built image tag substituted in.
6. Waits for the rollout, then polls the `Service` until DigitalOcean's Load
   Balancer has a public IP and prints it to the workflow summary.

The printed IP is the live URL — `http://<ip>/` serves the frontend,
`http://<ip>/health` is a liveness check, `http://<ip>/predict` is the API.

## Cost

Running continuously, this is roughly:

- DOKS control plane: free
- 1x `s-2vcpu-4gb` node: ~$24/month
- DO Load Balancer (`lb-small`): ~$12/month

~$36/month total. Delete the cluster (`doctl kubernetes cluster delete
loansight`) and registry (`doctl registry delete loansight`) when you're done
demoing it, or scale `DO_NODE_COUNT`/pause the cluster to cut cost.

## First-boot latency

The RAG vector store isn't baked into the image — it's built the first time
the container starts (`docker-entrypoint.sh` runs `rag/ingest.py` if
`rag_db/` is empty), because embedding the regulation PDFs needs to download
a small model from Hugging Face, and that shouldn't be assumed to work in
every build environment. Expect the first pod to take 1–3 extra minutes to
become ready; the `startupProbe` in `k8s/deployment.yaml` is set generously
to account for this. Pod restarts repeat this step since nothing is
persisted — fine for a single-replica demo; add a `PersistentVolumeClaim`
for `/app/rag_db` if you want faster restarts later.

## Troubleshooting

```bash
doctl kubernetes cluster kubeconfig save loansight
kubectl get pods -n loansight
kubectl logs -n loansight deploy/loansight
kubectl get svc loansight -n loansight   # external IP
```
