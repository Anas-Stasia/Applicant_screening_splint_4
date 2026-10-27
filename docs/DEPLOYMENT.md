# Deployment guide (Sprint 4)

One Docker image, three hosting options. Every option ends with the same check:
`serving/scripts/smoke_test.py` against the live URL.

| | Render | Railway | AWS (ECS Express Mode) |
|---|---|---|---|
| Config in repo | `render.yaml` | `railway.json` | `deploy/aws/deploy_ecs_express.sh` |
| Deploy trigger | merge to `main` → CI green → deploy (`autoDeployTrigger: checksPass`) | GitHub integration with **Wait for CI** | run the script (or wire it into CI) |
| Size to pick | `1c-2g` (1 CPU / 2 GB) | ≥ 2 GB memory | `--cpu 1024 --memory 2048` |
| Prediction logs | ephemeral disk → `SCREENING_LOG_PREDICTIONS_STDOUT=true` | same | stdout → CloudWatch Logs |
| Good for | the course demo; least setup | same, usage-based pricing | teams already on AWS; autoscaling 1–3 tasks |

**Memory:** torch + MiniLM + the cohort embeddings need roughly 0.5–1 GB. 512 MB instances
(Render free / `0.5c-512mb`) are likely to be killed for running out of memory, so use 2 GB.

---

## 0. Before any deploy: build and run locally

From the repo root (`applicant_screening_nb/`):

```bash
# only after re-running the Sprint 1 notebooks (new data or model):
python serving/scripts/export_artifacts.py        # refresh serving/artifacts + manifest
git add serving/artifacts && git commit -m "Refresh serving artifacts"

docker compose up --build                          # http://localhost:8000/docs
python serving/scripts/smoke_test.py --url http://localhost:8000 --expect-model all-MiniLM-L6-v2
```

Without Compose:

```bash
docker build -f serving/Dockerfile -t screening-api --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
docker run -p 8000:8000 -v screening-logs:/srv/logs -e SCREENING_API_KEY=change-me screening-api
```

The first build downloads torch (CPU) and the model, so expect a few minutes. Later builds
reuse the cached layers and take seconds if only code changed.

## 1. Render

1. Push the repo to GitHub (the `.github/workflows/ci.yml` workflow must exist on `main`).
2. Render dashboard → **New → Blueprint** → select the repo. Render reads `render.yaml`.
3. When prompted, set `SCREENING_CORS_ORIGINS` to the Frontend squad's URL
   (e.g. `https://screening-ui.onrender.com`). `SCREENING_API_KEY` is generated for you.
   Copy it from **Environment** and share it with the squads through a password manager, not chat.
4. Render waits until `/ready` returns 200 before switching traffic, so deploys have no downtime.
5. Check it:
   ```bash
   python serving/scripts/smoke_test.py --url https://applicant-screening-api.onrender.com \
     --api-key <key> --wait 300 --expect-model all-MiniLM-L6-v2
   ```
6. Optional: add the URL as a GitHub **variable** `PRODUCTION_URL` and the key as a **secret**
   `SCREENING_API_KEY`. The `deploy` job in CI then smoke-tests production after every merge.

## 2. Railway

1. Railway → **New Project → Deploy from GitHub repo**. Railway reads `railway.json` and builds
   `serving/Dockerfile`, health-checking `/ready`.
2. Service → **Settings → Source**: turn on **Wait for CI**, so only commits with a green workflow deploy.
3. **Variables**: `SCREENING_API_KEY`, `SCREENING_CORS_ORIGINS`, `SCREENING_ENVIRONMENT=railway`,
   `SCREENING_LOG_PREDICTIONS_STDOUT=true`. Railway sets `PORT`, which the image already uses.
4. **Settings → Networking → Generate Domain**, then run the smoke test against it.

## 3. AWS — ECS Express Mode

AWS App Runner stopped accepting new customers on 30 April 2026. AWS now recommends
**Amazon ECS Express Mode**, which creates the Fargate service, load balancer, HTTPS URL
and autoscaling from a single command. `deploy/aws/deploy_ecs_express.sh` wraps it:

```bash
export AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012
export EXECUTION_ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole
export INFRA_ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices
./deploy/aws/deploy_ecs_express.sh          # creates ECR repo, builds, pushes, creates the service
# later deploys:
export SERVICE_ARN=<ARN printed by the first run>
./deploy/aws/deploy_ecs_express.sh          # pushes a new image tag and updates the service
```

* The two IAM roles are one-time setup. The ECS console's Express Mode "create" screen can
  create them for you, or see the AWS guide *Create your first Amazon ECS Express Mode service*.
* The script builds for `linux/amd64`, so it works from an Apple-silicon laptop.
* Put the API key in AWS Secrets Manager and pass it through the container's `secrets` field. Don't
  use a plain environment variable for it.
* Logs go to CloudWatch Logs. Because `SCREENING_LOG_PREDICTIONS_STDOUT=true`, the prediction
  records land there too and can be queried with Logs Insights.

## 4. After every deploy

| Check | How |
|---|---|
| It's the build you expect | `GET /v1/model` → `build` = the git commit, `environment` = the platform |
| Model and artifacts loaded | `job_embeddings_source` / `cohort_embeddings_source` start with `loaded` (not `encoded at startup`) |
| End-to-end | `smoke_test.py` exits 0 |
| Monitoring is live | `GET /v1/monitoring/predictions` shows the smoke test's requests. `/metrics` responds |

## 5. Rolling back

* **Render / Railway:** redeploy the previous successful deploy from the dashboard (one click).
* **AWS:** `TAG=<previous sha> ./deploy/aws/deploy_ecs_express.sh` with `SERVICE_ARN` set, or roll back the deployment in the console.
* **Any platform:** every `main` build is in GHCR as `ghcr.io/<owner>/applicant-screening-api:<sha>`.

## Environment variables in production

| Variable | Set it to |
|---|---|
| `SCREENING_API_KEY` | a long random value (required outside local dev) |
| `SCREENING_CORS_ORIGINS` | the Frontend squad's deployed origin(s), comma-separated |
| `SCREENING_ENVIRONMENT` | `render` / `railway` / `aws` (shows up in logs and metrics) |
| `SCREENING_LOG_PREDICTIONS_STDOUT` | `true` on any host with an ephemeral disk |
| `SCREENING_SIMILARITY_FLOOR` | leave at `0.35` until it's tuned on labelled "no match" data |

Full list: [`serving/README.md` → Configuration](../serving/README.md#configuration).
