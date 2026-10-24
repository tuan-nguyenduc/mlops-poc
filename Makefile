# MLOps POC - run `make help`
SHELL       := /bin/bash
CLUSTER     ?= mlops-poc
TAG         ?= 0.1.0
MLFLOW_TAG  := 2.17.2
PY          ?= python3.12
VENV        := .venv
BIN         := $(VENV)/bin
export MLFLOW_TRACKING_URI ?= http://localhost:5050
PLACEHOLDER := https://github.com/YOUR_USER/mlops-poc

.PHONY: help prereqs configure venv up images bootstrap train promote release load \
        chaos-on chaos-off sync watch status urls passwords test lint scan down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

prereqs: ## Check the tools you need
	@for t in docker kind kubectl helm terraform $(PY) git; do \
	  command -v $$t >/dev/null && echo "  ok   $$t" || echo "  MISSING $$t"; done
	@docker info >/dev/null 2>&1 && echo "  ok   docker daemon running" || echo "  MISSING docker daemon (start Docker Desktop / OrbStack)"
	@command -v kubectl-argo-rollouts >/dev/null && echo "  ok   kubectl argo rollouts plugin" || echo "  optional: brew install argoproj/tap/kubectl-argo-rollouts"

configure: ## Point everything at your repo: make configure REPO=https://github.com/<you>/mlops-poc
	@test -n "$(REPO)" || (echo "usage: make configure REPO=https://github.com/<you>/mlops-poc"; exit 1)
	@grep -rl "$(PLACEHOLDER)" gitops infra | xargs sed -i '' "s#$(PLACEHOLDER)#$(REPO)#g"
	@echo 'gitops_repo_url = "$(REPO)"' > infra/terraform/terraform.tfvars
	@echo "Configured for $(REPO). Commit and push before 'make up'."

venv: $(BIN)/activate ## Create the Python virtualenv
$(BIN)/activate: ml/requirements.txt serving/requirements-dev.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r ml/requirements.txt -r serving/requirements-dev.txt
	@touch $@

up: ## Create kind cluster + ArgoCD + root app with Terraform
	cd infra/terraform && terraform init -upgrade && terraform apply -auto-approve
	kind export kubeconfig --name $(CLUSTER)

images: ## Build images and load them into kind (no registry needed)
	docker build -t mlops-poc/model-api:$(TAG) serving
	docker build -t mlops-poc/mlflow:$(MLFLOW_TAG) platform/mlflow-image
	kind load docker-image mlops-poc/model-api:$(TAG) mlops-poc/mlflow:$(MLFLOW_TAG) --name $(CLUSTER)

bootstrap: up images venv ## Everything in one go: cluster, images, venv
	@echo "Waiting for MLflow (ArgoCD is syncing the apps; first time takes a few minutes)..."
	@until curl -sf $(MLFLOW_TRACKING_URI)/health >/dev/null; do sleep 10; echo "  ...still waiting"; done
	@echo "MLflow is up. Next: make train && make promote && make release"

train: venv ## Run the training pipeline (pass options with ARGS="--n-estimators 300")
	cd ml && ../$(BIN)/python pipeline.py $(ARGS)

promote: venv ## Champion/challenger gate; updates gitops values if the candidate wins
	cd ml && ../$(BIN)/python promote.py $(ARGS)

release: ## Commit + push gitops changes (ArgoCD deploys them): make release MSG="..."
	git add gitops/model-api/values.yaml
	git commit -m "$${MSG:-release model-api}"
	git push

load: venv ## Send steady traffic to the model API
	$(BIN)/python scripts/load.py $(ARGS)

chaos-on: ## Make the next release fail 30% of requests (to demo auto-rollback)
	sed -i '' 's/^  errorRate: .*/  errorRate: "0.3"/' gitops/model-api/values.yaml
	@echo "Now: make release MSG='bad release (chaos)' and watch the canary abort"

chaos-off: ## Remove the injected failures
	sed -i '' 's/^  errorRate: .*/  errorRate: "0"/' gitops/model-api/values.yaml

sync: ## Ask ArgoCD to pull Git now instead of waiting for its 3-minute poll
	@for a in root model-api; do kubectl -n argocd annotate application $$a argocd.argoproj.io/refresh=normal --overwrite >/dev/null; done
	@echo "ArgoCD refresh requested"

watch: ## Watch the canary rollout live (needs the kubectl plugin)
	kubectl argo rollouts get rollout model-api -n model-api --watch

status: ## Quick health overview
	@kubectl get applications -n argocd
	@kubectl get pods -A | grep -E 'NAMESPACE|model-api|mlflow|argo|monitoring|ai-gateway' || true

urls: ## Print local URLs
	@echo "ArgoCD      http://localhost:8080   (admin / make passwords)"
	@echo "MLflow      http://localhost:5050"
	@echo "Model API   http://localhost:8800/docs"
	@echo "Grafana     http://localhost:3000   (admin / admin)"
	@echo "Prometheus  http://localhost:9090"

passwords: ## Show the ArgoCD admin password
	@kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d; echo

test: venv ## Unit tests
	$(BIN)/pytest -q ml/tests
	cd serving && ../$(BIN)/pytest -q

lint: venv ## Ruff + terraform fmt + helm lint
	$(BIN)/ruff check .
	cd infra/terraform && terraform fmt -check -recursive
	helm lint gitops/model-api

scan: ## Security scans (brew install trivy checkov)
	trivy image --severity HIGH,CRITICAL --ignore-unfixed mlops-poc/model-api:$(TAG)
	checkov -d infra/terraform --quiet --soft-fail
	checkov -d gitops --framework kubernetes --quiet --soft-fail

down: ## Destroy the cluster
	cd infra/terraform && terraform destroy -auto-approve
