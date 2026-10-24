variable "cluster_name" {
  description = "Name of the local kind cluster"
  type        = string
  default     = "mlops-poc"
}

variable "gitops_repo_url" {
  description = "Public Git repo that ArgoCD watches (your fork of this project)"
  type        = string
}

variable "gitops_revision" {
  description = "Branch ArgoCD tracks"
  type        = string
  default     = "main"
}

variable "argocd_chart_version" {
  description = "argo-cd Helm chart version"
  type        = string
  default     = "7.7.0"
}

variable "argocd_apps_chart_version" {
  description = "argocd-apps Helm chart version"
  type        = string
  default     = "2.0.2"
}

# NodePort on the cluster -> port on your Mac
variable "port_mappings" {
  description = "kind NodePort to host port mappings"
  type        = map(number)
  default = {
    "30080" = 8080 # ArgoCD UI
    "30300" = 3000 # Grafana
    "30500" = 5050 # MLflow (5000 is taken by macOS AirPlay Receiver)
    "30800" = 8800 # Model API
    "30900" = 9090 # Prometheus
    "30400" = 4000 # LiteLLM gateway (optional)
  }
}
