output "urls" {
  description = "Local URLs"
  value = {
    argocd     = "http://localhost:8080"
    grafana    = "http://localhost:3000"
    mlflow     = "http://localhost:5050"
    model_api  = "http://localhost:8800/docs"
    prometheus = "http://localhost:9090"
  }
}

output "kubeconfig_hint" {
  value = "kind export kubeconfig --name ${var.cluster_name}"
}
