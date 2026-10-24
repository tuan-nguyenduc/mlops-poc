{{- define "model-api.labels" -}}
app.kubernetes.io/name: model-api
app.kubernetes.io/part-of: mlops-poc
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "model-api.selector" -}}
app.kubernetes.io/name: model-api
{{- end }}
