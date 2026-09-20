# RetailPulse Production Deployment Guide

This document outlines the deployment, scaling, and monitoring procedures for the **RetailPulse Platform** (Zidio Development Domain Project).

---

## 1. Local Environment Setup

1. **Clone & Setup Directory**:
   ```bash
   git clone <repo-url>
   cd retail_pulse
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Unit Tests**:
   ```bash
   pytest tests/
   ```

4. **Launch Application Locally**:
   * **FastAPI Backend**:
     ```bash
     uvicorn app.api:app --reload --port 8000
     ```
   * **Streamlit Dashboard Frontend**:
     ```bash
     streamlit run app/app.py --server.port 8501
     ```

---

## 2. Docker Container Deployment

Build and run the unified multi-stage Docker container:

```bash
docker build -t zidio/retailpulse:v2.0 .
docker run -p 8501:8501 -p 8000:8000 zidio/retailpulse:v2.0
```

---

## 3. Kubernetes (K8s) Cluster Deployment

Deploy to Kubernetes (AWS EKS or GCP GKE):

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

Check status:
```bash
kubectl get pods -l app=retailpulse
```

---

## 4. Airflow & Monitoring Integration

* **Airflow Retraining**: Copy `airflow/dags/retraining_dag.py` into your Airflow DAGs folder.
* **Prometheus & Grafana**: Import `monitoring/prometheus.yml` and `monitoring/grafana_dashboard.json` for live SLA tracking.
