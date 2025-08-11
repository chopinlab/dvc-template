#!/usr/bin/env python3
"""
Prefect deployment script for model evaluation pipeline
"""
import os
from prefect import serve
from model_evaluation_flow import model_evaluation_pipeline

def deploy_flows():
    """Deploy flows to Prefect server"""
    
    # Set Prefect API URL to connect to local server
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    
    print("🚀 Deploying model evaluation pipeline to Prefect...")
    
    # Create deployment
    deployment = model_evaluation_pipeline.to_deployment(
        name="model-evaluation-pipeline",
        version="1.0.0",
        description="Model evaluation pipeline with MLflow tracking",
        tags=["ml", "evaluation", "mlflow"],
        parameters={"dataset_url": None},
        work_pool_name="default"
    )
    
    print("✅ Deployment created successfully!")
    print("📊 Available at: http://localhost:4200")
    
    return deployment

if __name__ == "__main__":
    deploy_flows()