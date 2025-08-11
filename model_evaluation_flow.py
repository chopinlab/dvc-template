import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from prefect import flow, task
import requests
from datetime import datetime

# MLflow tracking configuration  
artifact_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(artifact_dir, exist_ok=True)
MLFLOW_TRACKING_URI = f"file://{artifact_dir}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@task
def load_external_dataset(url: str = None) -> pd.DataFrame:
    """Load dataset from external source or generate synthetic data"""
    if url:
        try:
            # Load from external URL
            df = pd.read_csv(url)
            print(f"Loaded dataset from {url} with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to load from URL: {e}")
            print("Falling back to synthetic data...")
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Generated synthetic dataset with shape {df.shape}")
    return df

@task
def prepare_data(df: pd.DataFrame):
    """Prepare data for evaluation"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data for evaluation (we'll use a pre-trained model concept)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

@task
def load_or_create_model(X_train, y_train):
    """Load existing model or create a simple one for evaluation"""
    # For demo purposes, create a simple model
    # In real scenario, you'd load a pre-trained model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Created and trained RandomForest model")
    return model

@task
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"Model Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics

@task
def log_to_mlflow(model, metrics: dict, dataset_info: dict):
    """Log model and metrics to MLflow"""
    
    # Create or get experiment
    experiment_name = "model_evaluation_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("dataset_shape", dataset_info["shape"])
        mlflow.log_param("evaluation_date", datetime.now().isoformat())
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Get current run info
        run = mlflow.active_run()
        run_id = run.info.run_id
        
        print(f"Logged evaluation results to MLflow (Run ID: {run_id})")
        print(f"View results at: {MLFLOW_TRACKING_URI}/#/experiments/0/runs/{run_id}")
        
        return run_id

@flow(name="Model Evaluation Pipeline")
def model_evaluation_pipeline(dataset_url: str | None = None):
    """
    Complete model evaluation pipeline
    
    Args:
        dataset_url: Optional URL to external dataset
    """
    print("🚀 Starting Model Evaluation Pipeline")
    
    # Load dataset
    df = load_external_dataset(dataset_url)
    dataset_info = {"shape": df.shape}
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Load/create model
    model = load_or_create_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log to MLflow
    run_id = log_to_mlflow(model, metrics, dataset_info)
    
    print("✅ Model Evaluation Pipeline completed successfully!")
    return {
        "run_id": run_id,
        "metrics": metrics,
        "mlflow_uri": f"{MLFLOW_TRACKING_URI}/#/experiments/0/runs/{run_id}"
    }

if __name__ == "__main__":
    # This will only run for local testing
    # In production, use Prefect deployments
    result = model_evaluation_pipeline()
    print("\n📊 Final Results:")
    print(f"MLflow Run ID: {result['run_id']}")
    print(f"View results: {result['mlflow_uri']}")