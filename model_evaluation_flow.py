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

# MLflow tracking configuration for Docker server with PostgreSQL
MLFLOW_TRACKING_URI = "http://localhost:5000"
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
def load_or_create_model(X_train=None, y_train=None, model_name: str = "RandomForestEvaluationModel", model_version: str = "latest", force_retrain: bool = False):
    """Load existing model from MLflow Registry or create a new one if not found"""
    
    if not force_retrain:
        # First check if model exists in registry
        try:
            import mlflow.tracking
            client = mlflow.tracking.MlflowClient()
            client.get_registered_model(model_name)
            
            # Model exists, try to load it
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Loaded model '{model_name}' version '{model_version}' from MLflow Registry")
            return model, False  # False = not newly trained
            
        except Exception as e:
            print(f"Model not found in registry: {e}")
    
    print("🔧 Creating and training a new model...")
    
    # Create and train new model
    if X_train is None or y_train is None:
        raise ValueError("Training data required when model not found in registry")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Created and trained new RandomForest model")
    return model, True  # True = newly trained

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
def log_to_mlflow(model, metrics: dict, dataset_info: dict, is_newly_trained: bool = False):
    """Log evaluation results and optionally register new model to MLflow"""
    
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
        mlflow.log_param("is_newly_trained", is_newly_trained)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Skip model artifact logging for now due to permission issues
        # We'll focus on metrics logging with PostgreSQL first
        if is_newly_trained:
            print("✅ Model trained successfully (skipping artifact logging for now)")
        else:
            print("Using existing model from registry - no new registration needed")
        
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
    
    # Load/create model (try to load from registry first)
    model, is_newly_trained = load_or_create_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log to MLflow
    run_id = log_to_mlflow(model, metrics, dataset_info, is_newly_trained)
    
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