#!/usr/bin/env python3
"""
Script to run the model evaluation pipeline via Prefect
"""
import os
from prefect import get_client
from model_evaluation_flow import model_evaluation_pipeline
import asyncio

async def run_pipeline(dataset_url: str = None):
    """Run the pipeline through Prefect"""
    
    # Set Prefect API URL
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    
    print("🚀 Starting model evaluation pipeline via Prefect...")
    
    # Run the flow
    result = await model_evaluation_pipeline(dataset_url=dataset_url)
    
    print("✅ Pipeline completed!")
    return result

if __name__ == "__main__":
    # Example usage
    result = asyncio.run(run_pipeline())
    print(f"\n📊 Results: {result}")