---
author: "Kyle Jones"
date_published: "July 18, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/ml-model-persistence-from-development-to-production-with-python-86980ea695f5"
---

# ML Model Persistence From Development to Production with Python Time series models need to work beyond development. Accuracy in a
notebook means little if the model cannot be reused, tracked, or updated.

### ML Model Persistence From Development to Production with Python 

Time series models need to work beyond development. Accuracy in a notebook means little if the model cannot be reused, tracked, or updated.

Production systems require:

1.  [Saving models without retraining]
2.  [Updating models with new data]
3.  [Version control and rollback]
4.  [Performance monitoring]
5.  [Reproducibility across teams and systems]

Let's walk through a practical workflow for model persistence, versioning, and performance tracking.

### Development and Initial Save
Train and test your model, then save it with full metadata.

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

model = RandomForestRegressor()
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
persistence = ModelPersistence(model_directory="production_models")
metadata = {
    "author": "Jane Smith",
    "training_date": "2024-01-24",
    "training_data_range": "2020–2023",
    "performance_metrics": {
        "test_score": test_score,
        "rmse": 0.23
    },
    "feature_names": X_train.columns.tolist(),
    "target_variable": "monthly_sales"
}
persistence.save_model(model, "sales_forecaster_v1", metadata)
```

This ensures your model can be reused, audited, and shared later.

### Wrapping for Production Use
Wrap the model in a lightweight forecast class with tracking.

```python
class ProductionForecaster:
    def __init__(self, model_name):
        self.persistence = ModelPersistence()
        self.model_name = model_name
        self.model, self.metadata = self.persistence.load_model(model_name)
        self.tracker = ModelPerformanceTracker(model_name)

def make_forecast(self, input_data):
        forecast = self.model.predict(input_data)
        self.tracker.log_performance({
            "timestamp": pd.Timestamp.now(),
            "forecast_values": forecast.tolist()
        })
        return forecast
```

Used in production:

```python
forecaster = ProductionForecaster("sales_forecaster_v1")

def daily_forecast_job():
    today_data = get_latest_data()
    forecast = forecaster.make_forecast(today_data)
    store_results(forecast)
```

### Updating Models Safely
Compare performance before replacing a deployed model.

```python
def update_production_model(new_data, old_model_name="sales_forecaster_v1"):
    persistence = ModelPersistence()
    current_model, metadata = persistence.load_model(old_model_name)

new_model = RandomForestRegressor()
    new_model.fit(new_data['X'], new_data['y'])
    old_score = current_model.score(new_data['X_test'], new_data['y_test'])
    new_score = new_model.score(new_data['X_test'], new_data['y_test'])
    if new_score > old_score:
        new_metadata = metadata.copy()
        new_metadata.update({
            "update_date": pd.Timestamp.now(),
            "performance_improvement": f"{((new_score - old_score) / old_score) * 100:.2f}%",
            "training_data_range": new_data["date_range"]
        })
        persistence.update_model(old_model_name, new_model, new_metadata)
        return True, "Model updated successfully"
    return False, "New model did not improve performance"
```

### Monitoring Model Performance
Track rolling metrics to detect drift or degradation.

```python
def monitor_model_performance(model_name, threshold=0.1):
    tracker = ModelPerformanceTracker(model_name)
    trends = tracker.analyze_performance_trend()
recent = trends["mse_rolling_avg"].iloc[-1]
    baseline = trends["mse_rolling_avg"].iloc[0]
    if recent > baseline * (1 + threshold):
        send_alert(
            f"Model Performance Alert\n"
            f"Model: {model_name}\n"
            f"Baseline MSE: {baseline:.4f}\n"
            f"Current MSE: {recent:.4f}\n"
            f"Degradation: {((recent / baseline) - 1) * 100:.2f}%"
        )
        return "Degradation detected"
    return "Performance stable"
```

### Common Production Scenarios
Let's cover some common scenarios for data science teams. After a model is create, we may need to update it.

This approach focuses on weekly, regular model updates.

```python
def weekly_model_update():
    new_data = collect_weekly_data()
    success, message = update_production_model(new_data)
    if not success:
        send_alert(f"Model update failed: {message}")
```

Another way to update things is using A/B Testing where we have two models running in production at the same time.

```python
def ab_test_new_model(new_model, test_days=7):
    persistence = ModelPersistence()
    persistence.save_model(new_model, "sales_forecaster_beta")
    results = {"production": [], "beta": []}
    for _ in range(test_days):
        # Logic to compare both models
        pass
    return compare_results(results)
```

If we promote a model to production and it doesn't work, we can use Emergency Rollback to return to the previous, stable version.

```python
def rollback_model(model_name, version):
    persistence = ModelPersistence()
    history = get_model_version_history(persistence, model_name)
    for entry in history:
        if entry["model_version"] == version:
            persistence.restore_version(model_name, version)
            return f"Rolled back to version {version}"
    return "Version not found"
```

### Practical Example
```python
def implement_model_persistence(model, model_name="time_series_model"):
    persistence = ModelPersistence()
    metadata = {
        "feature_names": model.feature_names_in_.tolist(),
        "training_date": datetime.datetime.now().isoformat(),
        "performance_metrics": {"mse": 0.25, "mae": 0.4},
        "model_parameters": model.get_params()
    }

path = persistence.save_model(model, model_name, metadata)
    print(f"Model saved to: {path}")
    model_loaded, meta_loaded = persistence.load_model(model_name)
    print("Model loaded")
    tracker = ModelPerformanceTracker(model_name)
    for _ in range(5):
        tracker.log_performance({
            "mse": 0.25 + np.random.normal(0, 0.05),
            "mae": 0.4 + np.random.normal(0, 0.05)
        })
    return tracker.analyze_performance_trend()
```
### Best Practices 

Use Version History to track the current and past versions. This lets you use tools like Git and set up automatic hooks for things like AWS Lambda with \$LATEST

```python
def get_model_version_history(persistence, model_name):
    path = persistence.model_directory / model_name
    history = []
    for version in path.glob("archive/v*"):
        with open(version / "metadata.json") as f:
            history.append(json.load(f))
    return sorted(history, key=lambda x: float(x["model_version"]))
```

And while we never want bad things to happen, they sometimes do. So data scientists should back up their models just in case.

```python
def create_model_backup(persistence, model_name):
    backup_dir = persistence.model_directory / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{model_name}_{timestamp}"


    model, metadata = persistence.load_model(model_name)
    persistence.save_model(model, str(backup_path), metadata)
```
Production-ready time series forecasting depends on robust model management. Save models with full metadata. Track their performance. Update and roll back with confidence. You do not need a complex MLOps platform to do this well. Start with clean code and a disciplined workflow. Add automation later.
