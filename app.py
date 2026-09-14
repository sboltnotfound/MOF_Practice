from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "extra_trees_logqe_model.joblib"
STATIC_DIR = BASE_DIR / "static"


class PredictionRequest(BaseModel):
    features: dict[str, float | None]


def load_model_bundle(path: Path) -> tuple[Any, list[str], int]:
    if not path.exists():
        raise RuntimeError(
            f"Model file is missing: {path}. "
            "Export it from the training notebook as described in README.md."
        )

    saved = joblib.load(path)

    if not isinstance(saved, dict):
        raise RuntimeError(
            "The joblib file must be a dictionary containing "
            "'model' and 'feature_columns'."
        )

    model = saved.get("model")
    feature_columns = saved.get("feature_columns") or saved.get("features")
    log_base = saved.get("log_base", 10)

    if model is None or not feature_columns:
        raise RuntimeError(
            "The model bundle must contain 'model' and 'feature_columns'."
        )

    return model, list(feature_columns), int(log_base)


model, feature_columns, log_base = load_model_bundle(MODEL_PATH)

app = FastAPI(
    title="MOF logQe Predictor",
    version="1.0.0",
    description="Predict logQe using a trained Extra Trees regression pipeline.",
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(feature_columns),
    }


@app.get("/api/features")
def get_features() -> dict[str, Any]:
    return {
        "features": feature_columns,
        "target": "logqe",
        "log_base": log_base,
    }


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    unknown = sorted(set(request.features) - set(feature_columns))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown feature(s): {', '.join(unknown)}",
        )

    ordered_values: dict[str, list[float]] = {}
    for name in feature_columns:
        value = request.features.get(name)
        ordered_values[name] = [np.nan if value is None else float(value)]

    input_frame = pd.DataFrame(ordered_values, columns=feature_columns)

    try:
        predicted_logqe = float(model.predict(input_frame)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc

    if log_base == 10:
        predicted_qe = 10.0 ** predicted_logqe
    elif log_base == math.e:
        predicted_qe = math.exp(predicted_logqe)
    else:
        predicted_qe = log_base ** predicted_logqe

    return {
        "predicted_logqe": predicted_logqe,
        "predicted_qe": predicted_qe,
        "log_base": log_base,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
