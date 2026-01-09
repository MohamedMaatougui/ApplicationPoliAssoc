import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from utils.db import fetch_df
from utils.models_kit import MODELS


load_dotenv()
app = Flask(__name__)

# strip any accidental spaces / new-lines
REPORT_ID = os.getenv("REPORT_ID", "").strip()
TENANT_ID = os.getenv("TENANT_ID", "").strip()
EMBED_URL = f"https://app.powerbi.com/reportEmbed?reportId={REPORT_ID.strip()}"

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    try:
        embed_token = get_embed_token(REPORT_ID)
    except Exception as e:
        embed_token = None
        print("Token error:", e)          # terminal only

    config_json = {
        "reportId": REPORT_ID,
        "embedUrl": f"https://app.powerbi.com/reportEmbed?reportId={REPORT_ID}",
        "accessToken": embed_token,
    }
    return render_template("index.html", config_json=config_json)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    sample_id = data.get("sample_id")
    model_id  = data.get("model_id")

    if not sample_id or not model_id:
        return jsonify({"error": "sample_id and model_id required"}), 400
    if model_id not in MODELS:
        return jsonify({"error": f"model_id '{model_id}' not found",
                        "available": list(MODELS.keys())}), 400

    df = fetch_df("SELECT * FROM features WHERE sample_id = :sid", {"sid": int(sample_id)})
    if df.empty:
        return jsonify({"error": "sample_id not found"}), 404

    entry   = MODELS[model_id]
    X       = entry["features"].transform(df) if entry["features"] else df
    y_pred  = entry["model"].predict(X)[0]
    return jsonify({"model_id": model_id, "prediction": float(y_pred)})