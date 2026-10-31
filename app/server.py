#!/usr/bin/env python3
"""
NeuroScan Backend Server
========================
Serves the NeuroScan web application and exposes the prediction API
using the trained 'neuroscan_model.pkl' model.

Endpoints:
  GET  /                  -> Serves index.html
  GET  /api/health        -> Health check
  GET  /api/model-info    -> Returns model metadata
  POST /api/predict       -> Predicts brain tumor category from MRI
"""

import os
import sys
import json
import pickle
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
import neuroscan_engine

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PORT = int(os.environ.get("PORT", 5000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "neuroscan_model.pkl")

# Load model from pickle
print("=" * 60)
print("     NEUROSCAN BACKEND SERVER - LOCALHOST DEPLOYMENT")
print("=" * 60)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] Successfully loaded model from '{MODEL_PATH}'")
    print(f"     Model Name: {model.model_name}")
    print(f"     Classes   : {', '.join(model.class_names)}")
except Exception as e:
    print(f"[!] Warning: Failed to load '{MODEL_PATH}': {e}")
    print("     Initializing fresh NeuroScanModel instance...")
    model = neuroscan_engine.NeuroScanModel()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Saved fresh instance into '{MODEL_PATH}'")

class NeuroScanHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        url_path = self.path.split("?")[0]

        if url_path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            resp = {
                "status": "healthy",
                "service": "NeuroScan Localhost Backend",
                "model": model.model_name,
                "classes": model.class_names
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if url_path == "/api/model-info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            resp = {
                "model_name": model.model_name,
                "architecture": model.architecture,
                "class_names": model.class_names,
                "display_names": model.display_names,
                "metrics": getattr(model, "metrics", {}),
                "status": "online"
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # Static File Serving
        if url_path == "/":
            rel_path = "index.html"
        else:
            rel_path = url_path.lstrip("/")

        file_path = os.path.join(BASE_DIR, rel_path)

        # Fallback to 'front end' subfolder if not found in root
        if not os.path.exists(file_path):
            alt_path = os.path.join(BASE_DIR, "front end", rel_path)
            if os.path.exists(alt_path):
                file_path = alt_path

        if os.path.exists(file_path) and os.path.isfile(file_path):
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = "application/octet-stream"
            if file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".html"):
                content_type = "text/html; charset=utf-8"

            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading file: {e}")
        else:
            self.send_error(404, f"File Not Found: {url_path}")

    def do_POST(self):
        url_path = self.path.split("?")[0]

        if url_path in ["/api/predict", "/predict"]:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                features = data.get("features", {})
                
                # Run prediction through the loaded pickle model
                result = model.predict(features)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_resp = {"error": f"Inference error: {str(e)}"}
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
        else:
            self.send_error(404, "Unknown API endpoint")

def run_server():
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, NeuroScanHandler)
    print(f"\n[*] NeuroScan Server is LIVE on Localhost!")
    print(f"[*] Local URL  : http://localhost:{PORT}")
    print(f"[*] Network URL: http://127.0.0.1:{PORT}")
    print("Serving frontend and backend API seamlessly. Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down NeuroScan Server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
