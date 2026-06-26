from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_simulate_profile_endpoint_returns_image():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:] = [255, 255, 255]

    success, encoded = cv2.imencode(".png", image)
    assert success

    payload = {
        "file": (BytesIO(encoded.tobytes()), "face.png"),
        "source_landmarks": "[[0,0],[10,10],[20,20]]",
        "target_landmarks": "[[5,2],[12,11],[18,22]]",
    }

    response = client.post("/v1/simulate-profile", files=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
