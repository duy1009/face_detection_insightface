
# Setup on server
### Pull image
```sudo docker pull duy1009/insightface_jetson:1.0```

### Start
```sudo docker run -p 8000:8000 -it duy1009/insightface_jetson:1.0 /bin/bash```

### Run
```uvicorn main:app --host 0.0.0.0  --port 8000 --reload```

# Test
```python3 src/test_request.py```

# Build images for Jetson nano
```sudo docker build --platform="linux/arm64/v8" --progress=plain -t duy1009/insightface_jetson:1.0 .```