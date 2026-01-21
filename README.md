# Human Parsing API Service

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![API Status](https://img.shields.io/badge/API-Production%20Ready-brightgreen.svg)

**Extended SCHP Human Parsing API Service** - Production-ready FastAPI service for human parsing and mask generation with advanced clothing segmentation capabilities.

![Human Parsing Demo](./demo/lip-visualization.jpg)

## 🚀 Key Features

- [x] **Production-ready API** with FastAPI framework
- [x] **Dual model support**: SCHP (ATR/LIP) + Segformer for clothing
- [x] **Advanced mask generation** with configurable processing steps
- [x] **Multiple input formats**: Base64 JSON + File upload
- [x] **Docker deployment** with volume mounts for development
- [x] **Flexible target groups**: clothing, upper_clothes, lower_clothes, accessories
- [x] **Debug mode** with intermediate results saving
- [x] **High-performance** with model caching and async processing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│  SCHP Model     │
│                 │    │                  │    │  (ATR/LIP)      │
│ - Web UI        │    │ - Base64/Upload  │    │                 │
│ - Mobile App    │    │ - Validation     │    │ - Human parsing │
│ - CLI Tools     │    │ - Processing     │    │ - 18 labels    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Segformer       │    │  Mask Generator │
                       │  (Clothing)     │    │                 │
                       │                  │    │ - Expansion     │
                       │ - Upper clothes  │    │ - Body inclusion│
                       │ - Better accuracy│    │ - Head protect  │
                       │ - 18 labels     │    │ - Soft edges    │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd human_parser

# Build and run with Docker Compose
docker-compose up --build

# For local development with volume mounts
docker-compose -f docker-compose.local.yml up
```

### Local Development

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate schp

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
# ATR model (recommended for fashion)
wget -O exp-schp-201908301523-atr.pth https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP/view?usp=sharing

# Start API server
python api/main.py
```

## 📋 Available Models

| Model | Dataset | mIoU | Labels | Best For |
|-------|---------|-------|---------|----------|
| SCHP | ATR | 82.29% | 18 | Fashion AI, clothing |
| SCHP | LIP | 59.36% | 20 | Complex scenes |
| SCHP | Pascal | 71.46% | 7 | Body parts |

### ATR Labels (Recommended)
```
0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
16: Bag, 17: Scarf
```

## 🔌 API Endpoints

### 1. Generate Mask (SCHP)

**POST** `/api/generate-mask/`

Generate inpainting mask from parsing map using SCHP model.

#### Request Body
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "target_labels": [4, 5, 6, 7, 8, 16, 17],
  "target_groups": ["clothing"],
  "protect_labels": [11],
  "protect_groups": ["head"]
}
```

#### Response
```json
{
  "mask_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 2. Generate Mask with File Upload

**POST** `/api/generate-mask-upload/`

Easier for local testing with file uploads.

#### Form Data
```
image: [file] - JPEG/PNG image
target_groups: "clothing,upper_clothes"
```

### 3. Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "model": "schp_atr"
}
```

## 🎯 Target Groups

| Group | Labels | Description |
|-------|--------|-------------|
| `clothing` | [4,5,6,7,8,16,17] | All clothing items |
| `upper_clothes` | [4,7,8] | Upper body clothing |
| `lower_clothes` | [5,6,9,10] | Lower body clothing |
| `body` | [12,13,14,15] | Body parts |
| `head` | [1,2,3,11] | Head and accessories |

## 📚 Usage Examples

### Python Client

```python
import requests
import base64

# Read image and encode to base64
with open("person.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# API request
response = requests.post("http://localhost:8000/api/generate-mask/", json={
    "image_base64": image_data,
    "target_groups": ["clothing"],
    "protect_groups": ["head"]
})

# Save mask
mask_data = base64.b64decode(response.json()["mask_png_base64"])
with open("mask.png", "wb") as f:
    f.write(mask_data)
```

### cURL Examples

#### Base64 JSON Request
```bash
# Convert image to base64
base64 -w 0 person.jpg > image.b64

# Send request
curl -X POST "http://localhost:8000/api/generate-mask/" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$(cat image.b64)\",
    \"target_groups\": [\"clothing\"],
    \"protect_groups\": [\"head\"]
  }" \
  -o mask.png
```

#### File Upload (Easier for testing)
```bash
curl -X POST "http://localhost:8000/api/generate-mask-upload/" \
  -F "image=@person.jpg" \
  -F "target_groups=clothing,upper_clothes" \
  -o mask.png
```

#### Different target groups
```bash
# Only upper clothing (bra, dress, belt)
curl -X POST "http://localhost:8000/api/generate-mask-upload/" \
  -F "image=@person.jpg" \
  -F "target_groups=upper_clothes" \
  -o upper_mask.png

# Only lower clothing (skirt, pants, shoes)
curl -X POST "http://localhost:8000/api/generate-mask-upload/" \
  -F "image=@person.jpg" \
  -F "target_groups=lower_clothes" \
  -o lower_mask.png
```

## 🐳 Docker Configuration

### Production (`docker-compose.yml`)
```yaml
version: '3.8'
services:
  human-parser-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./debug:/app/debug
    environment:
      - DEBUG_LOCAL=false
```

### Development (`docker-compose.local.yml`)
```yaml
version: '3.8'
services:
  human-parser-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./api:/app/api
      - ./parsing_inference.py:/app/parsing_inference.py
      - ./generate_mask.py:/app/generate_mask.py
      - ./debug:/app/debug
    environment:
      - DEBUG_LOCAL=true
```

## 🔧 Development

### Debug Mode
Set `DEBUG_LOCAL=true` to enable:
- Intermediate parsing maps saving
- Mask generation steps visualization
- Detailed logging

```bash
export DEBUG_LOCAL=true
python api/main.py
```

### Model Performance
- **SCHP ATR**: ~50ms per image (CPU)
- **Memory usage**: ~2GB RAM
- **Supported formats**: JPEG, PNG
- **Max resolution**: 1024x1024

## 📊 Benchmark Results

| Model | Dataset | mIoU | Inference Time | Memory |
|-------|---------|-------|---------------|---------|
| SCHP | ATR | 82.29% | 50ms | 2GB |
| SCHP | LIP | 59.36% | 55ms | 2.1GB |
| SCHP | Pascal | 71.46% | 45ms | 1.8GB |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Citation

Please cite our work if you find this repo useful in your research.

```latex
@article{li2020self,
  title={Self-Correction for Human Parsing}, 
  author={Li, Peike and Xu, Yunqiu and Wei, Yunchao and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2020},
  doi={10.1109/TPAMI.2020.3048039}}
```
