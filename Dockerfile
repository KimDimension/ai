FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU 전용 PyTorch 먼저 설치 (CUDA 라이브러리 2GB+ 제외)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ai/ 패키지로 인식되도록 서브폴더에 복사
COPY . ./ai/

CMD ["uvicorn", "ai.main:app", "--host", "0.0.0.0", "--port", "8001"]
