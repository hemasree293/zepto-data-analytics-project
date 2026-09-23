FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN PIP_REQUIRE_HASHES=0 pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "8001"]