# GPU runtime image — for local GPU runs and reproducible benchmarks.
# For CPU development: `pip install -e ".[dev]"` in your local env is enough.
FROM pytorch/pytorch:2.12.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir setuptools wheel

COPY mini_vllm/ ./mini_vllm/
RUN pip install --no-cache-dir -e ".[gpu]"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "mini_vllm.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
