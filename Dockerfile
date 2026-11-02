# 知绣坊 · 本地智能客服 —— 容器化运行
# 用法：
#   docker build -t zhixiufang .
#   docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway zhixiufang
#
# 注意：
#   - Ollama 在宿主机运行时，容器内通过 host.docker.internal:11434 访问；
#     需通过环境变量覆盖配置：-e EMBROIDERY_LLM_BASE_URL=http://host.docker.internal:11434/v1
#     -e EMBROIDERY_EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
#   - 员工口令务必用环境变量注入：-e EMBROIDERY_STAFF_PASSCODE=<强口令>
#   - 本镜像在 Linux 下构建，使用 pyproject 声明的依赖（requirements-lock.txt 是 Windows 开发机快照，不用于镜像）

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用缓存层）
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api,retrieval,dev]"

# 再拷业务代码与数据（知识库必须进镜像）
COPY . .

EXPOSE 8000

# 启动前自动确保索引最新；然后启动服务
CMD ["sh", "-c", "python scripts/ensure_kb.py && uvicorn embroidery_agent.api.server:app --host 0.0.0.0 --port 8000"]
