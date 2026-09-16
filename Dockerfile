# 多智能体研究助手 · L2 生产镜像
FROM python:3.12-slim

# 运行环境
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存（requirements 变化少）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝源码
COPY . .

# 非 root 运行：创建用户并授权工作目录
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查打到 /api/health（该端点无需鉴权）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# 单 worker：L2 用同步流式 + 进程内令牌桶限流；多并发请上 L3 异步队列。
# 需要更高吞吐可加 --workers，但会削弱进程内限流的准确性。
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
