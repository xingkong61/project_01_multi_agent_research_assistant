# L2 部署指南（FastAPI + Docker + Nginx/HTTPS）

面向"小范围公网可用、带访问控制与限流"的目标。相比 Streamlit 测试页，本方案用独立的
古典风格前端（`web/static/`）+ FastAPI 后端（`web/server.py`），支持 SSE 实时进度、
Basic Auth 鉴权、双层限流、SSRF 防护与非 root 容器运行。

## 目录结构（新增部分）

```
web/
├── server.py        # FastAPI：/api/research(SSE) · /api/health · 静态托管 · Basic Auth
├── auth.py          # 凭据校验 + 进程内令牌桶限流
└── static/
    ├── index.html   # 古典学术风前端
    ├── style.css
    └── app.js       # fetch + SSE 消费 + Markdown 渲染
Dockerfile           # python:3.12-slim，非 root，健康检查
docker-compose.yml   # 环境变量注入密钥，仅本机暴露 8000
.dockerignore        # 排除 .env / venv / reports
deploy/nginx.conf    # 反向代理 + SSE 关键参数 + QPS 限流 + HTTPS 模板
```

## 一、准备密钥与口令

在项目根创建 `.env`（**已在 .gitignore/.dockerignore 中排除，切勿提交**）：

```dotenv
LLM_MODEL=qwen3.8-flash
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的真实百炼Key
TAVILY_API_KEY=你的TavilyKey            # 可选

WEB_USERNAME=research
WEB_PASSWORD=一个足够强的口令            # 必填：访问控制
RATE_LIMIT_PER_MIN=6
RATE_LIMIT_BURST=3
```

> ⚠️ 若之前把含真实 Key 的 `.env` push 到过 Git 远端，请立即到百炼/Tavily **吊销并重新生成**。

## 二、方式 A：Docker Compose（推荐）

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/api/health      # → {"status":"ok"}
```

服务只监听 `127.0.0.1:8000`，不直接暴露公网；对外访问一律经下面的 Nginx。

## 三、方式 B：裸机 systemd（无 Docker）

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env && vi .env            # 填密钥/口令
# 前台试跑
.venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 8000
```

systemd 单元 `/etc/systemd/system/research.service`：

```ini
[Unit]
Description=Multi-Agent Research (FastAPI)
After=network.target

[Service]
User=appuser
WorkingDirectory=/opt/research
EnvironmentFile=/opt/research/.env
ExecStart=/opt/research/.venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now research
```

## 四、Nginx 反向代理 + HTTPS

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/research
sudo ln -s /etc/nginx/sites-available/research /etc/nginx/sites-enabled/
sudo nginx -t && sudo nginx -s reload
# 签证书（自动改写为 HTTPS 并配置续期）
sudo certbot --nginx -d research.example.com
```

要点（已在模板中体现）：
- `proxy_buffering off` + `proxy_read_timeout 600s`：否则几分钟的 SSE 研究会被缓冲或掐断。
- `limit_req`：应用层令牌桶之外的第二道限流。
- 站点通过域名 + HTTPS 提供，用户浏览器首次访问会弹 Basic Auth 输入口令。

## 五、验证清单

- [ ] `curl 127.0.0.1:8000/api/health` 返回 ok
- [ ] 不带口令访问 `/` 或 `/api/research` 返回 401
- [ ] 带正确口令能打开古典首页并提交研究，看到实时"工作纪要"滚动
- [ ] 连续快速提交触发 429（限流生效）
- [ ] 让模型抓内网地址（如诱导 fetch `http://169.254.169.254`）被 SSRF 拦截
- [ ] HTTPS 正常、证书未过期

## 六、L2 的已知边界（何时需要升级到 L3）

| 现象 | 原因 | L3 解法 |
|---|---|---|
| 并发用户多时偶发超时/断连 | L2 同步流式，长任务占用连接 | Celery/Redis 异步任务队列 |
| 重启丢正在进行的任务 | 无持久化任务状态 | 任务表 + 结果后端 |
| 无法按用户统计/配额 | 共享口令，无账号体系 | 数据库 + 用户级配额计费 |
| 多实例限流不准 | 令牌桶在单进程内存 | Redis 集中式限流 |

先满足"安全地给少量外部用户试用"即可，出现上述压力再上 L3。
