# 上线前检查清单（L2 · 少量可信用户试用）

面向"给少量可信外部用户试用"这一目标。按顺序勾完即可安全上线。
带 🔴 的是**阻断项**（不做不能对外），🟡 是**强烈建议**。

---

## 0. 密钥与凭据卫生 🔴

- [ ] **吊销泄露的 Tavily Key**：`.env.example` 曾包含真实 `tvly-dev-47zxBX…`，已改回占位符；
      但若该文件进过 Git 远端，此 key 视为已泄露 → 登录 https://app.tavily.com 删除并重新生成。
- [ ] **核查 Git 历史有无泄密**：在项目根跑
      ```bash
      git log --all -p | grep -E "sk-[A-Za-z0-9]|tvly-[A-Za-z0-9]|gsk_" || echo "无明文密钥泄露"
      ```
      若命中且已 push 到远端 → 对应平台全部吊销重发，并用 `git filter-repo`/BFG 清理历史。
- [ ] **确认 `.env` 未被跟踪**：`git status --ignored | grep .env` 应显示为 ignored；
      `.dockerignore` 也已排除 `.env`（密钥只经环境变量注入，不进镜像）。
- [ ] 新的 LLM_API_KEY / TAVILY_API_KEY 写在服务器本地 `.env` 或 compose 环境变量里，权限 `600`。

## 1. 访问控制 🔴

- [ ] 设置强口令（不要沿用示例值）：
      ```dotenv
      WEB_USERNAME=research
      WEB_PASSWORD=<至少16位随机串>
      ```
      生成强口令：`python -c "import secrets;print(secrets.token_urlsafe(18))"`
- [ ] 验证未授权被拒：启动后 `curl -i http://<host>/api/research -X POST` 应返回 **401**，不带口令打不开首页数据接口。
- [ ] 🟡 若用户极少且固定，可再加 Nginx 层 IP 白名单（`allow x.x.x.x; deny all;`）做第二道锁。

## 2. HTTPS / 反代 🔴

- [ ] 域名解析指向服务器；`deploy/nginx.conf` 里的 `server_name` 改成你的域名。
- [ ] 启用 443 server 块 + certbot 证书：
      ```bash
      sudo certbot --nginx -d research.example.com
      ```
- [ ] 打开 HTTP→HTTPS 跳转（模板里注释的那行取消注释）。
- [ ] 确认 SSE 关键参数在位：`proxy_buffering off; proxy_read_timeout 600s;` —— 否则几分钟的研究会被掐断。
- [ ] Basic Auth 口令只在 HTTPS 下传输（HTTP 明文口令不可接受）。

## 3. 限流与成本护栏 🟡

- [ ] 应用层令牌桶已开（默认 `RATE_LIMIT_PER_MIN=6 / BURST=3`），按用户数微调。
- [ ] Nginx `limit_req` 双层已生效（模板含 `rate=10r/m`）。
- [ ] 🔴 **额度监控**：百炼/Tavily 控制台设用量告警——全员共用你一个 Key，务必盯住月额度，防止意外刷爆账单。
- [ ] 评审轮数上限已由前端 select 限制在 1~6，后端 `max_iterations` 也做了 ge/le 校验。

## 4. 服务健康与持久化 🟡

- [ ] `curl http://127.0.0.1:8000/api/health` → `{"status":"ok"}`。
- [ ] Docker 容器 healthy：`docker compose ps` 看 STATUS 为 `(healthy)`。
- [ ] `restart: unless-stopped` 已配，进程崩溃/重启后自愈。
- [ ] 报告持久卷 `./reports` 已挂载（如启用了落盘），容器重建不丢。
- [ ] 日志能收集：`docker compose logs -f web` 或 systemd journal，便于排障。

## 5. 功能冒烟（真实跑一遍）🔴

- [ ] 浏览器打开 `https://research.example.com` → 弹 Basic Auth → 输口令进入古典首页。
- [ ] 提交一个真实议题 → "工作纪要"实时滚动 → 最终渲染带引用报告 → 下载 .md 正常。
- [ ] 观察是否触发之前修过的两类问题：结论截断自动重试、伪造引用被剔除（终端/纪要留痕可见）。
- [ ] 连续快速点几次「开始研究」→ 应出现 429 限流提示（证明护栏有效）。
- [ ] （可选安全测）诱导抓取内网地址，确认被 SSRF 拦截返回 `[工具错误] …拒绝抓取`。

## 6. 备份与回滚 🟡

- [ ] `.env`（含密钥）在服务器上有安全备份，不入库不入镜像。
- [ ] 记录当前镜像 tag，出问题时 `docker compose down && docker compose up -d` 可快速回到已知良好状态。
- [ ] 通知试用用户访问方式时，用私密渠道发送口令，勿公开群发。

---

## 全绿即上线 ✅

以上 🔴 全部完成、🟡 尽量完成 → 可以交给少量可信用户使用。

### 试用期间留意（决定是否要升 L3）
- 多人同时用时是否频繁超时/断连 → L3 异步任务队列（Celery+Redis）
- 是否需要区分"谁跑了什么/各自配额" → 账号体系 + 数据库
- 服务重启是否会打断正在跑的研究、能否接受 → 任务持久化
