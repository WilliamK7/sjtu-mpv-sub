# sjtu-mpv-sub

面向上海交通大学「课堂实录」（`https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/`）的**零视频下载、轻量流式、渐进双语字幕**本地 `mpv` 连播工具。

- **零视频落盘 / 零物理拼接**：通过 `mpv` EDL 清单直接拉流播放，将同一堂课的连续节次合为单一连续进度条并附带按节章节跳转。
- **按课程记忆画面偏好**：自动记住每门课偏好「PPT 屏录」（`viewNum: 5`）或「教室画面」（`viewNum: 1`）；遇到无音轨的纯屏录流时自动挂载教室画面音轨。
- **零常驻按需唤醒与退出即止**：平时 0 后台进程、0 内存占用；在网页点击播放时通过 `sjtu-mpv://wake` 无黑窗拉起伴侣服务。播放期间若关闭 `mpv` 窗口（或点击网页浮窗的 **「⏹ 停止」**），会立即中止剩余后台分片转录不再继续消耗额度；全部任务结束且 `mpv` 关闭后空闲自动退出。
- **课前静音自动跳过**：针对课堂实录提前约 5 分钟开录及课间无声片段，自动识别百炼 `ASR_RESPONSE_HAVE_NO_WORDS` 并静默跳过，不报错、不重试、不中断后续字幕生成。
- **百炼最新 ASR + LLM 双语精校**：
  - ASR 固定采用 **`qwen-audio-3.1-asr-flash-filetrans`**（`ffmpeg` 抽取 48kbps 单声道 MP3 切片 → 百炼免费 48 小时临时存储 `oss://` → RESTful 异步转写），原生支持毫秒级句/词时间戳，并自动注入课程术语表与时间窗 PPT 关键词作即时热词与上下文增强。
  - 双语精校默认采用 **`qwen3.8-flash`**（已默认配置 `llm_enable_thinking: false` 关闭思考链以保证首段秒开速度，也可在 `config.yaml` 一键切换 `qwen3.7-plus` 或 `deepseek-v4.1-flash`）。

---

## 三步快速上手

### 1. 填写阿里云百炼 API Key（及可选 Workspace ID）

首次运行会自动生成 `config.yaml`（或直接从 `config.example.yaml` 复制一份为 `config.yaml`），在其中填入你的百炼 `api_key`（也可设置在系统环境变量 `DASHSCOPE_API_KEY` 中）：

```yaml
dashscope:
  api_key: "sk-xxxxxxxxxxxxxxxx"
  workspace_id: "" # 若使用业务空间 Key (sk-ws-...)，建议填写业务空间 ID 如 "ws-xxxx"
  asr_model: "qwen-audio-3.1-asr-flash-filetrans"
  llm_model: "qwen3.8-flash" # 备选: qwen3.7-plus / deepseek-v4.1-flash
  llm_enable_thinking: false
```

> 默认的 `mpv_path`、`ffmpeg`、`ffprobe` 已自动指向本机环境；首次注册 `sjtu-mpv://` 协议只需运行一次：
> ```powershell
> pwsh -File "D:\Program Files\sjtu-mpv-sub\install_protocol.ps1"
> ```

### 2. 安装 Tampermonkey 油猴脚本

在浏览器安装 [Tampermonkey](https://www.tampermonkey.net/) 扩展后，新建脚本并粘贴导入：

- [`userscript/sjtu-mpv-bilingual.user.js`](userscript/sjtu-mpv-bilingual.user.js)

### 3. 打开课堂实录一键播放

1. 打开 `https://v.sjtu.edu.cn/jy-application-resourcemanage-ui/` 并进入任意回放播放页。
2. 右下角浮窗会自动识别当天连堂节次，选择「PPT 屏录」或「教室画面」，点击 **「▶ 用 mpv 双语播放」**。
3. **仅首次使用**时浏览器会弹窗询问是否允许打开 `sjtu-mpv://wake`，勾选 **「始终允许」** 后即可永久无感唤醒；之后同一堂课再次打开将直接命中本地 `.ass` 字幕缓存秒播。

---

## 常用维护命令

```powershell
# 在 D:\Program Files\sjtu-mpv-sub 下检查依赖与协议注册状态
uv run sjtu-mpv-sub check

# 运行完整单元与 mpv IPC 集成测试
uv run pytest -v

# 如需卸载 sjtu-mpv:// 注册表项
pwsh -File "D:\Program Files\sjtu-mpv-sub\uninstall_protocol.ps1"
```

## 目录说明

- `config.yaml`：本地服务、工具路径、百炼模型与双语字幕样式配置（`.gitignore` 已忽略）
- `courses.yaml`：按课程教学班 ID 自动记录画面偏好与自定义术语表（可在网页「📝 术语表」弹窗编辑，`.gitignore` 已忽略）
- `data/cache/`：按课程与节次缓存的 `.ass` 双语字幕与 `session.edl`
- `data/logs/server.log`：伴侣服务运行日志
- `AGENTS.md`：项目架构说明、已验证的平台/百炼接口细节与本地维护约定
