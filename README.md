# teentree-android-maa

基于 Rust 与 MAA Framework 的安卓自动化答题示例项目。

项目会通过 ADB 连接设备，加载本地 MAA 资源流水线，监听识别回调，将 OCR 内容发送到本地 LLM 服务，解析答案后自动点击选项并切到下一题。

## 功能概览

- 自动连接已发现的第一个 ADB 设备
- 加载 resource 目录下的任务资源
- 识别答题界面文本并筛选题目窗口
- 调用本地 LLM 推断答案关键词
- 自动点击匹配选项并执行“下一题”
- 识别到答题结果后自动点击“关闭”

## 环境要求

- Windows
- Rust 工具链（建议 stable 最新版）
- Android ADB 环境（设备已开启 USB 调试）
- 本地可访问的 LLM API（默认地址见 src/main.rs）

## 目录说明

- src/: Rust 主程序
- resource/: MAA 资源与流水线
- config/: 运行时配置目录（运行后可能自动生成）
- debug/: 运行时调试输出目录
- MAA-win-x86_64-v5.9.0/: MAA SDK 与运行库

## 使用方式

1. 安装依赖并确认 ADB 可用

    rustup toolchain install stable
    cargo --version
    adb devices

2. 启动本地 LLM 服务（确保与代码中的接口一致）

    默认接口:
    - LLM_API_BASE: http://localhost:11434/api/generate
    - LLM_MODEL: deepseek-v3.1:671b-cloud

3. 在项目根目录运行

    cargo run

## 可配置项

当前版本将以下参数写在 src/main.rs 中，可按需修改：

- LLM_API_KEY
- LLM_API_BASE
- LLM_MODEL
- LLM_TIMEOUT_SECS

如果你希望改成环境变量方式，可后续扩展为读取 .env 或系统环境变量。

## 常见问题

- 未找到设备
  - 确认 adb devices 能看到设备
  - 确认设备已授权 USB 调试

- 连接控制器失败
  - 检查 ADB 路径和设备地址
  - 确认未被其他自动化工具独占

- LLM 请求失败
  - 确认 LLM 服务已启动
  - 确认接口路径与模型名正确
  - 检查本机防火墙或端口占用

## 备注

项目已通过 .gitattributes 将 MAA 目录标记为 vendored，以避免影响 GitHub Language 统计。
