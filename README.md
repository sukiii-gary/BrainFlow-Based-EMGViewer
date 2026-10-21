# OpenBCI EMG Viewer · 源码

基于 PyQt6、BrainFlow 和 pyqtgraph 的 8 通道 OpenBCI Cyton 肌电采集与分析程序。支持 WiFi Shield UDP（1000 Hz）、串口 Dongle（250 Hz）、模拟板、数据录制、片段审查、指标导出、阻抗检测，以及摄像头和 Delsys 触发同步。

完整功能介绍、采集步骤与数据格式见 [项目 README](../README.md)。以下命令均在 `SourceCode` 目录中执行。

## 安装与运行

环境：Windows x64、Python 3.12 x64。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 无需硬件，自动开始模拟采集与录制
.\.venv\Scripts\python.exe app/main.py --synthetic

# 启动普通采集界面
.\.venv\Scripts\python.exe app/main.py

# 使用实际 Shield IP 和电脑本地 UDP 接收端口
.\.venv\Scripts\python.exe app/main.py --wifi 192.168.4.1 6677

# 使用串口 Dongle，替换为实际串口
.\.venv\Scripts\python.exe app/main.py --serial COM3
```

实际采集前填写被试 ID，点击“开始”即自动录制，点击“停止”完成保存。默认数据目录为 `app/recordings`，可在界面调整。

## 测试与构建

`scripts/project.ps1` 默认使用本目录的 `.venv`，也可通过 `-Python '环境路径\python.exe'` 指定解释器。

```powershell
# 安装运行和构建依赖
.\scripts\project.ps1 -Action install

# 自动化测试
.\scripts\project.ps1 -Action test

# 模拟板、MP4 编解码和本地 UDP 回环自检
.\scripts\project.ps1 -Action selftest

# 构建 Windows 单文件 EXE
.\scripts\project.ps1 -Action build
```

构建结果位于 `dist/EMG_Viewer_UDP_Sync_Camera_Impedance.exe`。如 PowerShell 阻止脚本执行，可先运行 `Set-ExecutionPolicy -Scope Process Bypass`，仅对当前会话生效。

## 源码入口

| 路径 | 内容 |
| --- | --- |
| `app/main.py` | 程序入口和启动参数 |
| `app/config_viewer.py` | 连接、滤波、显示和录制默认配置 |
| `app/ui/` | 主界面、实时视图和分析对话框 |
| `app/board_manager.py`、`app/wifi_udp.py` | 板卡连接、通道配置和 UDP 数据接收 |
| `app/recorder.py` | Session/Trial 管理与原始数据保存 |
| `app/tests/` | 自动化测试 |
| `emg_python/` | 信号预处理、质量指标和动作分段算法 |
| `firmware/delsys_trigger/` | Arduino Nano 触发固件 |
| `EMG_Viewer_UDP.spec` | PyInstaller 打包配置 |
