# 安卓分页滚动截图脚本

一个基于 ADB 的命令行脚本：连接安卓设备/模拟器，执行“截图 → 下滑 → 截图”直至当前页底部，再点击“下一页”并重复，最后一页会在确认到底后结束。所有截图按顺序保存在指定前缀目录下。

## 简介

本项目提供一个“开箱即用”的安卓分页滚动截图工具：通过 ADB 连接设备，自动完成“截图 → 滚动 → 翻页”直到末页，并支持一键解析为 JSON。所有参数均有合理默认值，基本无需手动配置。

## 如何使用

1) 安装依赖（Python）：

```bash
python -m pip install -r python/requirements.txt
```

2) 开始采集：

```bash
python python/capture/main.py
```

3) 解析为 JSON：

```bash
python python/parse/main.py
```

可选覆盖（按需）：

```bash
# 指定设备
python python/capture/main.py --device-id 192.168.0.101:5555
# 点击滚动
python python/capture/main.py --mode tap
# 按键滚动
python python/capture/main.py --mode keyevent
```

默认行为简介：
- 自动选择在线设备或连接 `127.0.0.1:5555`
- 自动计算稳定的上滑参数与翻页坐标（底部右/左）
- 截图默认存入 `screenshots`；解析默认输出 `activity_items.json`

## 模块结构

- 采集模块：`python/capture/`
  - `main.py`：采集启动器，最小命令即可运行
  - `orchestrator.py`：截图/滚动/翻页编排与默认参数计算
  - `adb_client.py`：ADB 封装（连接设备、截图、滑动、点击、按键）
- 解析模块：`python/parse/`
  - `main.py`：解析启动器，默认读取 `screenshots` 输出 `activity_items.json`
  - `activity_parser.py`：解析核心逻辑

## 解析逻辑（简述）
- 使用 EasyOCR 提取文本与位置框，默认小 batch、单线程，支持可选 ROI 裁剪。
- 在截图右半区域查找分数字段（支持千分位），只保留 1000 的倍数作为有效分数。
- 以分数的垂直中心为参考，在左侧窗口聚合同一行的文字，按 x 从左到右拼接为“道具描述”。
- 规范化数量表达（统一乘号为 `x`，将 `PAIRINGx1` 变为 `PAIRING x1`），并过滤噪声（比值 `1/7`、纯数字、符号、短大写等常见 UI 干扰）。
- 对同一分数保留更完整的描述（优先长文本），最终输出 `items` 列表与 `missing`（从 1000 到 `--max` 的缺失分数段）。

更多参数与 Node 版本用法，请查看 `AGENTS.md`。