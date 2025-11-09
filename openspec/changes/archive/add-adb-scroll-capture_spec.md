## ADDED Requirements（Python 版实现）

### Requirement: ADB 分页滚动采集器
系统 SHALL 提供一个 Python 命令行采集器，自动完成滚动与分页截图，输出到指定目录；所有参数均有可用默认值，可直接最小化运行。

#### Scenario: 基本采集流程
- WHEN 用户运行 `python python/capture/main.py`
- AND 设备已在线或系统连接 `127.0.0.1:5555`
- THEN 系统执行：截图 → 滚动（可为 swipe/tap/keyevent）→ 截图，循环直至当前页底部
- AND 当截图哈希重复或检测到底部空白，则视为当前页到底
- THEN 系统点击默认 “下一页” 坐标，等待页面加载后重复上述流程
- AND 所有截图以顺序文件名保存在 `prefix/` 目录下（默认 `capture/`）

#### Scenario: 末页与补滑
- GIVEN OCR 识别到页码形如 `n/m` 且 `n==m`
- WHEN 当前页到底但仍可进行少量补滑
- THEN 系统执行最多 8 次补滑与截图；当哈希不变或底部空白检测为真，则确认最后一页最底并结束

#### Scenario: 错误处理
- WHEN `adb` 不可用或未能连接设备
- THEN 系统输出有用错误信息并退出非零状态码

### Requirement: CLI 配置项（Python）
系统 SHALL 提供以下 CLI 参数以提升可用性，且提供默认值以支持最小运行：
- `--device-id <id>`：可选，ADB 设备 ID；未提供时自动选择在线设备或连接 `127.0.0.1:5555`
- `--adb <path>`：可选，ADB 可执行路径，默认 `adb`
- `--mode <swipe|tap|keyevent>`：可选，滚动模式，默认 `swipe`
- `--prefix <string>`：可选，输出文件前缀与目录名，默认 `capture`
- `--interval <ms>`：可选，步骤间隔毫秒，默认 `600`
- `--count <n>`：可选，每页截图张数上限，默认 `6`
- `--langs <list>`：可选，OCR 页码语言，默认 `ch_sim,en`

#### Scenario: 默认参数计算
- GIVEN 未提供 `--mode swipe` 的滑动坐标
- WHEN 获取到屏幕尺寸
- THEN 系统自动计算中轴从下至上的滑动参数（约 80%→25% 高度，持续 500ms）

#### Scenario: 翻页坐标默认值
- GIVEN 未提供下一页或上一页坐标
- WHEN 获取到屏幕尺寸
- THEN 系统使用底部右侧（下一页）与底部左侧（上一页）坐标进行翻页

#### Scenario: 输出命名规范
- GIVEN 采集第 `p` 页第 `i` 张截图
- THEN 文件名为：`<prefix>/<prefix>_p{p}_{i:04}.png`

### Implementation Notes
- 滚动到底部判定：基于相邻截图的哈希一致性与底部空白统计（均值/方差/白像素比例/边缘密度）。
- 页码识别：在底部右侧区域进行 OCR（`easyocr`），匹配 `n/m` 模式以确定是否为最后一页。
- 翻页稳健性：翻页后预览截图与上一页哈希相同时重试最多 3 次，判定末页后停止。