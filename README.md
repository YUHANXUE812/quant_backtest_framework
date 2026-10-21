# A股动量因子稳健性研究框架

该项目面向 A 股横截面动量因子研究。当前版本建立了从数据校验、因子计算、组合形成、成本模拟到参数敏感性分析的最小闭环。它保留 `src/` 布局，但不构建 wheel，也不安装项目自身。

## 研究口径

默认动量定义为：

```text
momentum(t) = P(t - skip_recent) / P(t - lookback) - 1
```

`skip_recent` 用于检验跳过最近若干交易日后结果是否稳健。因子在收盘后计算，目标权重在下一交易日生效，避免使用当天收盘信息获取当天收益。

默认策略是高动量组、月度调仓、仅做多。所有窗口、分组、权重、成本和调仓假设均由 YAML 配置控制。

## 目录

```text
quant_backtest_framework/
├── configs/
│   ├── base.yaml
│   ├── strategy.yaml
│   ├── portfolio.yaml
│   ├── execution.yaml
│   └── dataset.yaml            # 免费数据集口径与路径
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/backtester/
│   ├── data/                   # 读取、清洗、对齐、校验
│   ├── signals/                # 动量、截面变换、信号融合
│   ├── strategies/             # 因子到目标权重
│   ├── portfolio/              # 分配、波动率目标、约束
│   ├── execution/              # 调仓订单和成本模型
│   ├── accounting/             # 账户、持仓、台账
│   ├── engine/                 # 事件边界和回测循环
│   ├── analytics/              # 指标、归因、稳健性工具
│   └── reporting/              # 结果和图表
├── scripts/
│   ├── download_free_data.py
│   ├── build_dataset.py
│   ├── run_backtest.py
│   └── run_validation.py
├── examples/
├── notebooks/
├── tests/
├── environment.yml
└── pyproject.toml
```

## 数据格式

`data/raw/asset_prices.csv` 使用长表：

```csv
date,symbol,close
2024-01-02,000001.SZ,10.25
2024-01-02,600000.SH,8.73
```

`data/raw/benchmark_prices.csv`：

```csv
date,close
2024-01-02,3386.35
```

正式研究应输入经过复权的价格，并进一步补充上市状态、停牌、涨跌停、ST、成交额和行业字段。当前版本不会擅自推断这些数据。

## 免费数据集：申万电子（801080）

不需要 Tushare token。AKShare 用于取得申万电子当前成分股及其接口所示计入日期，BaoStock 用于取得前复权/不复权日行情、交易状态、ST 标记和换手率。流通股本按 `成交量 / (换手率 / 100)` 推导，再与不复权收盘价相乘得到流通市值。

先下载原始数据，再生成月度回归样本：

```powershell
cd D:\quant_backtest_framework
.\.conda-env\python.exe scripts\download_free_data.py
.\.conda-env\python.exe scripts\build_dataset.py
```

只验证或刷新行业名单时可运行：

```powershell
.\.conda-env\python.exe scripts\download_free_data.py --members-only
```

月度样本写入 `data/processed/electronics_momentum_monthly.csv`；逐月筛选计数写入 `data/processed/electronics_momentum_audit.csv`；单只股票下载失败会记录在 `data/raw/download_failures.csv`。

这套免费口径有两项必须保留在研究结论中的限制：

- 免费申万接口只提供当前成分，没有完整历史调入调出记录；而当前接口返回的电子行业最早计入日期晚于本研究区间。默认 `membership_mode: current_constituents` 因而把当前成分股回溯到研究期，并由上市前无行情自动排除，存在明显幸存者偏差。原始计入日期仍保留在名单中，输出的 `membership_quality` 会明确标记该限制。若以后获得真正的历史成分表，将模式改为 `dated_membership` 即可使用调入、调出日期。
- BaoStock 有历史 ST 字段，但没有可靠的历史 PT 专用字段。当前 `is_pt` 只能置为 0，输出中的 `pt_filter_quality` 会标记该缺口。若以后获得可信的历史成分或 PT 数据，可直接替换原始 CSV 后重新构建，不需要改因子处理逻辑。

## 运行

项目已有独立 Conda 环境。新增依赖后可用以下命令同步：

```powershell
cd D:\quant_backtest_framework
.\.conda-env\python.exe -m pip install "akshare>=1.19,<2" "baostock>=0.9.4,<1" "pyyaml>=6,<7"
```

生成确定性的多股票样例：

```powershell
.\.conda-env\python.exe examples\generate_sample_data.py
```

执行基准回测：

```powershell
.\.conda-env\python.exe scripts\run_backtest.py
```

执行动量回看窗口与跳过期敏感性分析：

```powershell
.\.conda-env\python.exe scripts\run_validation.py
```

运行测试：

```powershell
.\.conda-env\python.exe -m pytest -q
```

结果写入 `data/results/`，包括净值台账、实际生效权重、年度收益、摘要、图表和敏感性表。

## 当前边界

- 回测按目标权重计算，不模拟逐股成交数量和 A 股 100 股整数手约束。
- 缺失收益在持仓收益计算时按零处理，正式研究前应结合停牌和可交易状态改进。
- 成本参数只是研究假设，不代表当前券商或交易所收费标准。
- 尚未加入市值、行业和流动性中性化；相关字段应在获得可靠数据后实现。

