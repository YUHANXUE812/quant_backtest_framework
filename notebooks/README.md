# 因子研究展示

- `momentum_factor_research.ipynb`：13 个动量因子的定义、数据口径与三个相关性表格。
- `momentum_factor_research.ipynb` 是唯一运行入口：读取数据、计算因子、相关性分析、截面回归和结果保存都在 notebook 单元格中完成。
- 13 个因子位于 `src/backtester/signals/momentum_factors.py`；相关性与回归统计函数位于 `src/backtester/analytics/`。这些是可测试的计算函数，不会自动进入策略或回测流程。
- `momentum_factor_correlation_report.html`：无需启动 Jupyter 即可查看的三表报告。
- `factor_regression_report.html`：按参考格式展示的因子回归与 t 值汇总表。
- `factor_ic_report.html`：IC 均值、标准差、IR、方向稳定性和显著性频率汇总表。
- `results/layer_backtest/index.html`：13 个因子的五层组合回测报告入口。
- `results/`：因子暴露、相关性、回归、IC 以及各因子分层回测表格和图形。

重新运行时打开 `momentum_factor_research.ipynb` 并执行全部单元格。

因子先按月执行 MAD 去极值、截面标准化和缺失值置零，再在每个月的股票截面计算 Pearson 相关系数。三张表分别是最近 36 个月相关系数序列的均值、样本标准差及 `均值 / 标准差`。

截面回归以相对沪深300的下月超额收益为因变量、标准化因子暴露为解释变量，并使用 `sqrt(流通市值)` 作为 WLS 权重。结果包括月度因子收益、标准误、t 值及六项汇总评价指标。

IC 研究计算当月因子暴露与下月个股收益的横截面 Pearson 相关系数，并汇总 IC 均值、样本标准差、`|均值|/标准差`、IC 正值占比和 `|IC|>0.02` 占比，同时保存累计 IC 曲线。

分层回测将每个因子从小到大划分为五个行业内等权组合，基准为全部合格股票等权组合，多空为组1减组5。因子在月末计算，下一自然月首个交易日收盘建仓，并持有至月末；当前研究结果不计交易成本。

