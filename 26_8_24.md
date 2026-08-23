# DCBF Update - 2026-08-24

## 中文

- 主动学习因覆盖率 plateau 停止时，`app.log` 现在会记录触发指标、当前覆盖率、硬阈值、最近增量以及 `plateau_generations` 和 `min_coverage_delta`，不改变原收敛判据。
- `dcbf plot-errors` 的内置默认力显示模式改为 `components`；显式命令行参数和用户保存的默认值仍优先。
- Reduce 的默认 `chunk_size` 改为 `1000000`，并新增 84 元素 `l2k3` 通用势作为 Reduce 默认通用模型；自定义模型行为不变。
- 自动 coverage query 复用 `npt_max_cell_volume_filter_factor` 检查 MD 稳定性。首次出现 `V/V0` 超限时立即停止该条轨迹，在 `app.log` 报告 stability failure，只保留爆炸前的稳定帧参与覆盖率计算。
- `augment_existing` Dataset Builder 现在会在 DFT 前使用原有严格结构指纹剔除与已有数据重复的候选，减少重复 SCF 计算；DFT 后的最终去重仍保留为保险。

## English

- When active learning stops because of a coverage plateau, `app.log` now records the triggering metric, current coverage, hard target, recent increments, `plateau_generations`, and `min_coverage_delta`, without changing the convergence criterion.
- The built-in force display default for `dcbf plot-errors` is now `components`; explicit CLI arguments and saved user defaults still take precedence.
- Reduce now defaults to `chunk_size=1000000` and ships a new 84-element `l2k3` universal Reduce potential. Custom model behavior is unchanged.
- Automatic coverage-query MD reuses `npt_max_cell_volume_filter_factor` for stability checks. A trajectory is stopped when `V/V0` first exceeds the limit, the failure is reported in `app.log`, and only the stable prefix is retained for coverage analysis.
- In `augment_existing` mode, Dataset Builder now applies the existing strict structure fingerprint before DFT to remove candidates already present in the input dataset. The final post-DFT deduplication remains as a safety check.
