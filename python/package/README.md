# PINC：从变化检测到来源诊断

## Scientific scope and saved KS delivery — 2026-10-07, version 0.2.0.dev2026100702

The software separates first-stage alarms from final finite-family model-involvement reports. A first alarm can lead to an unresolved follow-up; technical and support unavailability remain explicit. The conditional two-stage error statement requires admitted true-law coverage, matched conditional calibration and a follow-up guarantee uniform conditional on the full entry history. Package execution and observed proportions do not verify those assumptions or supply switching-source posterior probabilities.

RND2 scientific H10 adapters explicitly invoke native `evaluation="bellman", remaining_horizon=10`. The packaged engineering four-path example uses H3; it is not the H10 detection experiment. The generic native API keeps its explicit nearest reproduction option. Future auxiliary continuation remains approximated, so actual-state queries do not certify true economic Q, optimal repair, or latent recovery.

The completed KS attempt017 uses frozen pre-event fitted G/Q with causally updated auxiliary histories, actual-action fitted-Q support and the raw previous-action anchor. Its measurement is squared FULL employment-prediction residual. It is not the full online RA procedure that refits G/Q after every release. Monitor dates70–77 precede source dates78–85, entered only for the relevant actual first alarm. N and PUBLIC_MACRO are the two no-employment-change candidates, each using79 fixed calibration slots. The fixed128 sample retains127 corrected completed economies and one technical failure; first alarms, final model reports, unresolved, no-alarm and unavailable states are distinct. Macro seeds are the replication unit. Fixed-price KS partial equilibrium, group readouts and finite-family involvement do not establish household causal localization, full latent/model/mixed classification, GE, welfare or repair.

**The KS scientific adapter/host/sampler is not inside either package.** It remains in `experiments/ks_main_application_20261005/attempt017_fixed128_macro/run.py` and its research dependencies. The new installed entry only regenerates tables and three PNG/PDF figures from already completed corrected saved CSV readback; it does not import that runner, fit a model, draw calibration or simulate an economy. It uses `017_OUTCOMES.csv`, `017_ECONOMIC_RESPONSES.csv`, `017_BOUNDARY_READBACK.csv` and `017_READBACK_STATUS.json`, supplied separately by the caller. Install plotting support with `pip install 'pinc-toolbox[reporting]'` or equivalent `matplotlib>=3.6` alongside the supplied wheel. No KS data bank is bundled.

From the research checkout, with an installed Python and a **new** output directory:

```sh
/absolute/installed/python -I -m pinc_toolbox.ks_saved_exhibits_cli \
  --input-dir "$PWD/experiments/ks_main_application_20261005/attempt017_fixed128_macro" \
  --output-dir /absolute/new/ks_saved_exhibits
```

The Dynare-facing archive exposes the same operation through `pinc_ks_saved_exhibits(input_dir, output_dir, python_executable)` after adding its `matlab` directory. Tables reconstruct counts and pointwise Wilson pipeline-delivery intervals from saved outcome rows; economic figures average saved paired path differences over127 available economies. Technical failure is retained in the128 outcome denominator. The new figures may differ cosmetically from the original exhibits; all original results remain untouched. Actual installation and regeneration checks are in `note/middle_loop_closure_20261007/SOFTWARE_DELIVERY.md`.


## RA-GQ actual-state update — 0.2.0.dev2026100301 (2026-10-03)

The current joint record is `(X_t, E_t)`, with `E_t = (delta_gamma, delta_m, productivity residual)` from released completed rolling history. Historical auxiliary values stay paired with their dated physical observations; nodes sample those joint records. Previous action anchors the response comparison. The issued base forecast is stored before its outcome and the new residual is computed only after release, without rewriting past errors with a refreshed fit.

For new RA-GQ use, explicitly select `q_evaluation="bellman"` in `prepare_rnd2_learner` or the counterfactual JSON `learner` object. The bundled four-path example now selects Bellman with horizon 3. The low-level native API retains its historical nearest default for reproducibility. Native JSON / `pinc_native_gq` queries now accept `evaluation="bellman"` and optional positive `remaining_horizon`; omission inherits the fitted horizon. For example: `{"x": [0.5, 0.0, 0.05], "aux": [0.0, 0.0, 0.0], "evaluation": "bellman", "remaining_horizon": 10}` for a full model fitted at horizon at least 10. `detail=true` is the legacy nearest explanation and is rejected for Bellman.

Actual-state evaluation uses the current reward and G prediction directly and projects only successor continuation onto historical joint nodes. Future response/motion coordinates remain held approximations and residual follows simulated physical innovation; this update does not certify a true auxiliary transition law, latent recovery, economic optimality, repair success or detection power. The root research entry point and the portable native backend use their respective existing kernels; no research evidence is bundled or replaced.

The new Python archive contains the wheel, this README and the four-path engineering example. The Dynare-facing archive contains the same wheel, POSIX MATLAB/Octave bridges (including `pinc_counterfactual.m`) and that example. Install the wheel with its ordinary Python dependencies. In Octave/MATLAB add the extracted `matlab` directory, then call the bridges with an absolute path to that installed Python. The bridge executes the Python learner; it is not a separate Dynare-native Bellman implementation. Current installed Python and actual Octave validation is recorded in `reports/RAGQ_TOOLBOX_SYNC_20261003.md`; MATLAB and Dynare-internal model execution are not claimed by that check. Old archives remain intact.


历史开发包 **0.2.0.dev2026092702**：本次增加实验性的联合组件报告接口：在调用者给定预先路由及匹配条件校准记录后，分别报告有限目录下的模型参与与潜在状态参与。新安装的 Python wheel 与解压的 Octave 适配器对四条已保存的 routed 输入逐字段一致；这只验证报告算术与接口，不代表条件模拟工作流、模型来源准确率或潜在来源识别已经完成。旧版本及其示例保留。接口不生成条件未来、不验证历史/核假设、不提供来源概率或真实经济 Q 保证。MATLAB 执行和 Dynare 模型模拟未在本次验证。

## Branch learner 的 Q 查询

`RND2BranchLearner` 默认保留原 nearest-node Q 查询。需要 direct Bellman 查询时，可显式传入 `q_evaluation='bellman'` 和 `query_horizon=10`；这两个设置保存在 learner 实例及其副本中，并交由原生 `RND2GQ.q_values` 执行相同的支持与有限/无限 horizon 检查。`prepare_rnd2_learner` 和 JSON 接入的 `learner` 设置也接受这两个选项。`query_horizon=None` 继承拟合模型的期限：有限模型使用其有限期限，无限模型使用无限续期。该路由只指定 Q 查询方式，不构成来源准确率或 power 结论。

## 当前入口

- **A：有行动数据的原生 GQ / residual-augmented 分支与两层诊断。** 使用已安装的 `native_gq_cli`、原生 eta query 和 `eta_online_gq_two_stage`；完整当前输入 schema、固定诊断参考与滚动行为 fit 的区别、source 独立参考、实际命令和 `.m` 调用见 [行动数据工作流](examples/action_workflow/README.md)。它是已提取的 RND2 原生程序与受限线性诊断路线，不能冒称任意模型的自动 RA-GQ。
- **B：观测 HS 环境/增长应用。** 使用下方自包含 driver，显式选择 complete-only、stopped-prefix 或 grouped-composite 来源规则。
- **C：固定行动、固定公开环境与双固定诊断。** `pinc_toolbox.counterfactual.run_four_paths` 同步运行一主三子，各分支保留自己的状态、已释放历史和辅助记忆。四分支语义、带符号对比和缺失处理见下方 [Four synchronous path diagnostics](#four-synchronous-path-diagnostics)。这一路输出有限路径回报诊断，不自动提供来源检验。

A 的 `native_source_loop` 主循环目前仅支持 base（拒绝 `auxiliary != base`），连接初始化、eta、在线检测、支持内采集和两层结果；full RA 四分支使用 C 的新入口，不表示自动两层 host 已支持 full RA。示例 host 是显式模拟器，真实应用须提供执行与发布适配器。见 [原生主循环说明](examples/native_source_loop/README.md)。安装下节 wheel 后运行：

```sh
.venv/bin/python -I -m pinc_toolbox.native_source_loop_cli --input examples/native_source_loop/request.json --host-file examples/native_source_loop/simulation_host.py --output native-loop.json --html-output native-loop.html
```

分步接口与工程 fixtures：

```sh
.venv/bin/python -I -m pinc_toolbox.native_gq_cli --input examples/action_workflow/native_gq.json --output native.json
.venv/bin/python -I examples/action_workflow/native_eta_query.py --native-request examples/action_workflow/native_gq.json --eta-selection examples/action_workflow/eta_selection.json --output native_eta.json
.venv/bin/python -I -m pinc_toolbox.two_stage_report_cli --input examples/action_workflow/eta_online_gq_two_stage.json --output action_report.json --html-output action_report.html
.venv/bin/python -I -m pinc_toolbox.source_acquisition_cli --input examples/action_workflow/source_acquisition.json --output acquisition.json
```

这些工程 fixtures 分别展示当前接口，不是假称一条匹配的在线科学实验。一般应用仍须提供真实动作/结果、可用性时钟、可预测权重和预留来源参考；归档历史不是当前接口文档。

### 条件历史 saved-terminal 报告（工作树 opt-in）

当前工作树可选择 `conditional_history_saved_report`，仅把已保存的 terminal JSON 转成报告；它不运行 conditional sampler。该入口尚未加入现有 package014 distributions。请求沿用 `two_stage_report_cli`：`stage1` 给正常报警记录，`context.route` 设为该路由，可选提供 `history_reference`、`detector_reference` 和 `maintained_assumptions`，`source` 直接放 terminal 对象。

以下只示意请求形状；`source` 必须替换为真实的完整 terminal 对象，示例占位符不是有效输入：

```json
{
  "stage1": {"status": "alarm", "H": true},
  "context": {
    "route": "conditional_history_saved_report",
    "history_reference": "saved-history-id",
    "detector_reference": "detector-id",
    "maintained_assumptions": {}
  },
  "source": {"status": "replace-with-saved-terminal-object"}
}
```

`stage1` 只接受 `status` 和可选的 `H`；两者必须与 terminal 派生状态一致。假设字段是调用者声明，软件不会验证其真实性。允许的假设名为 `finite_fully_specified_catalogue`、`complete_public_history_fixed`、`fixed_detector_alarm_measurable`、`correct_candidate_conditional_kernels`、`independent_CAL_and_observed_future`、`original_scores_tails_and_qualification`、`predeclared_capped_first_available_sampling`。

保存为请求 JSON 后，从仓库根目录用工作树源码运行（不要启用 `-I`）：

```sh
PYTHONPATH=software/pinc_toolbox/src python -m pinc_toolbox.two_stage_report_cli --input request.json --output report.json --html-output report.html
```

model、latent 和 mixed 均保留 present/absent/unresolved 三值；mixed 布尔值只表示已断言共同存在。invalid 或未明确维护全部假设时不填来源错误界，Type II 仍未估计，p 值不是来源后验。

Dynare/Octave 调用也须让 Python 从工作树源码导入该入口；现有 package014 安装不包含此入口。新路由只接受原请求 JSON 的文件路径，结构体输入会明确报错并提示改传文件。Octave 的 `jsondecode`/`jsonencode` 会把 `null` 改成 `[]`；二者分别可表示无结果和空集合，不能互相替换。原有路由的结构体输入保持不变。例如在仓库根目录、使用已有兼容 Python 环境：

```matlab
setenv('PYTHONPATH', fullfile(pwd, 'software', 'pinc_toolbox', 'src'));
addpath(fullfile(pwd, 'software', 'pinc_toolbox', 'distribution', 'dynare_eta'));
[result, raw_json, report_text] = pinc_two_stage_report( ...
    '/absolute/path/request.json', '/absolute/path/python', '/absolute/path/report.html');
```

传递原文件路径，不要先把请求 `jsondecode` 为结构体。

## 安装与第一次运行

在解压后的分发根目录执行：

```sh
python -m venv .venv
.venv/bin/python -m pip install ./pinc_toolbox-0.2.0.dev2026092701-py3-none-any.whl
.venv/bin/python -I examples/hs2022/run_example.py --config examples/hs2022/smoke.json --source-policy stopped_prefix --case reference --output hs_null
.venv/bin/python -I examples/hs2022/run_example.py --config examples/hs2022/smoke.json --source-policy stopped_prefix --case mixed --output hs_change
```

打开 `hs_change/report.html` 查看两层结果和 Details。输出目录须为新目录；固定 smoke 只是工程检查，不能用于功效声明。 HS balanced16 is complete; see the current validation note. HS80 and RND480 are complete; see the final confirmation summary.

**显式选择来源规则。** 上述 `stopped_prefix` 是本轮前瞻评估使用的来源选项。省略该参数或指定 `--source-policy complete_only` 会运行原 complete-only 程序，便于复现。两者使用相同检测 core、参考和 eta。请在生成 CAL 前选择规则，不根据结果选择。新规则尚不能称为“更准”。

## 四分支同步诊断（.2412）

工作文档于 2026-09-25 更新；已有 .2412 发布归档保持不变。

在已审查的 hidden-v004 host 中，`fixed_public_environment` 冻结的是提供给 learner 的代理输入，不是真实 Z 或其演化规律。初始物理／潜在状态和潜在／物理噪声相同、行动按接口规定配对时，主路／固定公开输入、固定行动／双固定在共同执行日期上的物理结果和回报完全相同；完整配对回报差及四格 interaction 恒为零。这一结论依赖该 host 中 w 不直接进入物理方程、Z 不受行动影响，不能推广到任意 adapter。预测、Q、辅助量和支持仍可能随输入及拟合历史变化，但这种敏感性本身不是来源识别。

四条路径的结果仍需由 host 或模型提供。learner 不读取 Z，不等于反事实结果来自可观测主历史。在上述同动作 v004 耦合下，主路物理结果可用于主路／代理冻结的输入敏感性重放；改变行动的子路通常需要模型或额外真实结果。缺失不补零，本地有限 Q 不构成完整 eta 认证，当前四分支尚未连接有效的 hidden-v004 来源校准。拒绝和缺失属于完整诊断结果；仅筛选完成路径会改变比较总体，不能直接继承未筛选输出的信息或误差界。

四分支从同一公开初态和已拟合 learner 分叉，主分支先选本期行动，再由子分支执行；各分支只用自己的已释放结果更新 G/Q 与辅助量。

| 分支 | 当期执行行动 | 当期公开环境输入 |
| --- | --- | --- |
| `main` | 主 learner 在线选择 | 当期公开输入 |
| `fixed_action` | 分叉时主分支的首个行动 | 当期公开输入 |
| `fixed_public_environment` | 本期刚选出的主行动 | 分叉时公开输入 |
| `both_fixed` | 分叉时主分支的首个行动 | 分叉时公开输入 |

固定公开环境分支不自行 greedy，也不预读未来行动。固定公开输入不等于固定隐藏 Z；物理状态、真实上一期执行行动和学习历史仍逐期递推。`RND2BranchLearner` 使用 full RA-GQ；公开状态顺序为 `state,w,previous_action`，辅助量顺序为 `delta_gamma,delta_m,residual`。前缀辅助量按已释放历史重建，在线 residual 来自结果发布前实际发出的 base-G 预测。

安装后在分发根目录运行随包 RND2 工程示例，并保存同一请求：

```sh
.venv/bin/python -I examples/counterfactual/rnd2_four_paths.py --output cf_example.json --request cf_request.json
.venv/bin/python -I -m pinc_toolbox.counterfactual_cli --input cf_request.json --host-file examples/counterfactual/rnd2_environment.py --output cf_cli.json
```

输出文件须不存在。示例使用小型模拟器与工程参数；它不运行来源 CAL，不估计来源功效。真实应用需要提供可信的环境 host 文件及公开转移数据。CLI 的 JSON 输入包含 `history`、`learner`、`steps`、`environment_config`；可选 `stage1`、`context` 和 `existing_report` 是另行提供的报告信息。`learner` 声明 `as_of`、`warmup_as_of`、`window`、`path_id`、`refit_every`、`beta`、`horizon`、`nodes_per_memory`、`noise_nodes`，并非从完整未来样本自动选择。

MATLAB/Octave/Dynare 宿主脚本调用同一个 Python core：

```matlab
addpath('/absolute/extracted/examples/counterfactual');
[report, raw_json] = pinc_counterfactual( ...
  '/absolute/extracted/cf_request.json', ...
  '/absolute/extracted/examples/counterfactual/rnd2_environment.py', ...
  '/absolute/extracted/cf_dynare.json', ...
  '/absolute/extracted/.venv/bin/python');
```

Python 自定义 adapter 的入口为 `run_four_paths(learner=..., environment=..., initial_state=..., start_time=..., steps=..., beta=...)`。`PublicState` 分开记录 physical/auxiliary；`PublicInput` 记录输入值与 `available_at`；`PublicOutcome` 记录下一状态、reward 与 `release_time`。adapter 必须声明公开输入坐标，执行干预且保留内生状态；接口本身不能证明用户提供的变量确实可观测。

当前四分支 RND2 接口的行动资格为 `native_local_support_and_finite_Q`，`eta_status=not_supplied`；有限 Q 与本地支持不构成完整 eta 或经济 Q 精度认证。`trace.issued` 保留 Q/null、资格 mask、实际执行 Q、base/full 预测及其时点。

工作源码的 RND2 receipt 另提供可选字段 `base_physical_prediction` 与 `full_physical_prediction`，顺序均为 `[next_state, next_w]`；`w` 是算法可观测的代理变量，不是潜在状态真值。这两个字段在 `time` 发出，预测 `time+1` 的发布值，模型时点由原有 `base_fit_as_of` / `full_fit_as_of` 标记。首坐标分别等于旧字段 `base_state_prediction` / `full_state_prediction`；确定性的行动记忆不包含在向量内。旧 receipt 可以没有这些字段，消费者应按需读取；缺失时不能补零或推断 next-w。新增记录复用同一次预测，不增加拟合、Q 求解或随机抽样，也不改变行动、辅助残差或默认标量报警统计。向量检验需要匹配的向量校准，不能复用旧标量阈值。旧的已安装 2412 仍不含此变更；本地 .2501 wheel 已包含当前源码，旧运行环境不变。 已有 32 个真实保存快照通过旧 receipt 投影及向量预测完全一致检查，验证新增拟合、Q 求解和 DGP 步数均为零；详细版本差异与独立 H8 配对结果见下方共用附录说明。

查看 `report.counterfactual.branches` 中的逐期 trace、实际行动、公开输入、发布时点、状态与失败；`comparison.values` 保留四个有限期折现回报，`comparison.signed_differences` 保留四个条件差分及 interaction。终值贡献为零；即使行动 planner 用 infinite-Q，这也不是生命周期福利或 true regret。任何必需分支未完成时完整比较 unavailable，支持拒绝、软件错误与主行动不可用分别保留。对比可以有正负抵消，未知项不填零。

本接口不从四条路径自动产生 p 值或来源结论。已有两层报告可附带保存，但其假设、窗口和校准范围保持独立。旧 RND480 的来源成功来自候选解释族排除；旧 HS80 使用 observed Z。两者均不构成本次同步四分支或 hidden-Z HS 的识别功效证据。

## 可选分组组合来源接口（.2410）

新增 Python API `score_grouped_hs_source` 按实际报警后的完整 F/d/Q 路径分组，保留全部候选标签；max-prefix 与 terminal 各使用 .025，任一拒绝即排除。它逐组校准和流式保存证据，unknown 不作为兼容见证，CMIX 全部排除单列为候选族不足。Python CLI 与 MATLAB/Octave/Dynare wrapper 均可显式选择 `grouped_composite`，调用同一已安装控制器；原有两个选项与默认值保持原行为。接口、回调与限制见 [分组组合 API](docs/hs_grouped_composite/README.md)。独立 HS80 确认已完成：mixed 完整成功为12/20，点态95%区间为[.361,.809]；不能宣称已达到可靠高功效识别。

直接运行新的组合模式（固定工程示例，不是独立确认实验）：

```sh
.venv/bin/python -I examples/hs2022/run_example.py --config examples/hs2022/smoke.json --source-policy grouped_composite --case mixed --output hs_grouped
```

同一 MATLAB/Octave/Dynare 调用只需将第六个参数换成 `'grouped_composite'`。默认将配置 `source_alpha` 平分给 max 与 terminal；Details 显示两支预算和分数。现有 smoke 的 B=19 仅检验接线，最小秩 p=.05，因此不能在每支 .025 下拒绝。

## 输入、初始化与 eta

可以用实际观测替代示例生成的数据：

```sh
.venv/bin/python -I examples/hs2022/run_example.py --config examples/hs2022/smoke.json --source-policy stopped_prefix --case mixed --input-csv hs_change/observed.csv --output hs_csv
```

CSV 列为 `release,Y,Z`；release 从 0 开始连续，Y/Z 是有限的已观测水平。Z 在本例中是观测到的环境变量，不是隐藏状态或行动。`--case` 不修改 CSV 数值或强制报警。

配置中的 R 条旧转移用来拟合 G、残差变换 K 和选择 eta；仅用旧前缀，保留 127 个实际旧预测锚点。随后声明的检测区间使用当前观测。eta 按旧前缀与维持的基准高斯模型预测包络选取，支持信息及选取结果见 `report.json.context.reference`、`detector.json`。初始化失败显示 unavailable。

CSV 至少要覆盖旧前缀和完整检测区间。完整来源报告还须覆盖实际报警时点之后的 source_H 条转移；缺少来源观测时显示 unresolved，不补造数据。完整时钟、配置和输入要求见 [HS 示例](examples/hs2022/README.md)。

## 两层输出如何读

1. **检测层**报告声明区间内是否报警、首次报警时点和可用性。无报警不证明没有变化；无报警时来源层为 `not_entered`，不执行来源 CAL。
2. **来源层**只从实际报警状态开始，使用完整候选 F/d/协方差及新条件 CAL。它分别检查环境类 CL、增长机制类 CM 和包含两类的 CMIX。排除 CL 可支持增长机制必要，排除 CM 可支持环境机制必要；两类均排除且联合类有兼容见证时才报告二者必要。未分离、数值未决和候选类不足均单独保留。

“必要”以维持的候选族为条件，不是“仅有这种变化”；兼容见证不证明真实机制属于候选族。来源报告针对声明的报警后窗口；没有时序/持续性前提时，`trigger_attribution` 保持 unavailable，不能把窗口内的变化自动写成早先报警的原因。

`stopped_prefix` 保留非空、有限的 CAL 支持前缀分数，分别标记“窗口完整”和“分数可排名”。支持退出不再计算后续残差。空前缀和数值失败按正无穷占据原 CAL 槽位，不删除、不重抽。观测窗口不完整时始终 p=1/unresolved，不能作为兼容见证。`complete_only` 则把所有不完整 CAL 都按正无穷处理。

## Details、完整证据与统计限制

CAL 失败槽给出的 `calibration_failure_p_floor` 只是 p 值下界。`calibration_allows_rejection=false` 表示该下界已阻止拒绝；`true` 只表示失败槽数没有阻止拒绝，不保证统计量能达到拒绝区间，也不表示功效或来源概率。第一层仅在明确提供 `cal_total_B`、`cal_failed_count` 和 `alpha` 时计算；缺失时不猜值。这些报告字段不改变报警、来源路由或 η 的适用条件。

- `report.html`：轻量摘要、可展开的错误范围/假设/窗口，以及候选排名表。未知显示 unavailable，不显示为零。
- `report.json`：明确标记为摘要；显示观测完整性、观测分数可排名性、CAL 完整数与可排名数、原 B 和失败导致的 p 值下限。
- `report.json.gz`：完整、无损的候选输出、逐路径 flags、CAL 分数、F/d/协方差和控制器报告。每个候选只写一次；中断文件保留 `.partial`。
- `observed.csv`、`source.csv`、`detector_arrays.npz`、`detector.json`：保留输入、实际来源窗口、检测数组和校准信息。

名义 alpha、已证明的界、经验频率和后验概率不同。可配置示例的证明界默认为 null；正确模拟器、条件交换性、完整候选族包含真实规律等前提须另行核实。检测误报范围是整个声明的基准无变化日历；来源错误范围条件于实际报警，不能用无条件来源界替代。Type II、功效和连续候选族保证尚未自动建立。

## MATLAB / Octave / Dynare

安装上述 wheel 后，在 MATLAB/Octave 或 Dynare 的宿主脚本中调用同一个 Python runner：

```matlab
addpath('/absolute/extracted/examples/hs2022');
[report, raw_json] = pinc_hs_example( ...
  '/absolute/extracted/examples/hs2022/smoke.json', 'mixed', ...
  '/absolute/new-hs-output', '/absolute/extracted/.venv/bin/python', ...
  '', 'stopped_prefix');
```

第五个参数可换为 CSV 绝对路径；第六个参数显式选择来源规则。`raw_json` 保留 JSON null，避免 Octave 解码后重编码成空数组。适配器不实现第二套估计器；Details 是本地 HTML，不是 Dynare 内置界面。其他 Dynare/GQ/eta 接口见 [接口与历史说明](README_HISTORY_PRE2409.md)，相应 `.m` 文件随包保留。

## 复现与资料

- [完整 HS driver、正式参数模板与来源](examples/hs2022/README.md)。H128/H1024 模板不等于已经运行的独立确认。
- [历史版本与接口说明](README_HISTORY_PRE2409.md)：原文保留，版本、旧结果和验证范围不改写为当前结果。
- [理论附录](docs/two_stage_diagnosis/appendix_two_stage_diagnosis.pdf)与下方同步说明。一般保证仍以实际满足的假设为限。


Current final results and execution limits: [HS80, RND480 and separate boundary sample](docs/final_confirmation/README.md). Earlier balanced16 results below belong to the prior stopped-prefix rule and are not pooled with HS80.

## .2411 原生闭环实现与工程验证记录

The base RND2 workflow now has one installed entry connecting released initialization, actual eta selection, native H10/infinite behavior, live detection, acknowledged supported acquisition, released source observations and two-stage JSON/HTML Details. See [native host loop guide](examples/native_source_loop/README.md).

```sh
.venv/bin/python -I -m pinc_toolbox.native_source_loop_cli --input examples/native_source_loop/request.json --host-file examples/native_source_loop/simulation_host.py --output native-loop.json --html-output native-loop.html
```

The bundled host is an explicit small simulator; an application supplies its real execution/release adapter. This completes the minimum base loop, not autonomous RA auxiliary learning or an arbitrary Dynare model adapter. The `.m` wrapper calls the same installed CLI. Thirteen fixed engineering fixtures, eleven targeted negative checks, original-driver diagnostic mapping and actual Octave/Dynare parity were checked. All 60 earlier package modules are byte-identical. These checks do not supply new scientific power evidence or validate the scientific choice of HS score; existing HS80/RND480 limits remain. See `validation/implementation_2411/FINAL_VERIFICATION.json` for the two-build audit trail.



## Historical scientific validation: earlier HS balanced16

Four independent paired prefix clusters give n=4 per case; the 16 paired histories are not 16 independent replications. Null alarms: 0/4. All 12 alternatives alarmed after change. Full predeclared primary success requires a postchange alarm, all observed candidate computations complete, the true candidate retained and exactly the requested source conclusion: growth/model 3/4, environment 4/4, mixed 0/4. All four mixed reports were environment-required only; usable two-source attribution remains unmet.

One growth-only path rejected its true candidate at p=.042 while reporting ambiguous. This candidate false rejection is distinct from zero false asserted-family conclusions and must not be described as a zero-error batch. Z is observed. Source conclusions concern the future 1024-transition window and finite 268-candidate catalogue, not the cause of an earlier alarm. These small descriptive results establish neither precise high power nor posterior probability, and do not close the full middle loop. RND480 is complete; see the final confirmation summary.

See the [report](docs/hs_balanced16/REPORT.md), [figure caption](docs/hs_balanced16/FIGURE_CAPTION.md), [T2 final review](docs/hs_balanced16/HS_BALANCED16_FINAL_REVIEW.md) and [registered EvidenceMatrix record ECMA-20260924-15](docs/hs_balanced16/EVIDENCE_MATRIX_POINTER.md).

<!-- TWO_STAGE_APPENDIX_START -->
## Two-stage diagnosis and statistical details

[Appendix: notation, landscape decision tree, and error-rate definitions](docs/two_stage_diagnosis/appendix_two_stage_diagnosis.pdf)

The aim is reliable switching-source discrimination where the permitted information identifies the candidates: prediction-error statistics test a specified maintained law under matched calibration; supported counterfactual measurements and matched candidate tests then delimit source conclusions. Audited detection gains, support limits and fallback execution are established in their stated experiments; useful source power for the current native candidate family remains unestablished. Identification boundaries and working software do not replace that objective.

- **Stage 1 — Detection:** report the window, horizon, entry rule, specified maintained null (including what no change means), statistic and calibrated decision when available. A p-value measures extremeness under that null and selection rule; neither p nor the level is a posterior switch probability. Rejection alone does not identify a source; uncalibrated outputs remain descriptive. No alarm does not establish no change.
- **Stage 2 — Sources:** under applicable matched inference, report the declared candidates, rejected/retained sets and unresolved status alongside component-necessity labels. The retained set is a frequentist compatibility set, not posterior probabilities. True-candidate coverage does not ensure a singleton or exclusion of all false candidates; an empty set does not identify an unnamed source. Fixed-date results do not automatically carry alarm-conditional guarantees.
- **Joint result:** preserve detection, primary/secondary route, original support/refusal status, missing values and candidate conclusions separately. A fallback does not fill the original missing contrast. Availability is not accuracy. End-to-end evaluation retains all assigned histories; alarm-conditioned rates use a separate denominator and paired branches are not independent replications.
- **Reliability:** identify the actual null/error event, nominal level, conditioning event, assumptions, deadline and random-calibration or fixed-bank scope. Separate proved bounds from empirical errors/power, with assigned/completed counts and uncertainty. Candidate false exclusion means excluding the true declared candidate; failure to exclude a false candidate and power require a named alternative. Direct-entry conditioning does not automatically match alarm-and-availability selection, and random-bank bounds do not certify the realized bank. These reporting requirements do not claim new runtime fields or completed calibration.
- **Notation:** S means a set of diagnostic implications, not switching. S_L allows latent changes with the model unchanged; S_M allows model changes with the latent law unchanged; S_MIX allows both, including pure sources. C is the common uncertainty region.
- **Details (implemented local HTML):** the optional disclosure displays saved error definitions, budgets, applicable bounds and empirical evidence without rerunning inference. For component reports, Type I is a false necessity declaration and Type II is failure to complete the requested report under a specified alternative; for candidate tests, state the actual candidate-exclusion error. Mark unavailable quantities unknown. The reporting specification above does not imply that every field or guarantee is implemented; this is not a Dynare-internal GUI.
- **Numerical meaning:** nominal budgets, proved upper bounds and empirical estimates are separate. Unknown values are `not established` / `not estimated`, not zero. Error rates are not single-event source probabilities.
- **Historical implementation evidence (.2409/.2410):** Development wheel `0.2.0.dev2026092410` retains the two-stage controller, scoped reset and predictable SimpleGQ routes, learned-eta domain adapters, post-alarm supported-action recommendations, and optional HTML Details. The selected domain restricts current native action queries and diagnostic weights; it does not certify Bellman continuation values or e_eta. Version .2410 adds an optional exact-conditional-law grouped HS source API with maximum and terminal score branches, each at .025, preserving every candidate label and family membership. Installed tests and all 180 existing development banks reproduce the audited scores, ranks and family reports exactly; the fixed independent HS confirmation is now complete: 20 independent prefixes with four paired cases, with full protocol success of 19/20 null, 12/20 mixed, 19/20 growth-only and 20/20 environment-only. These casewise rates have wide intervals and do not establish uniform source power. An explicit stopped-prefix HS source option distinguishes full-window completion from a valid finite calibration score. Empty or undefined calibration slots remain extreme; incomplete observations remain unresolved. Installed tests and saved-data parity verify implementation, not source power; prospective confirmation is separate. The original complete-only option and its evidence remain available. Version .2409 was freshly installed outside the research checkout and tested with Python -I, actual Dynare 6.4 / Octave 9.4.0, the included linear-model core example, and the HS CSV wrapper. Python and host-adapter outputs agree in these engineering checks. The .2410 grouped-composite option is also exposed by the portable HS command and Dynare-facing wrapper; a fresh isolated Python installation and actual Octave 9.4.0 / Dynare 6.4 execution verified matching report and complete-evidence outputs for the same CSV example. Separate fixtures checked incomplete-source and partial-unknown cases. These small engineering examples do not estimate source power. Historical .2405 validation remains a separate record; none of these runtime checks establishes scientific power. MATLAB runtime and a Dynare-internal GUI remain unvalidated or unimplemented. Conditional-on-alarm bounds apply only to the proved reset, predictable-reference or correct conditional-simulator constructions under their stated assumptions; no generic bound or posterior source probability follows from caller-supplied arrays. A compatible family witness cannot verify true-family membership. Statistical-radius arithmetic qualification remains explicit.


### Four synchronous path diagnostics

The current `pinc_toolbox.counterfactual.run_four_paths` interface starts from one public state and fitted learner. It records four paths with separate state, learning history and auxiliary memory:

| Branch | Executed action | Public environment input |
| --- | --- | --- |
| `main` | Main learner's current online choice | Current released public input |
| `fixed_action` | Main action chosen at the first decision | Current released public input |
| `fixed_public_environment` | Main action just chosen at this date | Public input at the first decision |
| `both_fixed` | Main action chosen at the first decision | Public input at the first decision |

The public-environment branch follows the current main action, not its own greedy policy. Inputs carry availability dates, and next-period outcomes are released before the next learner update. Freezing an input does not freeze endogenous state, learning or actual previous-action memory. An adapter declares public coordinates and their intervention; a hidden `Z` is not made observable by this API. The RND2 adapter is a model-specific implementation, not an automatic adapter for an arbitrary Dynare model.

Working documentation updated 25 September 2026; the archived .2412 release is unchanged.

In the reviewed hidden-v004 host, `fixed_public_environment` freezes the supplied proxy input, not the true latent state Z or its transition law. With the same initial physical/latent state, identical latent and physical innovation tapes, and the prescribed common actions, the main and fixed-public-input paths have equal physical outcomes and rewards, as do the fixed-action and both-fixed paths, at every common executed date: w is not a primitive in this host's physical transition and Z is action-independent. Their complete paired return differences and the complete four-cell interaction are therefore zero at every admissible source amplitude. This is specific to that host/coupling, not a claim about every adapter. Predictions, Q, auxiliaries and support can still respond to the changed input/history; those diagnostics do not by themselves identify the source.

The four paths require a host or model to supply branch outcomes. Keeping Z out of learner inputs does not make simulator-generated counterfactual outcomes observations from the main history. Under the stated same-action v004 coupling, main physical releases can be replayed for comparison with the fixed-public-input path; changed-action branches generally require a model or additional actual outcomes. Such fitted-model rollouts remain conditional on their model assumptions.

This full-RA route reports `native_local_support_and_finite_Q` with `eta_status=not_supplied`. Neither full eta/economic-Q coverage nor a valid hidden-v004 source calibration is connected to these four-branch comparisons. A missing or refused branch is unavailable, never zero or evidence excluding a source; a complete two-branch readback and the API's all-four comparison have different completion requirements. Refusals and missing paths are part of the diagnostic output. Conditioning on completion changes the comparison population; information or error bounds for the unselected output do not automatically transfer.

The output keeps four discounted finite-path returns, each with zero terminal contribution, and signed action/environment contrasts. For values `(main, fixed_action, fixed_public_environment, both_fixed) = (V11,V01,V10,V00)`, interaction is `V11 - V01 - V10 + V00`. The identity `V11 - V00 = (V01 - V00) + (V10 - V00) + interaction` retains cancellation. These are intervention-specific path comparisons, not a true regret estimate or a source test. Infinite-horizon planning does not turn the reported finite sum into a lifetime value.


**Native four-branch detection pilot (25 September 2026).** A separate installed `.2412` full RA-GQ experiment used H10 planning and an eight-release detection window. Sixteen independent calibration banks each supplied four assigned evaluations per condition. The original residual-window statistic and nineteen-slot, at-most-three-attempt calibration procedure were used without changing support rules. Counts below retain all 64 assigned evaluations per condition.

| DGP condition | Amplitude | Alarms / assigned evaluations | Complete four-branch episodes / assigned evaluations |
| --- | ---: | ---: | ---: |
| Unchanged | — | 6/64 | 28/64 |
| Latent change | 0.5 | 3/64 | 37/64 |
| Latent change | 1.0 | 3/64 | 29/64 |
| Model change | 0.5 | 18/64 | 31/64 |
| Model change | 1.0 | 40/64 | 42/64 |
| Mixed change | 0.5 | 16/64 | 37/64 |
| Mixed change | 1.0 | 42/64 | 30/64 |

All 448 evaluation main windows completed, so the low latent-change detection in this pilot is not a main-path support refusal. Child support refusals still limit subsequent comparisons. Two calibration refusals were retained and handled within the predeclared capped slots; no slot exhausted. The 64 evaluations in a condition share sixteen banks and must not be treated as 64 independent calibration replications. Bank-level uncertainty is wide; these results neither validate the nominal false-alarm rate nor establish a precise amplitude frontier. They are detection measurements, not source-classification probabilities, eta qualification or economic-Q coverage. The separate fixed-bank feasibility experiment is now complete; its results are reported below and are not pooled with these sixteen-bank counts. Research receipts: `experiments/ecma_four_tracks_20260924/track3_identification/native_power_frontier_attempt001/execution001/{RESULT.json,PARENT_BATCH_AUDIT.json,SUMMARY.json}` and `track4_confirmation/native_power_frontier_review/FINAL_REVIEW.md` under the same experiment root.

**Completed fixed-bank feasibility follow-up.** Using the same predesignated bank 0, a separate experiment assigned 100 fresh episodes each to unchanged, latent-change, model-change and mixed-change conditions at amplitude 1. Main alarms were 18/100, 18/100, 87/100 and 82/100; all 400 main windows completed. All-four completion was 59/100, 47/100, 60/100 and 54/100. Sixteen predeclared joint events (main alarm AND the specified branch score available) received simultaneous one-sided 95% exact-binomial lower bounds, each using tail .05/16. For the unchanged condition's main event, the lower bound is .08924, above .05: at this confidence level, the fixed bank exceeds 5% null alarm probability and cannot carry the claimed conditional 5% guarantee. This does not contradict a guarantee averaged over newly drawn random banks. These are feasibility measurements, not source-classification accuracy, alarm-conditional source calibration or eta qualification. The complete 16-event table and independent reviews are recorded in `track1_production/source_feasibility_bank 0_attempt001/execution001/RESULT.json`, `track2_errors/SOURCE_FEASIBILITY_ACTUAL_SCOPE_REVIEW.md` and `track4_confirmation/native_power_frontier_review/SOURCE_FEASIBILITY_FINAL_REVIEW.md`, under `experiments/ecma_four_tracks_20260924/`.

**Training requirements before extending detection.** The estimation window selects released rows for each refit; detection horizon H counts issued forecast errors. Each required full-RA fit needs enough retained rows for its declared columns, full joint column rank, and the existing response-identification and numerical checks. A longer detection horizon does not supply those training conditions. With three active auxiliaries the current dictionary has 20 joint columns; the recorded window-16 stop at date 40 had only 16 rows after the panel prefix expired (Evidence49). Increasing row count alone does not ensure identification.

For the current anchored quadratic action dictionary, at most two distinct **actually executed actions in the retained training window** force exact column dependence, regardless of sample size. At least three distinct actions are necessary, not sufficient: state-action coverage, other dependencies and numerical conditioning still matter. The nominal action grid, actions considered by Q, or exploratory prefix actions that have left the window do not meet this requirement. Greedy adaptation need not provide the required variation. If `IdentificationError` stops a refit, inspect the retained training supply, actual action coverage and design rank/conditioning before specifying a continuation. This is not a no-switch result, a support refusal, or a probability that the source is unknown. No automatic new diagnostic output is claimed here; defaults, source code and released archives are unchanged. The four-history engineering pilot supplies no power result. Algebra and scope: `track2_errors/POST_PREFIX_FOUR_HISTORY_PILOT_SCOPE_REVIEW.md` under `experiments/ecma_four_tracks_20260924/`.

**Optional observed-vector receipts: working-source scope.** The working Python RND2 adapter adds optional `base_physical_prediction` and `full_physical_prediction`, each ordered `[next_state, next_w]`. The first coordinate exactly retains the existing scalar forecast; w is an observed proxy, not latent Z. Predictions are issued before the next release, exclude deterministic action memory, reuse the existing prediction calls and leave actions, fits, Q, auxiliaries and the default scalar statistic unchanged. All 32 saved pre-release snapshots passed old-receipt projection and vector parity checks with zero fits, Q solves or DGP steps. Historical receipts may omit these fields; absence is not zero. The checked installed .2412 module lacks the fields and remains unchanged. The local .2501 wheel includes the current adapter source. The Dynare/MATLAB/Octave wrapper delegates to its explicitly selected Python core: neither this README nor the shared appendix upgrades that runtime. The current Dynare working files have no implementation of these added receipts, and no new Dynare execution has validated them. Evidence: `track1_production/vector_receipt_integration_attempt001/SAVED_CHECK.json` under `experiments/ecma_four_tracks_20260924/`.

**Completed matched scalar/vector H8 experiment (Evidence ECMA-20260925-48).** One shared set of 59 independent null CAL histories supplied separate scalar and vector strict-max thresholds. The vector score used observed (x,w), unit weights and identity weighting; it did not reuse the scalar threshold. Each condition received 32 fresh EVAL histories, with one scoring look at release 32. The experiment ran a copy of the working Python source in the .2412 interpreter, not an updated installed package or a Dynare-native execution.

| Condition | Available / assigned | Scalar alarms / assigned | Vector alarms / assigned |
| --- | --- | --- | --- |
| Unchanged | 32/32 | 0/32 | 0/32 |
| Latent change, strength 0.5 | 31/32 | 0/32 | 0/32 |
| Latent change, strength 1 | 32/32 | 1/32 | 0/32 |

The weak-change refusal remains in the denominator and produces no alarm. Both predeclared one-sided paired McNemar improvement tests have p=1, including Holm adjustment. No vector improvement was demonstrated; this is not equivalence, general scalar superiority or evidence that the proxy lacks information. Six simultaneous 95% Clopper-Pearson intervals conditional on this bank are [0, .15740593] for every 0/32 count and [.00013047, .21536903] for 1/32. N0/32 does not establish 5% fixed-bank size. Under matching-law assumptions, the B59 tolerance result gives CAL confidence at least .95150547 per detector; the union lower bound for both is only .90301095, not simultaneous 95%. The alarms are not OR-combined and no winner is selected. The 155 executed histories used 5115 G fits, 1240 Q solves, 1240 reward fits and 120279 transitions. This experiment establishes no source-classification success, eta qualification, bank-averaged superiority or new long-run/anytime monitoring guarantee. Later exploratory lag statistics are not part of this confirmation. Results and paired uncertainty: `track3_identification/vector_scalar_evidence_attempt001/PAIRED_RESULTS.md`; final audits: `track2_errors/VECTOR_SCALAR_FINAL_STATISTICAL_REVIEW.md` and `track4_confirmation/vector_scalar_strictmax_review/FINAL_REVIEW.md`, under `experiments/ecma_four_tracks_20260924/`. The corresponding record is ECMA-20260925-48 in `note/UNIFIED_THEORY_GQ_DGP_EVIDENCE_MATRIX.md` in the research project.

`branches` retains each trace, completion state, failure and partial return. `comparison` is unavailable if any required branch is incomplete; an unknown cell is never zero. Support refusal, software failure and missing current main action are separate outcomes. `counterfactual_report` attaches the result to the existing two-stage report; any separately supplied inference keeps its own scope. Counterfactual p-values and source probabilities remain unavailable. Historical HS80 (observed Z) and RND480 scientific results below do not validate this new interface or a hidden-Z HS application.

### Optional support-only action-pair routing (.2501 local validation package)

The Python and Dynare entries share one reporting implementation. Routing is off by default: existing reports and the five original contrasts are unchanged. For a host whose frozen public input changes only learner queries, callers may explicitly declare the model-specific applicability:

```python
report = counterfactual_report(result, pair_routing={
    "applicability": "input_only_matched_physical_paths"
})
routed = report["counterfactual"]["routed_action_diagnostic"]
```

This declaration is **not verified by the reporting function**. Equality of the two action contrasts requires corresponding branches to share physical/private initial states and dated innovations; the main-action tape and companion actions must match; fixed-action/both-fixed use the same first action; conditioned input must not enter physical transitions, rewards or latent evolution; and both returns must use the same horizon and discount. The current v004 input-only host satisfies the stated action-independent latent premises. A generic environment where the public input physically changes outcomes need not satisfy them. Branch names, runtime shape checks or setting this option do not prove equivalence. In such an environment the two expressions can be different measurement objects; do not declare this applicability merely to obtain another value.

`selected_branch` is primary when main and fixed_action complete. Secondary is used only after primary SupportError-type unavailability and completion of fixed_public_environment and both_fixed. Identification failures, mixed identification/support failures, malformed horizons, nonfinite returns/differences and software faults do not trigger fallback. `reason`, `value`, both pair statuses/values/failures and the original all-four status remain visible. Zero is a valid value; false/missing is not zero. An independently verified identification receipt may be passed as `verified_prepare_failures` with the exact branch/date/error type/reason and classification; the reporting helper checks correspondence, not the scientific validity of the caller's classification. Without such evidence a raw software failure stays invalid. Primary nulls and original all-four-unavailable records are never filled or rewritten.

For an already saved result, create request JSON with `counterfactual_result` and optionally `pair_routing`, `stage1`, `context` or `existing_report`. Then use the existing CLI's explicit reporting mode:

```sh
python -I -m pinc_toolbox.counterfactual_cli --input saved_request.json --report-only --output routed_report.json
```

The existing Dynare/Octave entry selects that mode with an **empty host_file**:

```matlab
[report, raw_json] = pinc_counterfactual(request_path, '', output_path, python_executable);
```

Report-only mode loads no host, reconstructs no learner and performs no fitting or DGP steps. A nonempty host_file retains the existing trusted-host execution behavior. Live JSON requests pass the same optional `pair_routing` configuration to the shared report function. MATLAB contains no routing arithmetic. These additions are included in the local .2501 wheel and both local distributions. The earlier .2412 archives and existing runtime remain unchanged. Source-checkout verification preceded the separate isolated-wheel installation checks recorded with this local package; no external publication is implied.


The commands above require this local .2501 wheel installed into the selected Python environment, or an explicit source-checkout launcher. The older .2412 installed runtime was not updated and does not support the new `--report-only` option. Merely changing directory into the checkout does not expose it under Python `-I`. For a zero-install Python source-checkout invocation, replace the absolute source path below:

```sh
python -I -c 'import runpy,sys; sys.path.insert(0,"/absolute/checkout/software/pinc_toolbox/src"); sys.argv=["pinc_toolbox.counterfactual_cli",*sys.argv[1:]]; runpy.run_module("pinc_toolbox.counterfactual_cli",run_name="__main__")' --input saved_request.json --report-only --output routed_report.json
```

For Octave, `python_executable` must likewise resolve these updated modules: either an environment containing them or an explicit executable source launcher that forwards the wrapper arguments to Python after adding that exact source directory. The earlier source parity check used the latter. The .2501 local-package verification separately uses the newly installed wheel under Python `-I`, without a source bootstrap; it does not update the old runtime.

`routed_action_diagnostic.inference.status` remains `unavailable`. The fixed-eight receipt established seven primary selections and one support-only secondary selection, not source accuracy or a calibration bank. Any future rank result must match the complete observation/candidate law, including entry event and this routing/completion rule. Equality on overlap alone does not validate calibration from a primary-only bank.


### Independent confirmation and its limits

The earlier RND2 source conclusions came from testing and excluding candidate explanation families CL/CM/CMIX. They were not obtained by validating the new parallel fixed-action/fixed-public-input paths. Research prototypes of path comparisons and the installed synchronous API are distinct implementations with distinct evidence. No RND480 success rate transfers to the new counterfactual API; hidden-Z HS source separation remains unestablished.

The final selected-eta/native RND2 experiment assigned 60 independent histories to each of eight cells. Strict protocol successes were 60, 60, 57, 56 for H10 null/latent/model/mixed and 60, 59, 57, 55 for infinite-Q null/latent/model/mixed. All 360 alternative histories alarmed; all 120 null histories did not. Eight support failures and eight completed mixed reports missing the latent component remain failures in the original denominators. Pointwise and eight-cell simultaneous 95% intervals are reported separately; 60/60 has respective lower bounds .9404 and .9083, not a 99% reliability guarantee. The actual run used a source-tree worker; archived module comparisons establish continuity with the current package, not an installed-wheel execution of all 480 histories. The full scientific driver remains in `experiments/ecma_four_tracks_20260924/track4_confirmation/` in the research project; portable examples are interface demonstrations, not that full replication.

This RND2 procedure uses known Gaussian noise, a full linear dictionary, two fixed monitoring looks after a preset onset, and a 512-observation post-alarm source window. The 720-row prechange collection allows selection among 180/360/720-row prefixes; selected prefix length is not a proved minimum collection budget. Finite-task eta readiness does not cover every future action or Bellman continuation. The matched boundary sample has 40 trajectories and 48 structural views, including weak-signal, rank-deficient, equivalent and outside-family examples. Its four-history cells are descriptive and are not pooled with the 480 histories.

HS independent confirmation uses observed Z, prechange eta and the actual first alarm to start a 1024-transition source window. Its 20 paired prefixes produced one null false alarm, mixed alarms in 20/20 histories, growth-only alarms in 19/20 and environment-only alarms in 20/20. Eight mixed histories retained a pure-environment explanation despite complete source calculations. This is incomplete separation, not proof of pure-environment truth. The independently checked full-path information bound does not explain these eight failures or establish impossibility. Neither experiment reports posterior source probabilities, latent-state recovery, the cause of an earlier alarm without persistence assumptions, economic-Q accuracy or welfare improvement.

Scientific reports: `track4_confirmation/ETA_480_REPORT.md` and `track3_identification/hs_grouped_confirmation80_attempt001/REPORT.md`, both under `experiments/ecma_four_tracks_20260924/` in the research project. Error details distinguish nominal budgets, applicable proved bounds, observed errors and unknown cases. The HS per-case confidence intervals are pointwise with 20 independent prefixes, not an 80-independent-history interval.

The manuscript and both package documentation copies are maintained from one source using `manuscript/figures/two_stage_diagnosis/sync_appendix.py` in the PINC project. Changes to the tree, notation or statistical interpretation must be rebuilt and copied together.

**Historical observed-Z HS method scope:** The HS80 source procedure evaluated above uses a residual moment weighted by the observed environment state. It does not use an action choice or a Q-function. Its conditional calibration argument controls candidate rejection under the stated simulator assumptions; it does not establish that this particular residual summary is sufficient or optimal for source separation. The experiment therefore validates the stated residual diagnostic, rather than an end-to-end application of the GQ/RA-GQ action-comparison procedure. That historical experiment does not validate the later hidden-Z native action workflow or the new four-path interface; those require their own evidence. The HS statistic is not an economic-Q error or a posterior source probability.

<!-- TWO_STAGE_APPENDIX_END -->


## Inspect native Q support without changing the policy

On an already fitted `RND2GQ` snapshot, Python callers can inspect the same
query used by `q_values`:

```python
detail = model.explain_q_values(states, auxiliary, feasible=public_mask)
q = detail["q_values"]  # same ndarray values as model.q_values(...)
first = detail["rows"][0]
print(first["same_memory_count"], first["nearest_node_index"])
print(first["actions"][0]["all_blockers"])
```

Arguments and validation match `q_values`, including `remaining_horizon` for
an existing finite-horizon table. Each row records matching-memory count,
selected node index and scaled distance. Each action records current physical
support, public feasibility, selected-node action support, table finiteness,
eligibility and all refusal reasons. Missing-memory node/table fields are
`None`. Unsupported Q cells remain `-inf`; per-action `q_value` and
`table_value` use `None` when unavailable/nonfinite, with `table_nonfinite`
retaining the nonfinite label. The top-level Q ndarray is not strict JSON.

This read-only explanation performs no fit or Bellman solve and does not
expand support or change `act`. Finite Q does not establish an eta guarantee.
The native JSON CLI also accepts a per-query `detail` boolean. Omission or
`false` preserves the previous response; `true` adds `detail` to that query's
output, containing the corresponding `rows[0]` above. Values such as `1`,
`"true"` or `null` are rejected. The CLI detail object contains no Q ndarray;
nonfinite table entries use `null` and a string label, so the response remains
strict JSON.

For an existing request, enable the field before running the CLI:

```python
request["queries"][0]["detail"] = True
# Save request as request_with_detail.json using json.dump.
```

```sh
python -m pinc_toolbox.native_gq_cli --input request_with_detail.json --output new_result.json
```

The existing Dynare/MATLAB bridge forwards the same JSON file unchanged:

```matlab
[result, raw_json] = pinc_native_gq('request_with_detail.json', '/absolute/path/to/python');
% Each completed event's snapshot.queries contains detail for enabled queries.
```

The selected Python environment must contain this updated source. The bundled .2601 wheel includes this addition;
older .2501 wheels do not. Install the new wheel into a separate environment. The query explanation itself performs no fit or
solve, while the CLI still executes the request's explicit fit/update events.

### Optional Bellman evaluation at the queried state (working source)

`RND2GQ.q_values(..., evaluation="bellman")` and `RND2GQ.act(..., evaluation="bellman")` evaluate the fitted reward and Bellman continuation at the actual query. Finite-horizon evaluation uses the saved one-period-shorter Q table; infinite-horizon evaluation uses the saved infinite-horizon table. The original destination nodes, transition quadrature, action-memory groups, and physical/public support remain in use. The default `evaluation="nearest"` retains the previous lookup.

This opt-in rule changes off-grid evaluation and can change actions and subsequent data. Monitoring, probes and calibration must therefore use a matching rule; previous calibration results cannot be transferred automatically. Finite output is neither an eta guarantee nor evidence of source identification. This working-source option is not included in existing package014 archives.

Four fresh matched conditional-history replications (Evidence ECMA-20260926-74) retained `{CM, MIX}` twice, `{N, CL}` once, and an empty candidate set once. Here `N` denotes no change, `CL` latent-only change, `CM` model-only change, and `MIX` both. These outputs identify the model component in three cases within the declared candidate catalogue, but leave the latent component unresolved in all four. An empty set requires an unresolved report; it is not evidence of no change. Four replications do not establish an accuracy or power rate. The saved-result report displays these candidate sets and rank p-values; it must not convert them into source probabilities.

The central claim concerns model involvement under the stated finite-catalogue and correct-conditional-kernel assumptions. Latent involvement is claimed only on an established identifiable range; unresolved outcomes remain admissible. The true-law exclusion bound controls erroneous present or absent assertions conditional on an alarm, averaging over source calibration and the observed future. It does not control the error conditional on a resolved report, or failures to report presence that include abstention. We therefore report erroneous assertions and resolution frequency separately. Larger latent shifts are not assumed to improve identification automatically.


### Experimental fixed-entry model-involvement report

The working Python source now exposes `pinc_toolbox.fixed_entry_model_involvement.fixed_entry_model_involvement`. It consumes ordered N and CL calibration proposal records and one observed paired-residual record. Each bank uses its first 39 complete, finite records within 128 proposals. Both upper-tail ranks must reject at 0.05 to return `model_involvement`; every other result is `unresolved`, with support failures, incomplete calibration and software errors distinguished. Non-rejection does not establish model absence. Candidate ranks are not source probabilities.

This interface reproduces all 32 decisions in the fixed-entry confirmation: model assertions N 0/8, CL 0/8, CM 7/8 and MIX 5/8, with all remaining paths unresolved. Those are one batch's counts, not population accuracy. Its assumptions concern a finite candidate catalogue and matched completion rules; it supplies neither an alarm-conditional guarantee nor eta coverage. It does not replace the existing two-stage interface. This addition is in working source only: existing released Python/Dynare archives have not been rebuilt to include it.

Evidence 81 and 82 are distinct first-union conditional outcomes: the unsigned test selected a true MIX history but remained unresolved (N p=.60, CL p=.55); the later single signed two-sided pilot asserted model involvement for its selected MIX history (N and CL p=.05 each), with latent involvement unresolved. This one prospective pilot does not establish power, a source probability, two-stage operating guarantees, or latent-frontier closure, and it does not overwrite Evidence 81. Evidence83 now reports three preassigned H10/direct-Bellman first-alarm trials (fixed remaining_horizon=10): one selected true MIX and two true CM histories, each with two-sided p=.05 for N and CL and latent unresolved. Seven of 10 reached monitors were nonalarms, all reached monitors were available, and six CAL support refusals per candidate family remain in the 123-proposal denominator. The original run02 free-space interruption is retained; the exact same allocation/seeds completed on one technical retry, not an added scientific replicate. The three-trial batch is not accuracy, power, or population error control; scan entries after first alarms were unmonitored. Evidence81’s unsigned result remains distinct.

Experimental alarm-conditional model-involvement rank helper: `pinc_toolbox.alarm_conditional_model_involvement.alarm_conditional_model_involvement` takes an explicit `no_alarm`/`not_run`/`support_unavailable`/`triggered` state and ordered N/CL signed-score records; after a trigger it selects the first39 complete finite scores within128 proposals, preserving support failures and other errors. It reports tie-inclusive lower/upper and two-sided ranks and returns only `model_involvement` or `unresolved`; latent remains unresolved, assumptions are unverified, eta and posterior probability are not supplied, and no power or global-monitoring guarantee is established. This is a deterministic rank-reporting helper, not a conditional simulator, and it does not change the production default.

The working-source MATLAB/Octave wrapper `pinc_alarm_conditional_model_involvement(payload, python_executable)` now calls that same reporter through its JSON CLI. Prefer an absolute JSON input file (containing the Python function keyword arguments) to preserve array shapes; the second return value is the unchanged response JSON. The Python executable must contain this source revision and its dependencies. Extreme-rank, ties, no-alarm and unsupported-observation transport cases passed exact Python/Octave parity. This is not a new package release or an automatic conditional-calibration simulator; existing distributed archives are unchanged.


### Current experimental source-window evidence

The source observation window is separate from the Q planning horizon. Evidence88 uses one post-alarm date with the same H10 direct-Bellman Q, the paired-union detector, and candidate futures conditioned on every acquired probe outcome. Its 16 fresh histories include four model-only and four mixed histories, all reporting model involvement; neither of the four no-change nor four latent-only histories reports model involvement. One no-change history lacks monitor support, and one no-change and one latent-only history alarm but remain unresolved. Every assignment is retained. Four histories per family do not establish high reliability.

Evidence87 retains the separate eight-date attribution experiment: all eight model-involved histories alarmed but source inference remained unresolved. Different evaluation samples prevent a paired causal comparison of window lengths. The current inference excludes only the specified N/CL no-model catalogue; it does not establish coverage of arbitrary latent changes, distinguish model-only from mixed, or supply posterior probabilities. The conditional simulator remains an experimental research adapter; the installed release has not acquired an automatic simulator through this documentation update.

The experimental reporter also accepts an explicit `no_model_candidates` list (default `['N', 'CL']`). For example, adding `CL_half` requires its own ordered, correctly matched conditional calibration records. A missing bank is unresolved; the reporter never silently drops a declared candidate. Model involvement requires rejection of every declared member at the supplied candidate-level cutoff. The list declares the scope of the calculation, not validated coverage of arbitrary latent laws. The same option passes through the existing Dynare/Octave JSON wrapper; no separate MATLAB statistic is introduced. This working-source extension does not rebuild installed archives or generate calibration paths.

### Independent fixed64 confirmation (same four DGP points)

The unchanged one-date/H10 experiment was evaluated on 64 new histories, 16 in each family. Model-only and mixed each produced 16 model-involvement assertions; no-change and latent-only each produced zero. All histories completed without support or technical unavailability. First-stage alarms were 2/16 for no change and 4/16 for latent-only; these six alarms remained source-unresolved. Thus latent-only detection and model-only versus mixed discrimination are not established by this confirmation.

For each model-present family, the exact 95% binomial interval for the all-assigned assertion rate is [0.7941, 1]; for each no-model family it is [0, 0.2059]. These describe sampling at the specified DGP point conditional on the fixed detector bank. They do not establish 99% reliability, uniform size, off-grid latent robustness, or a posterior probability of a source. The experiment retains the N/CL no-model catalogue; wider catalogues need matched calibration. Earlier development batches remain separate. This is experiment evidence, not a new release of the conditional simulator through this package.

### Finite latent-transition pilot and actual-record transport

An exploratory 16-history pilot used five no-model candidates (N, CL, CL_Q025, CL_Q050, CL_Q100), with four histories at each of three added latent transition parameters and four model-only histories. Alarm counts were0/4,1/4,3/4 for q=.25,.50,1.00; model assertions were0 at each latent point and4/4 at the model-only point. The parameter interpolates the RND2 latent transition matrix; it is not an HS beta parameter or a generic latent-level shift. Four histories per point do not establish a reliability frontier. Candidate retention does not identify latent involvement.

All16 actual terminal records were replayed through the working Python reporter and the Octave-to-Python Dynare transport, matching candidate ranks and final decisions. This verifies reporting on these saved records; it does not certify a released standalone conditional simulator, eta coverage, or continuous-parameter source identification. An independent seven-family confirmation is separate and is not pooled with this pilot.

### Interpreting model involvement and unresolved latent involvement

`decision = "model_involvement"` does not mean model-only. The reporter retains `latent = "unresolved"`; its p-values are candidate-law tests, not source probabilities. In a two-history development check using the existing one-date statistic, a new matching CM conditional bank did not reject either the first saved model-only history (p=.75) or the first saved mixed history (p=.35). Both used39 qualified proposals and the same .05 cutoff. This is not a source accuracy estimate, a proof of nonidentification, or confirmation of a CM fit. The original64-history confirmation supports its original model-involvement rule; it is development data for this subsequently considered CM-exclusion branch.

A technical failure from repeated instrumentation in the new two-history controller was preserved; only the second allocation was retried in a clean process without changing its random stream or scientific settings. The production reporter remains unchanged. No CM-versus-MIX decision or posterior probability has been added to the API.


### Model-strength development boundary (eight histories)

A fresh, fixed eight-history development sweep weakens the model-only coefficient to 0.25, 0.50, 0.75 and 1.00 times its original value, with two assigned histories at each value. It retains the same H10 direct-Bellman learner, detector and five no-model conditional candidates. At 0.25, one history produces no alarm and one alarms without a model-involvement assertion. At each other value both histories alarm and report model involvement. All eight execute; there are no monitor support refusals, technical failures or underfilled source banks. These are development counts, not an estimated reliability curve or evidence of monotonic power. A model-involvement assertion still leaves model-only versus mixed unresolved. No new release archive or standalone simulation capability is implied by this saved-result report.

The reproducible research record is `experiments/ecma_four_tracks_20260924/track3_identification/model_strength_frontier_dev_attempt001/`, including the terminal raw readback and `MODEL_STRENGTH_DEV8.pdf`.

Evidence94 independently evaluates the five-law no-model catalogue in 112 fresh histories, with sixteen assigned to each of N, four latent-transition specifications, CM and MIX. The H10 direct-Bellman learner, date-40 signed paired residual, fixed first-stage detector bank and conditional source-bank rule are unchanged. All CM and MIX histories alarm and assert model involvement (16/16 in each family); each no-model family has 0/16 model assertions. Alarms number 1/16 in N and 1/16, 0/16, 2/16 and 7/16 at latent-transition parameters .25, .50, .69375 and 1.00. The .25 group includes one support-unavailable monitor retained in its sixteen assignments. No history is technically invalid or unrun. Exact pointwise 95 percent binomial intervals have lower bound .7941 for 16/16 and upper bound .2059 for 0/16, conditional on independent histories, the fixed detector bank and fresh per-history conditional calibration. These results support model involvement within the declared finite catalogue; they do not establish 99 percent reliability, continuous latent robustness, CM-versus-MIX separation, or latent identification. All latent-source outputs remain unresolved. The working Python and Octave reporters reproduce all 112 records exactly and retain monitor support unavailability separately; this verifies the saved-result interface, not a released standalone simulator.

The version-2701 theory/calculation/assumption/output mapping is supplied as `docs/CONDITIONAL_REPORTING_2701.md` in both archives.

### Working-source four-class reporting extension

`pinc_toolbox.finite_source_set.finite_source_set` (Python) and `pinc_finite_source_set(payload, python_executable)` (Octave wrapper) report a finite retained source set. The input declares all four classes (`N`, `L`, `M`, `ML`), a single observed statistic and matching candidate calibration records. Missing or underfilled banks remain in the set. Model involvement alone does not imply model-only or mixed; an empty set is catalogue conflict. The existing nested-family diagnosis tree has different semantics and is not used for inheritance here.

This is working-source functionality, not part of the delivered2701 archives and not evidence of four-class detection power. Source probabilities remain null; matching conditional kernels and selection assumptions remain the caller's obligation. The current two-date CM-only experiment cannot populate the four-class interface. See `docs/CONDITIONAL_REPORTING_2701.md` for scope.


Evidence96 checks the original FULL snapshot40 numerical operator on one saved history against fixed FULL joint domains built from the 768 released pre-switch donors through date32. For each diagnostic eta value 1, 2 and 4, all 560 originally feasible node/action cells, 2,240 raw successors and 8,960 positive-weight native projection links retain every original feasible continuation option and floating previous-action anchor. Native transition targets match exactly; the saved H1-H10 recurrence has zero discrepancy. This is a no-op restriction for this specified finite numerical map, with no weight deletion or renormalization. Eta was supplied diagnostically, not selected. It is not a true-Q, population-support, statistical-eta or general-history certificate. No new fit, Q solve or DGP was run.

Saved HS H10 application figure: [PNG](../../experiments/ecma_four_tracks_20260924/track3_identification/hs_current_application_readback_attempt001/HS_H10_APPLICATION.png) ([PDF](../../experiments/ecma_four_tracks_20260924/track3_identification/hs_current_application_readback_attempt001/HS_H10_APPLICATION.pdf)). The four panels' path labels denote generator cases: N stable, M beta_y 1 to 1.5, S a -0.5 latent-state displacement under the fixed law, and MS both; changes first affect release133. Matched p-values are N=.15, M=.90, S=.05, MS=.05 from19 CAL proposals plus the observed path. This is a finite saved-input DEV illustration, not source classification, population power, or calibrated reliability.

Evidence97 is an independent fixed64 confirmation at four predeclared fractions of the model-change coefficient, with16 fresh histories per factor and the unchanged H10/date40 five-null conditional procedure. Alarm counts at factors .25/.5/.75/1 are7/16,10/16,16/16,16/16; model-involvement assertions are4/16,9/16,16/16,16/16. Unresolved counts are12/16,7/16,0/16,0/16, including all no-alarms; there were no monitor-support failures, technical-invalid or not-run assignments. The 9,555 conditional candidate proposals all qualified, with no underfilled banks. Pointwise exact two-sided95% binomial intervals (all assigned histories as denominator) are: 0.25: alarm 7/16 [0.1975, 0.7012], model assertion 4/16 [0.0727, 0.5238]; 0.5: alarm 10/16 [0.3543, 0.8480], model assertion 9/16 [0.2988, 0.8025]; 0.75: alarm 16/16 [0.7941, 1.0000], model assertion 16/16 [0.7941, 1.0000]; 1: alarm 16/16 [0.7941, 1.0000], model assertion 16/16 [0.7941, 1.0000]. These are pointwise finite-batch intervals conditional on the fixed first-stage detector and the declared per-history conditional-bank procedure, not simultaneous bands, conditional-on-alarm error rates, or99% guarantees. No monotonicity is inferred. The four named strengths do not establish a continuous boundary, latent identification, or CM-versus-MIX separation.

Evidence98 probes a further distinction between model-only and mixed change. Four fresh histories, two of each, used the unchanged first-stage monitor and a date-40 sum of squared FULL-w forecast errors over all 32 paths in the main low-action endpoint; companion actions followed their original schedule. Every history alarmed and had 39 qualified fresh CM calibration draws. Upper-tail ranks were .375 and .375 for the model-only histories, and .700 and .225 for the mixed histories, so none rejected the specified CM candidate at .05. This small prospective pilot did not reproduce the earlier exploratory separation signal. It supports neither a latent-involvement assertion nor a claim of observational equivalence. The model-involvement confirmation and this unresolved finer attribution remain distinct results.

Evidence99 adds a fixed32 q=1 comparison against the specified CM candidate: 16 fresh CM and16 observed-MIX histories, all32 triggering the unchanged H10 direct-Bellman union monitor. The candidate was rejected in0/16 CM histories (pointwise exact95% interval [0,.2059]) and9/16 MIX histories ([.2988,.8025]); three histories were source-U, and no assignment was technically invalid or not run. These are finite fixed-denominator event fractions, not proof of test size, latent posterior probability, or full four-class identification. The earlier DEV4 is not pooled.

The joint component procedure uses the signed action comparison and FULL-w residual energy from the same source episode, with separate qualification for each comparison. The signed comparison requires its original action pair; residual energy additionally requires all 32 paths. A route chosen from the released history before the source continuation determines which comparisons to attempt. Each component is reported only after all five laws in its finite no-component catalogue are rejected; otherwise it remains unknown. Under correct conditional simulation, true-law catalogue coverage whenever a component is absent, and matching observation/proposal qualification for each comparison, the two tests at level 0.025 have a union bound of 0.05 on false component reports conditional on the alarm-selected history. This bound does not condition on the other comparison's qualification or result. The proof and component-specific extension are given in the blue theory note. Neither the bound nor the implementation establishes power, pure-source classification, or a source posterior. The routed pilot is in progress; it is not part of the delivered package release or completed confirmation evidence.

### Experimental joint component reporting (working source only)

The working source now provides `pinc_toolbox.joint_component_report.joint_component_report` and the Dynare/Octave adapter `pinc_joint_component_report`. The JSON CLI is `python -m pinc_toolbox.joint_component_report_cli --input request.json --output result.json`. These interfaces report supplied conditional simulation records; they do not construct or certify candidate laws. They are not yet included in a newly delivered wheel or zip.

Supply `alarm_status` first. A non-alarm, unavailable monitor, technical failure or not-run case retains that status and cannot enter source inference. For a triggered alarm, supply the history-selected `source_route`, the observed pair and FULL-w qualification receipts, and ordered `attempts_by_candidate` carrying both component receipts for each proposal ID. The pair-only route retains all five no-model candidates; the joint route uses nine candidates with shared N. Each required comparison uses its own first 79 qualified proposals within 128, at level 0.025.

Outputs contain `model_involvement`, `latent_involvement`, an explicit unresolved reason, and candidate-level ranks in `details`. Unknown is not absence. Probability and posterior fields remain null. The four routed pilot records agree exactly through Python and the Octave adapter: unchanged and latent-only did not alarm; model-only and mixed reported model involvement, with latent respectively unknown and support-unavailable. This is interface parity and development evidence, not source-classification accuracy or independent confirmation.
# PINC toolbox
