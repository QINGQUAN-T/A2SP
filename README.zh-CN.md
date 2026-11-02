# A2SP — 非对称两步配对（Asymmetric Two-Step Pairing）

**一个配对内核，任意领域。**

`a2sp` 是 **A2SP 框架的参考实现**：一类两步过程的最小精确结构——所有通道速率都携带同一个*载体因子*（伙伴池：燃料平衡、传播者池……），它在约化轨迹方程中**精确消去**，只留下仅含不对称参数的闭式、免拟合定律。

- **内核来源（理论）**：Q. Tang《A Unified Theory for Generalized Step-Growth Polymerization: From Flory's Equal-Reactivity to Kinetic Symmetry-Breaking》（JACS 投稿；ChemRxiv 预印本 DOI：**10.26434/chemrxiv.15009468/v1**）。高分子"两点动力学方案"是本框架的数学源头。
- **第一个推广应用**：Q. Tang《Terminal-state theorems for a two-stage Maki–Thompson rumor model…》（预印本；**arXiv:2610.41206**）。
- **引文行**：*Reference implementation of the A2SP framework introduced in [arXiv:2610.41206]; kernel source: the step-growth polymerization unified theory (ChemRxiv 10.26434/chemrxiv.15009468/v1).*

同一内核适用于一切符合两步配对结构的领域——谣言传播、逐步聚合，并可推广到金融、管理科学等（见下文「把 A2SP 用到你的领域」）。

## 内核：一个方程

载体因子精确消去后，所有 A2SP 轨迹服从单参数约化律

$$\frac{dY}{dX} = -1 + \kappa\, \frac{Y}{X}, \qquad \kappa > 0,$$

**共振线 κ = 1**（精确对数支）。三个实例：

| 家族 | 载体 | κ | 共振线 |
|---|---|---|---|
| 逐步聚合（内核来源） | 燃料平衡 [A]（消耗型） | α/2 | α = 2（对数尾） |
| 谣言·合并层 | 传播者池 S（催化型） | μ | μ = 1（对数临界） |
| 谣言·职业层（重激活） | 传播者池 S | γ = α + μ | γ = 1（对数支） |

包内闭式：`b1(b0; α) = 2(b0^(α/2) − b0)/(2 − α)`、Carothers `X̄n = (1+r)/(1+r−2pB)`、终态 arrest 方程、冻结库存律 `(x1, x2) = (μ, α)/(α+μ)`、指纹反演 `α = μ·x2/x1` 等（见 `a2sp/models`）。

> **术语说明：** *arrest*（内核论文用语，动力冻结）指限制性 A 基团耗尽（r < 1）导致的化学计量学冻结——所有通道速率同时归零，捕获一个**可复燃**的冻结态（残留活化库存 b0f、b1f > 0）。刻意不叫 "termination/终止"：逐步聚合中不存在终止反应，也没有任何活性中心被销毁（补加 A 单体反应即恢复）；"termination" 是链式聚合里"活性中心机理销毁"的专属词汇。"terminate(s)" 仅作普通动词使用。

## 安装（约 2 分钟）

```bash
git clone https://github.com/QINGQUAN-T/A2SP.git
cd A2SP
pip install -r requirements.txt      # 全量：numpy + matplotlib（示例出图用）
# 或仅装库本体（只依赖 numpy；matplotlib 为可选附加项）：
# pip install .
```

## 快速上手——两个示例，各一条命令

```bash
python examples/ex2_polymer.py     # 内核来源：逐步聚合（约 10 秒）
python examples/ex1_rumor.py       # 第一个应用：谣言传播（约 1 分钟）
```

每个示例打印对照已发表数值的验证表、将图存到 `examples/output/`、并写出记录全部随机种子的结果 JSON。加 `--quick` 为秒级冒烟模式。

**「30 分钟可用」的实测含义**：安装约 2 分钟；`python run_tests.py` 约 1 分钟（53 项回归）；`ex2_polymer.py` 约 10 秒；`ex1_rumor.py` 约 1 分钟（含出图）。无外部数据下载、无编译、无账号。

应当看到：

- `ex2_polymer.py` — 8/8 PASS：Carothers、Flory、X̄w 双路径锚（721/239 = 3.016736…）、冻结态锚（α=3, r=0.9）、ODE↔闭式 arrest 一致，以及实时两点 ODE 落在约化曲线上（偏差 ~1e-10，载体消去得到验证）。
- `ex1_rumor.py` — 3/3 PASS：冻结库存律（定理 2）经两条独立路线确认——层闭式精确求积（与定理吻合至 ~1e-12）与自适应 ODE 积分（Richardson 收敛验证 <1e-9）；充分混合 Gillespie SSA（N = 6×10⁴，配对自校正）与配置模型网络级联（⟨k⟩ = 8）收敛到同一终态构成 (x1, x2) = (μ, α)/(α+μ)。

## 把 A2SP 用到你的领域

若你的系统是两步配对过程，只需两件事：你的**不对称参数组合**与你的**载体类型**（消耗型 / 催化型）。约化律与闭式已为你写好：

```python
from a2sp.core import reduced_solution, residual

kappa = ...   # 由你的通道速率导出 dY/dX = -1 + kappa*Y/X（参照 a2sp/models 的两个范本）

Y = reduced_solution(X, kappa, X0, Y0)     # 任意状态出发的闭式轨迹
dev = residual(kappa, X0, Y0, X_end)       # ~1e-11：精确消去，自动验证
```

两个模型文件即工作范本：复制最接近的一个、改名参数、保持同样的锚点测试纪律。载体为消耗型（如 [A]）或催化型（如 S）不影响约化——改变的只是 X、Y 的物理含义。

## 验证

`python run_tests.py` 运行全部测试：

- **23 项已发表锚点回归**（`tests/test_anchors.py`）：所有期望值来自论文 SI 或 symer 内核 README——无一来自本代码库。
- **27 项审计与复核回归**（`tests/test_validation.py`）：2026-11-02 首轮审计的全部缺陷（4 项正确性、3 项崩溃/挂起、7 项健壮性）与二轮复核的 6 项低危问题各有专属回归测试；越界输入与真溢出均显式抛错而非静默出错。与任一已发表锚点的最大偏差 3.3×10⁻⁶（即发布值本身的 5 位有效数字舍入）；精确恒等式（Carothers、冻结库存律、指纹反演）达到机器精度。
- **随机模拟冒烟测试**（`tests/test_sim.py`）：SSA 与网络引擎在有限 N 容差内复现冻结库存律。

**来源说明**：锚点出自作者本人手稿与作者的 symer 复现包——即*自发布*的交叉核查（公式内部一致性、双路径求和与行走公式一致至 4×10⁻¹⁶），尚非独立第三方验证；论文发表后即可进行独立验证。

交叉验证对象：[symer v0.1.3](https://github.com/QINGQUAN-T/symer)（Zenodo: 10.5281/zenodo.23282847）——谣言论文的专属复现包。`symer` 存档该论文的图与数据管道；`a2sp` 是可移植的框架核心。二者相互独立、互相链接。

## 引用

见 `CITATION.cff`。请引用工具包的 Zenodo DOI（首次发布时生成），并配合理论论文（聚合统一理论）和/或第一个应用（arXiv 预印本）。

## 许可证

MIT — 见 `LICENSE`。
