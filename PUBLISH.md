# PUBLISH.md — 发布操作清单（GitHub → Zenodo）

维护者用。按顺序执行，全链路约 30 分钟。目标：v0.1.0 公开发布 + Zenodo DOI 生成 + 引文链闭合。

## 0. 发布前自查（1 分钟）

```bash
python run_tests.py          # 锚点回归 23/23 + 模拟冒烟 3/3 + 审计/复核回归 27/27 全 PASS
python examples/ex2_polymer.py --quick
python examples/ex1_rumor.py --quick
```

## 1. GitHub 建仓与推送（约 10 分钟）

1. 在 github.com/QINGQUAN-T 账号下新建 **Public** 仓库，名称 **A2SP**（不要加 README/License 初始化）。
2. 推送本目录：

```bash
cd A2SP
git init -b main
git add .
git commit -m "A2SP v0.1.0: reference implementation of the Asymmetric Two-Step Pairing framework"
git remote add origin https://github.com/QINGQUAN-T/A2SP.git
git push -u origin main
```

3. 仓库 About 侧栏填写 description（复制 pyproject 里的 description），Topics 添加：
   `a2sp` `asymmetric-two-step-pairing` `step-growth-polymerization` `rumour-spreading` `exact-solutions` `python`

## 2. Zenodo 绑定（约 5 分钟）

1. 登录 zenodo.org（与 symer 同一账号），右上角 → **GitHub**（或 Account → Applications → GitHub）。
2. 若未连接，授权 Zenodo 访问 GitHub；在仓库列表中把 **A2SP** 置为 **On**（Enabled）。
3. Zenodo 会显示该仓库即将归档；首次发布后生成 concept DOI。

## 3. 首次发布，生成 DOI（约 5 分钟）

```bash
git tag -a v0.1.0 -m "A2SP v0.1.0"
git push origin v0.1.0
```

然后在 GitHub → Releases → **Draft a new release** → 选择 tag `v0.1.0` → 标题 `A2SP v0.1.0` → Publish。
Zenodo 在数分钟内自动归档，得到 **concept DOI**（如 10.5281/zenodo.XXXXXX，解析到最新版本）。

## 4. DOI 回填（约 5 分钟，形成闭环）

> 版本号映射约定：Python 包/PyPI 用裸版本 `0.1.0`（PEP 440），git tag / Zenodo / CITATION.cff 用 `v0.1.0`——三者指同一发布，勿混改。

1. README.md / README.zh-CN.md：在 License 上方加徽章：
   `![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)`（换成实际 DOI）。
2. CITATION.cff：取消注释 `doi:` 行并填入实际 DOI；如需同时引论文 A，取消 preferred-citation 注释并补 arXiv 号。
3. commit + push（此后每次发版：改 pyproject version → tag → Release，Zenodo 自动出版本 DOI）。

## 5. 引文链闭合（随论文进度，不在本发布内）

- arXiv 号到手 → README 首屏引文行 `[arXiv:________]` 补号（两处语言版本 + CITATION.cff）。
- 论文 A 返修 / arXiv v2 → 加一句 "reference implementation available at github.com/QINGQUAN-T/A2SP (DOI)"。
- 论文 B 接收/返修 → 文末补引 a2sp 工具包（备忘事项，见工作传递包 v1.2 §5）。

## 6. 验收对照（工作传递包 v1.2 §6）

- [x] 全新虚拟环境一条命令安装（`pip install -r requirements.txt`）
- [x] 两个示例各一条命令跑通并 PASS
- [x] 陌生人 30 分钟测试：README → 安装 → 两示例 ≈ 5–10 分钟（含阅读）
- [x] README 首屏 A2SP 定位 + arXiv 引文行（号占位）与 DOI 位
- [x] 「扩展到你的领域」章节与模板
- [ ] Zenodo DOI 可访问且已回填 ← 执行本清单第 3–4 步后勾掉
- [x] 内核数值对齐 symer 公布锚点（最大偏差 3.3e-6，见 tests/test_anchors.py 输出）
- [x] 2026-11-02 外部审计全部发现已修复并有回归测试（tests/test_validation.py，27 项，含二轮复核 6 项）
- [x] 仓库内没有隐含任务
