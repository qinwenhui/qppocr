# 发布与版本管理

本仓库的引用模型与发布流程。写下来是为了让每一步可检查、可复现——
版本管理的事不靠记忆。

## 引用模型（谁是什么）

| 引用 | 作用 | 推送 |
|---|---|---|
| `internal-main` | 唯一开发主线，含内部叙述（bench/COMPARISON.md 等） | **绝不推送** |
| `public-snapshot` | 最近一次「公开快照」的指针 = 远端 `main` 的内容 | force push 到 `origin/main` |
| `vX.Y.Z`（tag） | 发布版，指向当次快照的孤儿提交 | push 触发 `release.yml` 出产物 |
| `release-v0.1.0 … v0.2.1` | 历史快照分支，与同名 tag 同一提交（冗余存量，**今后不再新建**） | 不推 |

公开快照 = 孤儿单提交（无父历史）：内容 = internal-main 指定提交
**剔除内部文件**（`bench/` 全目录：COMPARISON.md、ab.py、abx.py、
verify.py），提交信息为对外描述（发布快照固定为 `qppocr X.Y.Z`）。
内部历史（移植叙述、对拍数据、基准口径）因此永不出现在公开仓库。

**不建长命 release-prep 分支**：发布准备（升版 + CHANGELOG 落段）做成
internal-main 上的普通提交，从它切快照。0.3.0 曾开 `release-prep/0.3.0`
分支，主线前进 8 个提交后过期作废——这类分支只会制造「另一个要同步
的地方」。已在 2026-09-28 删除。

## 发布流程（checklist）

1. **冻结**：确定 internal-main 切点提交 `C`（发布内容 = 它的全部历史）。
2. **升版提交**（在 internal-main 上，一个提交）：
   - 根 `Cargo.toml` 的 `workspace.package.version` 升到 `X.Y.Z`（四个
     crate 经 workspace 统一，勿逐个改）；
   - `CHANGELOG.md`：`[Unreleased]` → `## [X.Y.Z] - 日期`，另起空的
     `[Unreleased]`。
3. **全量准入**（在切点上，全部通过才继续）：
   ```
   cargo build --workspace --all-targets && cargo test --workspace
   cargo clippy --workspace --all-targets && cargo fmt --all --check
   cargo build --release                 # bench/verify 用的 exe
   python bench/verify.py diff base      # 行为变化则基线重建（见下）
   python bench/verify.py diff pre-small small
   ```
4. **快照 + tag**（工作区不动，临时 index 操作）：
   ```
   export GIT_INDEX_FILE=$PWD/.git/snap-index
   git read-tree C
   git rm -r --cached bench
   TREE=$(git write-tree)
   rm $GIT_INDEX_FILE; unset GIT_INDEX_FILE
   COMMIT=$(git commit-tree $TREE -m "qppocr X.Y.Z")
   git tag vX.Y.Z $COMMIT
   git branch -f public-snapshot $COMMIT
   ```
5. **推送与核对**：
   ```
   git push origin public-snapshot:main --force   # 孤儿间无 fast-forward
   git push origin vX.Y.Z
   ```
   推后逐项确认：CI 在新 main 上全绿；Release 页产物齐（三平台 CLI）；
   `git ls-remote` 的 main / tag 与本地一致。

日常「公开 main 预览」随时可做（不等发布）：同样剔除 bench/ 做孤儿
提交、`git branch -f public-snapshot`、force push main——CI 即在公开
内容上验证。

## 改动准入（每次提交的纪律）

- **声明 = 断言**：写进 CHANGELOG/README 的默认值与行为，必须有对应
  测试锁住（先例：`preset_contract_matches_docs`、
  `documented_defaults_are_asserted`——区域重试的声明与行为脱节
  发布了 0.2.0/0.2.1 两个版本才被发现，2026-09-28 修复）。
- **行为变化**必须过 `bench/verify.py` 逐字符对拍；预期中的变化重建
  基线（`save`）并在提交信息写明前后行数。
- **性能改动**用 `bench/abx.py` 交错 A/B，报比值中位；先跑
  `examples/scaling_probe` 确认机器噪声 <1%（口径详见
  bench/COMPARISON.md §4）。
- 对拍/跑分前 `cargo build --release`——脚本用的是预编译 exe。
- 版本号：0.x 阶段破坏性变化也走 minor，CHANGELOG 条目标 `!` 或
  BREAKING 字样。
