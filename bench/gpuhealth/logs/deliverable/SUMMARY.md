# 交付文档 `docs/metal-m4.md` 的原始出处（2026-10-12）

本目录的六份 JSON（`cpu|metal` × `tiny|small|medium`）是
**`docs/metal-m4.md` §二/§三 两张表的唯一原始数据**。JSON 本身按仓库约定
不进版本库（`logs/*` 被 gitignore），这份 md 是它们的账本。

## 命令（逐份，共 6 次）

```bash
./target/release/qppocr bench/gpuhealth/gen100/img-*.jpg \
    --models models --tier <tiny|small|medium> --device <cpu|metal> \
    --workers 1 --quiet --json > bench/gpuhealth/logs/deliverable/<device>-<tier>.json
```

- **语料 = `bench/gpuhealth/gen100/`**（100 图 + `metadata.json`）。
  性能表用 `corpus100/`，**精度表必须用 gen100**——两套语料逐张像素尺寸相同
  但内容不同（sha256 100/100 全不同），拿 corpus100 配 gen100 的真值就是拿错
  答案对卷子。
- `--workers 1`：精度与分阶段都要单进程，否则输出顺序与计时都会混。
- 跑前过了 `bench/gpuhealth/health.sh`（判据是**实测带宽** 91.13 GB/s）。
- 二进制 commit `b6471e8`。

## 复算这两张表

```bash
# §二 分阶段（ms/图，按图求均值）
./bench/gpuhealth/stage_table.py \
    CPU=bench/gpuhealth/logs/deliverable/cpu-small.json \
    Metal=bench/gpuhealth/logs/deliverable/metal-small.json

# §三 精度：原样 / 去空格两种口径都要出
./bench/gpuhealth/acc_truth.py --tier small --device metal \
    --from-json bench/gpuhealth/logs/deliverable/metal-small.json
./bench/gpuhealth/acc_truth.py --tier small --device metal \
    --from-json bench/gpuhealth/logs/deliverable/metal-small.json --strip-space
```

## 本轮核过、写进文档的关键数

| 项 | 值 |
|---|---|
| 检出框数（CPU == Metal 逐图，300 图 0 差） | tiny 1034 / small 1020 / medium 1020 |
| 空文本行（曾经的「静默丢行」那列） | **全档 0** |
| cls 第二意见触发（重试行 / 改判行） | tiny 12/11、small 8/8、medium 2/2（CPU 侧同量级） |
| 对真值 exact（原样，CPU / Metal） | tiny 891/889、small 995/995、medium 1014/1015（分母 1011/1016/1017）|
| 对真值 exact（去空格） | tiny 995/997、small 1012/1011、medium 1015/1015 |
| Metal-vs-CPU 逐图文本差（实质 / 只差空白） | tiny 10/29、small 2/7、medium 2/1；**行数与框数 0 差** |

## 没写进文档、但复核时发现的

- `decode_ms` 在六份 JSON 的 600 张图里**全是 0.0** ⇒ 这条路径根本没给解码
  计时（不是「解码很快」）。含解码的墙钟只有 `qppocr bench` 的 `wall` 列有，
  两设备都约 +2.3 ms/图。
- 内存那一节（`docs/metal-m4.md` §四）**用的不是本目录的数据**——它是
  `/usr/bin/time -l` 单独量的，因为两个坑：`ps -o rss=` 轮询比 `ru_maxrss`
  少采 30~55%；同一命令的 RSS 在这台机器上有**两个稳定档**（相差 2~4×）。
  详见文档 §四.1 的两个警告框。
