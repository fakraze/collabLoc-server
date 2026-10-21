# Third-party code

以下目錄原本是獨立的 git repo,已移除其 `.git`,現以一般檔案納入本專案版本管理。

| 目錄 | 來源 | 版本 | 備註 |
|---|---|---|---|
| `RAFT/` | https://github.com/princeton-vl/RAFT.git | `aac9dd54726caf2cf81d8661b07663e220c5586d` (master, 2021-10-13, Merge PR #113) | 有本地修改:`core/corr.py`、`core/raft.py`;另新增 `model_converter.py`、`test_hloc.py`、`test_hloc/`、`vbgps.py` |
| `third_party/deep-image-retrieval/` | https://github.com/naver/deep-image-retrieval | 未知 | 原為 hloc 的 submodule,commit 資訊已遺失 |
| `third_party/r2d2/` | 待補 | 未知 | 原為 hloc 的 submodule,commit 資訊已遺失 |
| `third_party/d2net/` | 待補 | 未知 | 原為 hloc 的 submodule,commit 資訊已遺失 |
| `third_party/SuperGluePretrainedNetwork/` | 待補 | 未知 | 原為 hloc 的 submodule,commit 資訊已遺失 |

`datasets/` 已列入 `.gitignore`,不納入版本管理。
