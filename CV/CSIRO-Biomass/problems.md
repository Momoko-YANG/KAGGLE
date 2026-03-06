
## 发现的问题

### 问题 1：`head_state` 输出维度错误（严重）

```42:53:01_phase1_meta.py
        self.head_species = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_species_classes)
        )

        self.head_state = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_species_classes)  # ← 应该是 num_state_classes
        )
```

构造函数中定义了 `num_state_classes=6`，但 `head_state` 错误地使用了 `num_species_classes`（默认为3），导致状态分类器输出维度不正确。

**修复方案：**
```python
self.head_state = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_state_classes)  # 使用正确的参数
)
```

---

### 问题 2：`head_ndvi` 输出维度可疑

```34:39:01_phase1_meta.py
        self.head_ndvi = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_species_classes)  # ← NDVI 是回归任务，应该输出 1
        )
```

NDVI 通常是单个连续值的回归任务，但这里输出 `num_species_classes`（3个值）。后续代码用 `.flatten()` 处理，但这可能不是预期的行为。

**修复方案（如果 NDVI 是单值回归）：**
```python
self.head_ndvi = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1)  # 单值回归
)
```

---

### 问题 3：`common_defs.py` 中的拼写错误（严重）

```59:59:c:\Users\Y101355\Desktop\new\kaggle-image\common_defs.py
        kernel = np.ones((3, 3), np.unit8)  # ← 拼写错误！
```

`np.unit8` 应该是 `np.uint8`，这会导致运行时 `AttributeError`。

**修复方案：**
```python
kernel = np.ones((3, 3), np.uint8)
```

---

### 问题 4：`common_defs.py` 中 `BiomassOctaTileModel` 的括号错误（严重）

```134:134:c:\Users\Y101355\Desktop\new\kaggle-image\common_defs.py
        t_feat = self.global_pool(self.backbone(tiles.view(B*N, C, H, W)).flatten(1))
```

这里 `.flatten(1)` 在 `global_pool` 之前执行，会把 4D 张量变成 2D，然后 `AdaptiveAvgPool2d` 会报错（期望 4D 输入）。对比正确的第129行：

```python
g_feat = self.global_pool(self.backbone(g_img)).flatten(1)  # 正确顺序
```

**修复方案：**
```python
t_feat = self.global_pool(self.backbone(tiles.view(B*N, C, H, W))).flatten(1)
```

---

### 问题 5：`DEBUG_MODE` 验证不完整

```194:200:01_phase1_meta.py
    if DEBUG_MODE:
        gt_df = pd.read_csv(TEST_CSV)
        if 'Height_Ave_cm' in gt_df.columns:
            merged = pd.merge(output_df, gt_df[['image_path', 'Height_Ave_cm', 'NDVI']].drop_duplicates(),
                                on = 'image_path', how='left')
            print("\n [Phase 1 Validation Scores]")
            print(f"  - Height R2: {r2_score(merged['Height_Ave_cm'], merged['pred_height']):.4f}")
            # 缺少 NDVI R2 的输出
```

代码 merge 了 NDVI 列但没有计算和打印 NDVI 的 R² 分数。

**修复方案：**
```python
print(f"  - Height R2: {r2_score(merged['Height_Ave_cm'], merged['pred_height']):.4f}")
print(f"  - NDVI R2: {r2_score(merged['NDVI'], merged['pred_ndvi']):.4f}")  # 添加这行
```

---

### 问题 6：`img_dir` 类型安全性（建议改进）

```68:69:01_phase1_meta.py
    def __init__(self, df, img_dir, img_size=512):
        # ...
        self.img_dir = img_dir  # 假设是 Path 对象
```

虽然调用时传入的是 `Path` 对象，但建议显式转换以提高健壮性：

```python
self.img_dir = Path(img_dir)
```

---

## 问题汇总表

| 严重程度 | 文件 | 问题描述 |
|---------|------|---------|
| 🔴 严重 | `common_defs.py:59` | `np.unit8` → `np.uint8` 拼写错误 |
| 🔴 严重 | `common_defs.py:134` | `global_pool` 与 `flatten` 顺序错误 |
| 🟠 重要 | `01_phase1_meta.py:52` | `head_state` 使用了错误的类数参数 |
| 🟡 可疑 | `01_phase1_meta.py:38` | `head_ndvi` 输出维度可能不正确 |
| 🟢 建议 | `01_phase1_meta.py:200` | 缺少 NDVI R² 验证输出 |
| 🟢 建议 | `01_phase1_meta.py:69` | 建议显式转换 `img_dir` 为 Path |

---