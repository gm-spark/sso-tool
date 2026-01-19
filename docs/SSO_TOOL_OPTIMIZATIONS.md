# SSO Tool Optimizations

> **Purpose**: Document performance optimizations and architectural improvements for the SSO Tool conversion.

---

## Overview

The original Python script was designed for single-user desktop execution. Converting to a web application introduces new requirements:

- **Lower latency** - Users expect fast responses
- **Memory efficiency** - Cloud instances have limited RAM
- **Concurrent users** - Multiple analysts may run simultaneously
- **Cost optimization** - Pay-per-use infrastructure

This document outlines optimizations to address these requirements.

---

## 1. Vectorize WOS Calculations (Remove For Loop)

### Problem

The original code loops through each WOS target (1-13), creating a full copy of the dataframe each iteration:

```python
# Original: O(n * w) memory, O(n * w) time
# where n = rows, w = WOS targets (13)

df_dict = {}
for x in WOS_targets:  # 1-13
    df = sso.copy()    # Full copy each iteration!
    df['WOS_target'] = df['avg_pos_quantity_L8W'] * x
    df['WOS_target'] = df['WOS_target'].apply(lambda x: min(4, x))
    df['need'] = df['WOS_target'] - df['F4 Pipe']
    df['order_packs'] = df.apply(lambda row: calculate_order_packs(...), axis=1)
    # ... more calculations ...
    df_dict[f'df_{x}'] = df

combo = pd.concat(df_dict.values(), ignore_index=True)
```

**Issues:**
- 13 full copies of the dataframe in memory
- Redundant calculations (pipeline, retails repeated 13x)
- `.apply()` with `axis=1` is extremely slow (Python loop under the hood)

### Solution: Single-Pass Vectorization

#### Option A: Polars Cross-Join (Recommended)

```python
import polars as pl

def calculate_all_wos_targets(sso: pl.DataFrame, max_units: int) -> pl.DataFrame:
    """Calculate all WOS targets in a single vectorized pass."""

    # Create WOS target dimension
    wos_targets = pl.DataFrame({"wos_weeks": list(range(1, 14))})

    return (
        sso
        # Cross-join: every row × every WOS target
        .join(wos_targets, how="cross")

        # Calculate target and need (vectorized)
        .with_columns([
            pl.min_horizontal(
                pl.col("avg_pos_quantity") * pl.col("wos_weeks"),
                pl.lit(max_units)
            ).alias("wos_target"),
        ])
        .with_columns([
            (pl.col("wos_target") - pl.col("f4_pipe")).alias("need"),
        ])

        # Calculate order packs (vectorized, no apply!)
        .with_columns([
            (pl.col("need") / pl.col("warehouse_pack_quantity")).alias("pack_ratio"),
        ])
        .with_columns([
            pl.when(pl.col("pack_ratio") < 0.5)
              .then(0)
              .when(pl.col("pack_ratio") < 1.0)
              .then(1)
              .otherwise(pl.col("pack_ratio").round(0))
              .cast(pl.Int32)
              .alias("order_packs"),
        ])

        # Apply minimum pack override
        .with_columns([
            pl.when(pl.col("f4_pipe") <= 1)
              .then(pl.max_horizontal(pl.col("order_packs"), pl.lit(1)))
              .otherwise(pl.col("order_packs"))
              .alias("whse_packs_distro"),
        ])

        # Calculate final values
        .with_columns([
            (pl.col("whse_packs_distro") * pl.col("warehouse_pack_quantity"))
                .alias("eaches_distro"),
        ])
    )
```

#### Option B: NumPy Broadcasting (pandas)

```python
import numpy as np
import pandas as pd

def calculate_all_wos_targets_numpy(sso: pd.DataFrame, max_units: int) -> pd.DataFrame:
    """Calculate all WOS targets using NumPy broadcasting."""

    wos_targets = np.arange(1, 14)  # [1, 2, ..., 13]
    n_rows = len(sso)
    n_wos = len(wos_targets)

    # Get base arrays
    avg_qty = sso['avg_pos_quantity'].values  # (n_rows,)
    f4_pipe = sso['F4_Pipe'].values           # (n_rows,)
    whse_pack = sso['warehouse_pack_quantity'].values  # (n_rows,)

    # Broadcast calculations: (n_rows,) × (n_wos,) → (n_rows, n_wos)
    all_targets = avg_qty[:, np.newaxis] * wos_targets  # (n_rows, 13)
    all_targets = np.minimum(all_targets, max_units)

    all_needs = all_targets - f4_pipe[:, np.newaxis]

    # Vectorized order pack calculation
    pack_ratios = all_needs / whse_pack[:, np.newaxis]
    order_packs = np.where(
        pack_ratios < 0.5, 0,
        np.where(pack_ratios < 1.0, 1, np.round(pack_ratios))
    ).astype(int)

    # Apply override
    override_mask = (f4_pipe <= 1)[:, np.newaxis]
    order_packs = np.where(override_mask, np.maximum(order_packs, 1), order_packs)

    # Flatten back to long format
    # ... (expand sso rows and combine with calculated values)

    return result
```

### Performance Comparison

| Approach | Memory | Speed | Code Complexity |
|----------|--------|-------|-----------------|
| Original (for loop) | 13x base | 1x (baseline) | Simple |
| Polars cross-join | 1.5x base | 5-10x faster | Medium |
| NumPy broadcasting | 1.2x base | 8-15x faster | Higher |

**Recommendation**: Use Polars for best balance of performance and readability.

---

## 2. Replace pandas with Polars

### Problem

pandas loads entire files into memory and has significant overhead:

```python
# pandas: 500MB CSV → ~1.5GB RAM
df = pd.read_csv("large_file.csv")
```

### Solution

Polars uses Apache Arrow, lazy evaluation, and is 2-10x more memory efficient:

```python
# Polars lazy: 500MB CSV → ~200MB RAM (only loads what's needed)
df = (
    pl.scan_csv("large_file.csv")
    .filter(pl.col("department") == 29)
    .group_by("item")
    .agg(pl.sum("quantity"))
    .collect()
)
```

### Migration Guide

| pandas | Polars |
|--------|--------|
| `pd.read_csv(f)` | `pl.read_csv(f)` or `pl.scan_csv(f)` (lazy) |
| `df[df['col'] == x]` | `df.filter(pl.col('col') == x)` |
| `df.groupby('col').sum()` | `df.group_by('col').agg(pl.sum('*'))` |
| `df.merge(df2, on='col')` | `df.join(df2, on='col')` |
| `df.apply(fn, axis=1)` | `df.with_columns(...)` (vectorized) |
| `df['new'] = df['a'] + df['b']` | `df.with_columns((pl.col('a') + pl.col('b')).alias('new'))` |

### Memory Comparison by File Size

| File Size | pandas RAM | Polars RAM | Savings |
|-----------|------------|------------|---------|
| 10MB | ~30MB | ~15MB | 50% |
| 100MB | ~300MB | ~100MB | 67% |
| 500MB | ~1.5GB | ~300MB | 80% |

---

## 3. Eliminate Redundant `.apply()` Calls

### Problem

`.apply(axis=1)` iterates row-by-row in Python, defeating vectorization:

```python
# SLOW: Python loop disguised as pandas
df['order_packs'] = df.apply(
    lambda row: calculate_order_packs(row['need'], row['whse_pack']),
    axis=1
)
```

### Solution

Replace with vectorized operations:

```python
# FAST: True vectorization
pack_ratio = df['need'] / df['whse_pack']

df['order_packs'] = np.where(
    pack_ratio < 0.5, 0,
    np.where(pack_ratio < 1.0, 1, np.round(pack_ratio))
)
```

### Common Patterns to Vectorize

| Slow (apply) | Fast (vectorized) |
|--------------|-------------------|
| `df.apply(lambda r: min(r['a'], r['b']), axis=1)` | `np.minimum(df['a'], df['b'])` |
| `df.apply(lambda r: r['a'] if r['c'] else r['b'], axis=1)` | `np.where(df['c'], df['a'], df['b'])` |
| `df['x'].apply(lambda x: x.strip())` | `df['x'].str.strip()` |
| `df.apply(lambda r: fn(r['a'], r['b']), axis=1)` | `np.vectorize(fn)(df['a'], df['b'])` |

---

## 4. Lazy Loading for Large Files

### Problem

Loading all data upfront wastes memory when filtering reduces dataset significantly:

```python
# Loads 500MB, filters to 50MB, wastes 450MB
ss = pd.read_csv("Store_Sales.csv")
ss = ss[ss['department'] == 29]  # Only need 10% of rows
```

### Solution

Use Polars lazy evaluation or chunked reading:

#### Polars Lazy (Recommended)

```python
# Only reads rows matching filter from disk
ss = (
    pl.scan_csv("Store_Sales.csv")
    .filter(pl.col("department") == 29)
    .collect()
)
```

#### DuckDB (SQL-based, extremely efficient)

```python
import duckdb

# Queries CSV directly without loading into memory
result = duckdb.query("""
    SELECT *
    FROM read_csv_auto('Store_Sales.csv')
    WHERE department = 29
""").pl()  # Returns Polars DataFrame
```

#### pandas Chunked (Fallback)

```python
# Process in chunks if must use pandas
chunks = []
for chunk in pd.read_csv("Store_Sales.csv", chunksize=50000):
    filtered = chunk[chunk['department'] == 29]
    chunks.append(filtered)

ss = pd.concat(chunks, ignore_index=True)
```

---

## 5. Smart File Size Routing

### Problem

Small files don't need heavy optimization; large files need special handling.

### Solution

Detect file size and route to appropriate processing strategy:

```python
import os
from enum import Enum

class ProcessingStrategy(Enum):
    STANDARD = "standard"      # < 50MB: Regular Polars
    LAZY = "lazy"              # 50-200MB: Polars lazy
    STREAMING = "streaming"    # > 200MB: DuckDB streaming

def get_strategy(file_path: str) -> ProcessingStrategy:
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if size_mb < 50:
        return ProcessingStrategy.STANDARD
    elif size_mb < 200:
        return ProcessingStrategy.LAZY
    else:
        return ProcessingStrategy.STREAMING

async def process_file(file_path: str, params: dict):
    strategy = get_strategy(file_path)

    match strategy:
        case ProcessingStrategy.STANDARD:
            return process_with_polars(file_path, params)
        case ProcessingStrategy.LAZY:
            return process_with_polars_lazy(file_path, params)
        case ProcessingStrategy.STREAMING:
            return process_with_duckdb(file_path, params)
```

---

## 6. Pre-compute Static Data

### Problem

Some data is recalculated on every run but rarely changes:

- Walmart calendar week lookups
- Store exclusion lists
- Supplier configurations

### Solution

Cache static data in memory or database:

```python
from functools import lru_cache
import polars as pl

@lru_cache(maxsize=1)
def get_wmt_calendar() -> pl.DataFrame:
    """Load and cache Walmart calendar (refreshes on app restart)."""
    return pl.read_csv("wmt_calendar.csv")

@lru_cache(maxsize=1)
def get_excluded_stores() -> set[int]:
    """Load and cache FC stores to exclude."""
    # Could also fetch from database
    return {747, 2360, 2552, 2666, 2680, 3008, ...}

def get_current_week() -> int:
    """Get current Walmart week from cached calendar."""
    cal = get_wmt_calendar()
    today = date.today()
    row = cal.filter(pl.col("date") == today)
    return row["wmt_week"][0]
```

---

## 7. Backend Wake-Up Pattern

### Problem

Scale-to-zero deployments (Railway, Render free tier) have 10-15 second cold starts.

### Solution

Trigger wake-up when user navigates to the run page:

```typescript
// Frontend: Wake up backend while user fills form
useEffect(() => {
    fetch(`${API_URL}/health`).catch(() => {});
}, []);
```

```python
# Backend: Simple health endpoint
@router.get("/health")
async def health():
    return {"status": "healthy"}
```

**Timeline:**
```
0s    - User clicks "New Run"
0s    - Frontend sends /health ping (backend starts waking)
0-15s - User selects supplier, department, uploads files
15s   - Backend is now warm
15s   - User clicks "Calculate" → instant response
```

---

## 8. Async File Processing

### Problem

Sequential file processing blocks the API:

```python
# Slow: Each file processed sequentially
async def process_run(files: dict):
    sso_data = parse_csv(files['sso'])           # Wait...
    forecast = parse_csv(files['forecast'])       # Wait...
    inventory = parse_excel(files['inventory'])   # Wait...
```

### Solution

Process independent files concurrently:

```python
import asyncio

async def process_run(files: dict):
    # Parse all files concurrently
    sso_data, forecast, inventory = await asyncio.gather(
        parse_csv_async(files['sso']),
        parse_csv_async(files['forecast']),
        parse_excel_async(files['inventory']),
    )

    # Now do calculations with all data ready
    return calculate(sso_data, forecast, inventory)
```

---

## 9. Progress Streaming for Long Calculations

### Problem

Large supplier calculations may take 30+ seconds. Users see no feedback.

### Solution

Stream progress updates via Server-Sent Events (SSE):

```python
# Backend
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

@router.post("/calculate/stream")
async def calculate_stream(params: CalculationParams):
    async def generate():
        yield f"data: {json.dumps({'step': 'loading', 'progress': 0})}\n\n"

        data = await load_data(params)
        yield f"data: {json.dumps({'step': 'calculating', 'progress': 30})}\n\n"

        for i, wos in enumerate(params.wos_targets):
            result = calculate_wos(data, wos)
            progress = 30 + (i / len(params.wos_targets)) * 60
            yield f"data: {json.dumps({'step': f'wos_{wos}', 'progress': progress})}\n\n"

        yield f"data: {json.dumps({'step': 'complete', 'progress': 100})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

```typescript
// Frontend
const eventSource = new EventSource('/api/calculate/stream');
eventSource.onmessage = (event) => {
    const { step, progress } = JSON.parse(event.data);
    setProgress(progress);
    setCurrentStep(step);
};
```

---

## 10. Output File Generation Optimization

### Problem

Generating 13 NOVA Excel files sequentially is slow.

### Solution

Generate files in parallel and use efficient Excel library:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def generate_all_outputs(results: dict, template_path: str):
    """Generate all NOVA files in parallel."""

    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                generate_nova_file,
                results[wos],
                template_path,
                wos
            )
            for wos in results.keys()
        ]

        files = await asyncio.gather(*tasks)

    return files
```

### Alternative: Use xlsxwriter Instead of openpyxl

`xlsxwriter` is faster for creating new files (openpyxl is better for editing existing):

```python
import xlsxwriter

def generate_nova_fast(data: pl.DataFrame, output_path: str):
    """Generate NOVA file using xlsxwriter (2-3x faster)."""

    workbook = xlsxwriter.Workbook(output_path)
    worksheet = workbook.add_worksheet('NOVA form')

    # Write headers
    for col, header in enumerate(data.columns):
        worksheet.write(0, col, header)

    # Write data (fast native method)
    for row_idx, row in enumerate(data.iter_rows(), start=1):
        for col_idx, value in enumerate(row):
            worksheet.write(row_idx, col_idx, value)

    workbook.close()
```

---

## Summary: Expected Performance Gains

| Optimization | Impact | Effort |
|--------------|--------|--------|
| Vectorize WOS loop | 5-10x faster | Medium |
| Replace pandas → Polars | 2-5x less memory | Medium |
| Remove `.apply()` calls | 10-50x faster on those ops | Low |
| Lazy loading | 50-80% less memory | Low |
| Async file processing | 2-3x faster load | Low |
| Parallel Excel generation | 3-4x faster export | Medium |
| Wake-up pattern | Eliminates cold start UX | Low |

**Combined effect**: A calculation that takes 60 seconds with 2GB RAM could take 10 seconds with 500MB RAM.

---

## Implementation Priority

### Phase 1 (Must Have)
1. Vectorize WOS loop - biggest performance win
2. Replace `.apply()` with vectorized operations
3. Use Polars for core calculations

### Phase 2 (Should Have)
4. Lazy loading for large files
5. Backend wake-up pattern
6. Progress streaming

### Phase 3 (Nice to Have)
7. Parallel Excel generation
8. Smart file size routing
9. Pre-computed static data caching
