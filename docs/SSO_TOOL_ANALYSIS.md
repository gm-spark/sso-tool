# SSO Tool Analysis

> **Purpose**: This document analyzes the original `SSO_tool_dev.py` script to guide the conversion to a web-based application.

---

## Overview

The **Store Specific Orders (SSO) Tool** calculates optimal store replenishment orders for Walmart suppliers. It determines how much inventory to send to each store based on:

- Historical sales data
- Demand forecasts
- Current store pipeline (on-hand, in-transit, in-warehouse, on-order)
- Available supplier inventory

The output is a "NOVA format" Excel file used for Walmart's distribution system.

---

## Input Files

### 1. WMT Calendar (`WMT weeks.xlsx`)
**Location**: `System/`
**Purpose**: Maps dates to Walmart fiscal weeks
**Used For**: Determining current week, historical lookback periods, forecast periods

| Column | Type | Description |
|--------|------|-------------|
| `Date` | date | Calendar date |
| `WMT_week` | int | Walmart week ID (e.g., 202504) |
| `Year` | int | Fiscal year |
| `Week` | int | Week number within year |
| `Half` | int | First or second half of year |

---

### 2. Store Sales (`{Supplier} Store Sales Reduced L10W.csv`)
**Location**: `System/Vendors/{Supplier}/`
**Purpose**: Historical point-of-sale data by store and item
**Used For**: Calculating average weekly sales per store-item combination

| Column | Type | Description |
|--------|------|-------------|
| `all_links_item_number` | int | Walmart item ID |
| `store_number` | int | Store ID |
| `omni_department_number` | int | Department number |
| `pos_sales_this_year` | float | Dollar sales |
| `pos_quantity_this_year` | float | Unit sales |
| `walmart_calendar_week` | int | Week of the sales |

---

### 3. Store Items (`{Supplier} Store Items Prod.xlsx`)
**Location**: `System/Vendors/{Supplier}/`
**Purpose**: Item master data
**Used For**: Item names and attributes (currently minimally used)

| Column | Type | Description |
|--------|------|-------------|
| `all_links_item_number` | int | Walmart item ID |
| `item_name` | str | Item description |

---

### 4. SSO Sales Data (`{Supplier} SSO Sales Prod.csv`)
**Location**: `System/Vendors/{Supplier}/`
**Purpose**: Current store inventory pipeline and item attributes
**Used For**: Calculating current inventory position and warehouse pack sizes

| Column | Type | Description |
|--------|------|-------------|
| `all_links_item_number` | int | Walmart item ID |
| `store_number` | int | Store ID |
| `omni_department_number` | int | Department number |
| `store_on_hand_quantity_this_year` | float | Current store inventory |
| `store_in_transit_quantity_this_year` | float | In-transit to store |
| `store_in_warehouse_quantity_this_year` | float | At warehouse for store |
| `store_on_order_quantity_this_year` | float | On order for store |
| `warehouse_pack_quantity` | int | Units per warehouse pack |
| `base_unit_retail_amount` | str | Retail price (as string with $) |
| `traited_store_count_this_year` | int | Number of stores carrying item |

---

### 5. Demand Forecast (`{Supplier} Demand Forecast Prod.csv`)
**Location**: `System/Vendors/{Supplier}/`
**Purpose**: Predicted future sales by store and item
**Used For**: Forward-looking demand (optional, controlled by `forecast_trigger`)

| Column | Type | Description |
|--------|------|-------------|
| `all_links_item_nbr` | int | Walmart item ID |
| `store_nbr` | int | Store ID |
| `omni_dept_nbr` | int | Department number |
| `walmart_calendar_week` | int | Forecast week |
| `final_forecast_each_quantity` | float | Forecasted unit sales |

---

### 6. Supplier Inventory (`{Supplier} Available at Supplier.xlsx`)
**Location**: `Manual Inputs/`
**Purpose**: Current supplier on-hand inventory available to ship
**Used For**: Constraining orders to available supply
**Note**: This is a **manual input** file prepared by analysts

| Column | Type | Description |
|--------|------|-------------|
| `WMT Prime Item Nbr` | int | Walmart item ID |
| `Department Nbr` | int | Department number |
| `Supplier Has On Hand to Ship` | float | Available inventory |
| `Max quantity we are willing to send right now` | float | Optional cap on shipments |

---

### 7. NOVA Template (`NOVA template.xlsx`)
**Location**: `Templates/`
**Purpose**: Excel template for output format
**Used For**: Generating properly formatted distribution files

---

## Output Files

### 1. NOVA Distribution Files
**Location**: `Output/{Supplier}/NOVA format/`
**Filename**: `{WMT_week}_{Supplier}_Dept{XX}_{Y}WOS.xlsx`
**Purpose**: Store-level distribution instructions

| Column | Description |
|--------|-------------|
| `all_links_item_number` | Item ID |
| `store_number` | Store ID |
| `blank` | Empty column (template requirement) |
| `Whse Packs Distro` | Warehouse packs to send |
| `Eaches Distro` | Units to send |
| `Retail_amount_sent` | Retail value of shipment |
| `week_variable` | WOS target used |

---

### 2. Rollup Summary
**Location**: `Output/{Supplier}/Rollup format/`
**Filename**: `{WMT_week}_{Supplier}_Dept{XX}_all_runs_summary.xlsx`
**Purpose**: Aggregated summary across all WOS targets

| Column | Description |
|--------|-------------|
| `all_links_item_number` | Item ID |
| `from which WOS distro` | WOS target |
| `variable` | Metric name |
| `value` | Metric value |

---

## Configuration Parameters

```python
# Supplier/Department Selection
Supplier = 'DoctorsChoice'          # Vendor name
Department = 29                      # Walmart department number

# WOS (Weeks of Supply) Targets
WOS_targets = list(range(1,14))     # Calculate for 1-13 weeks of supply

# Forecast Settings
forecast_trigger = 'on'              # 'on' = use forecast data
forecast_only_trigger = 'off'        # 'on' = ignore historical, only forecast

# Time Windows
historical_weeks_to_look_back = 4    # Weeks of history for avg sales
forecast_end_week_weeks_out = 4      # Weeks of forecast to include

# Store Exclusions
FC_stores_to_remove = [747, 2360, ...] # Fulfillment centers to exclude
```

---

## Processing Logic

### Step 1: Determine Date Parameters
```
Load WMT calendar
Find current week based on today's date
Calculate:
  - LW (last week)
  - L4W (last 4 weeks)
  - lookback_week_list (weeks for historical average)
  - forecast_week_list (upcoming weeks for forecast)
```

### Step 2: Calculate Average Store-Item Sales
```
Load store sales data
Filter to department
Filter to lookback weeks
Group by (item, store)
Calculate mean sales quantity
```

### Step 3: Load Supplier Available Inventory
```
Load manual input file
Filter to department
Remove duplicates (keep last)
Convert to numeric, drop invalid
Calculate OH Available = Max willing OR On hand
Filter to items with positive availability
```

### Step 4: Prepare SSO Base Data
```
Load SSO sales data
Remove fulfillment center stores
Filter to traited stores only
Remove duplicates (keep highest on-hand)
Merge with average sales
Fill missing averages with 0
```

### Step 5: Incorporate Demand Forecast (Optional)
```
If forecast_trigger == 'on':
  Load forecast data
  Filter to department and forecast weeks
  Calculate average weekly forecast per store-item
  Merge with SSO data
  Use MAX(historical_avg, forecast_avg) as demand signal
```

### Step 6: Calculate Store Pipeline
```
F4 Pipe = on_hand + in_transit + in_warehouse + on_order
```

### Step 7: Calculate Need per WOS Target
```
For each WOS target (1-13):
  WOS_target_qty = avg_sales × WOS_target
  WOS_target_qty = MIN(4, WOS_target_qty)  # Cap at 4 units
  need = WOS_target_qty - F4_Pipe
  order_packs = round_to_warehouse_pack(need)
  order_eaches = order_packs × warehouse_pack_qty

  # Override: if pipe <= 1, send at least 1 pack
  if F4_Pipe <= 1:
    order_packs = MAX(order_packs, 1)
```

### Step 8: Check Against Supplier Availability
```
Group orders by item
Calculate total eaches needed per item
Compare to supplier available
Calculate surplus = available - needed
Flag items where surplus > 0 (can fulfill)
```

### Step 9: Generate Outputs
```
For each WOS target:
  Filter to items with positive surplus
  Filter to stores receiving product
  Calculate retail value
  Write to NOVA template
  Save as Excel file

Combine all rollups
Pivot to long format
Save summary Excel
```

---

## Key Functions to Create

### Core Calculation Functions

#### `calculate_order_packs(need: float, whse_pack: int) -> int`
Rounds need to warehouse pack quantities with special rounding rules.
```python
# Current logic:
# ratio < 0.5      → 0 packs
# 0.5 <= ratio < 1 → 1 pack
# ratio >= 1       → round(ratio)
```

#### `calculate_wos_target(avg_sales: float, wos_weeks: int, cap: int = 4) -> float`
Calculates target inventory level.
```python
# target = min(avg_sales * wos_weeks, cap)
```

#### `calculate_pipeline(on_hand: float, in_transit: float, in_warehouse: float, on_order: float) -> float`
Sums all pipeline components.
```python
# pipeline = on_hand + in_transit + in_warehouse + on_order
```

#### `calculate_need(target: float, pipeline: float) -> float`
Determines replenishment need.
```python
# need = target - pipeline
```

#### `apply_pack_override(order_packs: int, pipeline: float, threshold: int = 1) -> int`
Ensures minimum shipment for low-inventory stores.
```python
# if pipeline <= threshold: return max(order_packs, 1)
```

---

### Data Processing Functions

#### `load_wmt_calendar(source: str) -> WMTCalendar`
Loads and processes Walmart calendar data.
Returns object with methods for week lookups.

#### `get_lookback_weeks(calendar: WMTCalendar, current_week: int, num_weeks: int) -> list[int]`
Returns list of historical weeks for averaging.

#### `get_forecast_weeks(calendar: WMTCalendar, current_week: int, num_weeks: int) -> list[int]`
Returns list of future weeks for forecasting.

#### `calculate_avg_store_sales(sales_data: DataFrame, weeks: list[int]) -> DataFrame`
Aggregates sales by store-item for specified weeks.

#### `load_supplier_inventory(file: UploadFile, department: int) -> DataFrame`
Validates and loads manual supplier inventory input.

#### `merge_forecast_data(sso_data: DataFrame, forecast: DataFrame) -> DataFrame`
Incorporates forecast, using max of historical vs forecast.

#### `calculate_surplus(orders: DataFrame, supplier_inventory: DataFrame) -> DataFrame`
Compares order totals to available supply.

---

### File Handling Functions

#### `validate_upload(file: UploadFile, file_type: str) -> ValidationResult`
Checks uploaded file for required columns, data types, and business rules.

#### `parse_csv(file: UploadFile) -> DataFrame`
Reads CSV with appropriate settings.

#### `parse_excel(file: UploadFile, sheet_name: str = None) -> DataFrame`
Reads Excel with appropriate settings.

#### `generate_nova_output(data: DataFrame, template_path: str) -> bytes`
Creates NOVA-format Excel file from calculation results.

#### `generate_rollup_summary(runs: dict[int, DataFrame]) -> bytes`
Creates summary Excel across all WOS targets.

---

### Database Functions

#### `fetch_store_sales(supplier: str, department: int, weeks: list[int]) -> DataFrame`
Queries store sales from database (replaces file upload).

#### `save_run_results(run_id: str, results: RunResults) -> None`
Persists calculation results to database.

#### `get_run_history(user_id: str, limit: int = 50) -> list[Run]`
Retrieves past runs for display.

> **Note**: SSO Sales and Demand Forecast are user uploads (not yet in database).

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUTS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐ │
│  │ WMT Calendar│  │ Store Sales │  │  SSO Sales  │  │Demand Forecast│ │
│  │  (static)   │  │ (database)  │  │(user upload)│  │ (user upload) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘ │
│         │                │                │                  │         │
│         ▼                ▼                ▼                  ▼         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    DATA PREPARATION                              │  │
│  │  • Determine current week and date ranges                       │  │
│  │  • Calculate average store-item sales                           │  │
│  │  • Merge SSO data with sales averages                          │  │
│  │  • Incorporate forecast (max of historical vs forecast)         │  │
│  └─────────────────────────────┬───────────────────────────────────┘  │
│                                │                                       │
│  ┌─────────────────┐          │                                       │
│  │ Supplier Avail. │          │                                       │
│  │ (user upload)   │          │                                       │
│  └────────┬────────┘          │                                       │
│           │                   │                                       │
│           ▼                   ▼                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                           CALCULATION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  FOR EACH WOS TARGET (1-13):                                    │  │
│  │                                                                  │  │
│  │    1. Calculate target inventory (avg_sales × WOS, capped at 4) │  │
│  │    2. Calculate need (target - pipeline)                        │  │
│  │    3. Round to warehouse packs                                   │  │
│  │    4. Apply minimum pack override                                │  │
│  │    5. Calculate eaches and retail value                         │  │
│  │                                                                  │  │
│  └─────────────────────────────┬───────────────────────────────────┘  │
│                                │                                       │
│                                ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  SURPLUS CALCULATION:                                           │  │
│  │                                                                  │  │
│  │    1. Group orders by item                                       │  │
│  │    2. Sum total eaches per item                                 │  │
│  │    3. Compare to supplier available                             │  │
│  │    4. Filter to items with positive surplus                     │  │
│  │                                                                  │  │
│  └─────────────────────────────┬───────────────────────────────────┘  │
│                                │                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                             OUTPUTS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                │                                       │
│                                ▼                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐│
│  │ NOVA Files       │  │ Rollup Summary   │  │ Database Records     ││
│  │ (per WOS target) │  │ (all targets)    │  │ (run history)        ││
│  │                  │  │                  │  │                      ││
│  │ • Item           │  │ • Item           │  │ • Run metadata       ││
│  │ • Store          │  │ • Store count    │  │ • Parameters used    ││
│  │ • Packs to send  │  │ • Total packs    │  │ • Results summary    ││
│  │ • Eaches to send │  │ • Total eaches   │  │ • Output file links  ││
│  │ • Retail value   │  │ • Total retail   │  │                      ││
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Business Rules & Edge Cases

### Rounding Rules
- Order packs use custom rounding (not standard):
  - `< 0.5 packs` → 0
  - `0.5 to < 1 pack` → 1
  - `>= 1 pack` → standard rounding

### Caps & Limits
- WOS target is capped at 4 units max (hardcoded)
- Minimum 1 pack sent if current pipeline ≤ 1

### Exclusions
- Fulfillment center stores are excluded (hardcoded list)
- Stores with 0 traited count are excluded
- Items with 0 or negative supplier availability are excluded

### Duplicates
- SSO data: Keep row with highest on-hand quantity
- Supplier inventory: Keep last occurrence
- Retails: Keep first occurrence

### Missing Data
- Missing average sales → 0
- Missing forecast → 0
- Use MAX(historical, forecast) as demand signal

---

## Validation Rules for User Uploads

### Supplier Inventory File
```yaml
required_columns:
  - WMT Prime Item Nbr
  - Department Nbr
  - Supplier Has On Hand to Ship

optional_columns:
  - Max quantity we are willing to send right now

validations:
  - WMT Prime Item Nbr must be integer
  - Department Nbr must match selected department
  - Supplier Has On Hand to Ship must be numeric
  - At least one row with positive inventory

warnings:
  - Items not found in SSO data will be ignored
  - Negative quantities will be treated as 0
```

---

## Notes for Conversion

### Hardcoded Values to Make Configurable
1. `FC_stores_to_remove` list → Store in database, make editable
2. WOS cap of 4 → Make configurable per supplier
3. Minimum pack override threshold (1) → Make configurable
4. File paths → Replace with database queries / uploads

### Performance Considerations
1. Store sales aggregation is expensive for large suppliers
2. The 13 WOS target loop creates 13 copies of data
3. Consider calculating all WOS targets in a single pass
4. Use Polars for memory efficiency on large files

### Potential Improvements
1. Add validation before calculation (fail fast)
2. Add progress tracking for long calculations
3. Add dry-run mode (preview without saving)
4. Add comparison mode (compare two runs side-by-side)
