# SSO Tool Test Data Analysis

> **Purpose**: Map available test data by vendor to identify suitable candidates for testing the SSO Tool conversion.

---

## Summary

**13 vendors** identified in the system. Data availability varies significantly:

| Status | Count | Vendors |
|--------|-------|---------|
| Has Outputs | 12 | All except NDW |
| Has Manual Input | 12 | All except NDW |
| Has Input Data Files | 1 | Sakar only |

**Key Finding**: Most vendor input files (Store Sales, SSO Sales, Demand Forecast) are **not synced** to this Dropbox folder. Only Sakar has SSO Sales data available. The other input files will need to be generated/provided for testing.

---

## Vendors Ranked by Output Size (Smallest to Largest)

Smaller vendors are better for initial testing due to faster iteration.

| Rank | Vendor | Output Size | NOVA Files | Rollup Files | Recommendation |
|------|--------|-------------|------------|--------------|----------------|
| 1 | **PSI** | 248 KB | 25 | 1 | Best for initial testing |
| 2 | **Todson** | 604 KB | 12 | 1 | Good for testing |
| 3 | **Shubug** | 717 KB | 46 | 2 | Good for testing |
| 4 | **Rashti** | 2.4 MB | 64 | 2 | Good for testing |
| 5 | **Paris** | 2.6 MB | 146 | 9 | Medium |
| 6 | **Chillafish** | 4.6 MB | 83 | 4 | Medium |
| 7 | **Brandix** | 19.6 MB | 62 | 4 | Medium |
| 8 | **iFabric** | 20.6 MB | 271 | 13 | Medium-Large |
| 9 | **DoctorsChoice** | 22.0 MB | 137 | 11 | Medium-Large |
| 10 | **Flynn** | 138.7 MB | 66 | 7 | Large |
| 11 | **Sakar** | 327.2 MB | 1,215 | 68 | Very Large (has input data!) |
| 12 | **NDW** | 0 KB | 0 | 0 | No data |

---

## Last Run Dates by Vendor

Based on most recent output file timestamps:

| Vendor | Last Run Date | WMT Week | Department |
|--------|---------------|----------|------------|
| DoctorsChoice | 2026-01-06 | 202549 | Dept 29 |
| Sakar | 2025-12-16 | 202546 | Dept 7 |
| iFabric | 2025-12-02 | 202544 | Dept 29 |
| Flynn | 2025-11-21 | 202542 | Dept 23 |
| Todson | 2025-11-20 | 202542 | Dept 7 |
| Chillafish | 2025-10-07 | 202536 | Dept 7 |
| Brandix | 2025-08-14 | 202528 | Dept 29 |
| Shubug | 2025-07-08 | 202523 | Dept 56 |
| Paris | 2025-05-29 | 202517 | Dept 79 |
| Rashti | 2025-05-29 | 202517 | Dept 79 |
| PSI | 2025-04-04 | 202509 | Dept 7 |
| NDW | - | - | - |

---

## Manual Input Files (Supplier Available Inventory)

These files are manually created by analysts before each run.

| Vendor | File | Size | Last Modified | Notes |
|--------|------|------|---------------|-------|
| DoctorsChoice | DoctorsChoice Available at Supplier.xlsx | 10.5 KB | 2026-01-06 | Most recent |
| DoctorsChoice | DoctorsChoice Available at Supplier 2.xlsx | 10.5 KB | 2025-11-26 | Backup version |
| iFabric | iFabric Available at Supplier.xlsx | 11.5 KB | 2025-12-02 | |
| iFabric | iFabric Available at Supplier 2.xlsx | 11.3 KB | 2025-11-11 | |
| iFabric | iFabric Available at Supplier saved.xlsx | 11.1 KB | 2025-02-12 | Old backup |
| Sakar | Sakar Available at Supplier.xlsx | 10.4 KB | 2025-12-16 | |
| Flynn | Flynn Available at Supplier.xlsx | 11.9 KB | 2025-11-21 | |
| Todson | Todson Available at Supplier.xlsx | 10.4 KB | 2025-11-20 | |
| Chillafish | Chillafish Available at Supplier.xlsx | 10.2 KB | 2025-10-07 | |
| Brandix | Brandix Available at Supplier.xlsx | 10.8 KB | 2025-08-14 | |
| Shubug | Shubug Available at Supplier.xlsx | 10.3 KB | 2025-07-08 | |
| Paris | Paris Available at Supplier.xlsx | 10.3 KB | 2025-05-28 | |
| Rashti | Rashti Available at Supplier.xlsx | 10.3 KB | 2025-05-29 | |
| PSI | PSI Available at Supplier.xlsx | 10.4 KB | 2025-04-04 | |
| PSI | PSI Available at Supplier saved.xlsx | 10.5 KB | 2025-04-03 | Backup |

---

## Available Input Data Files

Only one vendor has input data available in this sync:

### Sakar (Only Complete Dataset)

| File | Size | Rows | Date |
|------|------|------|------|
| Sakar SSO Sales Prod.csv | 56.1 MB | 468,263 | 2024-06-25 |
| Sakar Available at Supplier.xlsx | 10.4 KB | - | 2025-12-16 |

**Missing for Sakar:**
- Store Sales Reduced L10W.csv
- Store Items Prod.xlsx
- Demand Forecast Prod.csv

**Missing for All Other Vendors:**
- SSO Sales Prod.csv
- Store Sales Reduced L10W.csv
- Store Items Prod.xlsx
- Demand Forecast Prod.csv

---

## Template Files

| File | Size | Location |
|------|------|----------|
| NOVA template.xlsx | 8.8 KB | /dropbox/Templates/ |
| NOVA template IDCs.xlsx | 8.8 KB | /dropbox/Templates/ |

---

## Testing Recommendations

### Tier 1: Best for Initial Development (Smallest)
Start with these for fast iteration:

| Vendor | Why |
|--------|-----|
| **PSI** | Smallest output (248 KB), simple structure |
| **Todson** | Small (604 KB), recent run (Nov 2025) |
| **Shubug** | Small (717 KB), variety in WOS targets |

### Tier 2: Medium Complexity
Use after basic functionality works:

| Vendor | Why |
|--------|-----|
| **Rashti** | 2.4 MB, good middle ground |
| **Paris** | 2.6 MB, many output files (146 NOVA) |
| **Chillafish** | 4.6 MB, different department (Dept 7) |

### Tier 3: Production-Scale Testing
Use for performance testing:

| Vendor | Why |
|--------|-----|
| **Sakar** | 327 MB output, **has SSO Sales input data** |
| **Flynn** | 139 MB output, large scale |
| **DoctorsChoice** | 22 MB, most recent run (Jan 2026) |

---

## Data Requirements for Testing

To test the conversion, we need to obtain or generate:

### For Small Vendor Testing (PSI, Todson, Shubug)

| File Type | Status | Action Needed |
|-----------|--------|---------------|
| Manual Input (Supplier Inventory) | Available | Use existing files |
| SSO Sales Prod.csv | Missing | Need to generate/export |
| Store Sales Reduced L10W.csv | Missing | Need to generate/export |
| Store Items Prod.xlsx | Missing | Need to generate/export |
| Demand Forecast Prod.csv | Missing | Need to generate/export |
| NOVA Template | Available | Use existing template |

### For Sakar (Full-Scale Testing)

| File Type | Status | Action Needed |
|-----------|--------|---------------|
| Manual Input (Supplier Inventory) | Available | Use existing file |
| SSO Sales Prod.csv | **Available** | 56 MB, 468K rows |
| Store Sales Reduced L10W.csv | Missing | Need to generate |
| Store Items Prod.xlsx | Missing | Need to generate |
| Demand Forecast Prod.csv | Missing | Need to generate |
| NOVA Template | Available | Use existing template |

---

## Department Distribution

Different vendors use different Walmart departments:

| Department | Vendors |
|------------|---------|
| Dept 7 | Chillafish, PSI, Sakar, Todson |
| Dept 23 | Flynn |
| Dept 29 | Brandix, DoctorsChoice, iFabric |
| Dept 31 | Paris |
| Dept 56 | Shubug |
| Dept 79 | Paris, Rashti |

---

## Output File Structure Example

Sample from DoctorsChoice (most recent run):

```
Output/DoctorsChoice/
├── NOVA format/
│   ├── 202549_DoctorsChoice_Dept29_1WOS.xlsx
│   ├── 202549_DoctorsChoice_Dept29_2WOS.xlsx
│   ├── ...
│   └── 202549_DoctorsChoice_Dept29_13WOS.xlsx
│
└── Rollup format/
    └── 202549_DoctorsChoice_Dept29_all_runs_summary.xlsx
```

Filename pattern: `{WMT_Week}_{Supplier}_Dept{XX}_{Y}WOS.xlsx`

---

## Next Steps

1. **Obtain input files** for a small vendor (PSI or Todson recommended)
2. **Run original Python script** to generate expected outputs for comparison
3. **Set up validation tests** comparing new implementation vs original outputs
4. **Scale up testing** to Sakar once basic validation passes

---

## Folder Structure Reference

```
/workspace/dropbox/
├── Manual Inputs/
│   ├── {Vendor} Available at Supplier.xlsx  (per vendor)
│   └── Import Push Inputs/
│
├── Output/
│   └── {Vendor}/
│       ├── NOVA format/
│       │   └── {week}_{vendor}_Dept{X}_{Y}WOS.xlsx
│       └── Rollup format/
│           └── {week}_{vendor}_Dept{X}_all_runs_summary.xlsx
│
├── System/
│   ├── Datasets/
│   │   └── Sakar SSO Sales Prod.csv  (only input file available)
│   └── WOS cache/
│
└── Templates/
    ├── NOVA template.xlsx
    └── NOVA template IDCs.xlsx
```
