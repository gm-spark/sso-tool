# SSO Tool Validation Strategy

> **Purpose**: Document file validation approach for the SSO Tool, covering both client-side and server-side validation.

---

## Overview

File validation happens in two stages:

1. **Client-side (Next.js)**: Instant feedback, preview, catch obvious errors
2. **Server-side (FastAPI + Pydantic)**: Authoritative validation, business rules, database cross-referencing

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FILE UPLOAD FLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User selects file                                               │
│     │                                                               │
│     ▼                                                               │
│  2. CLIENT-SIDE VALIDATION (instant, in browser)                    │
│     • Parse with SheetJS                                            │
│     • Check required columns exist                                  │
│     • Clean values (strip $, commas, etc.)                          │
│     • Show preview table                                            │
│     • Show warnings (X rows invalid)                                │
│     │                                                               │
│     ├── Invalid? → Show errors, don't upload                        │
│     │                                                               │
│     ▼                                                               │
│  3. User reviews preview, clicks "Upload"                           │
│     │                                                               │
│     ▼                                                               │
│  4. SERVER-SIDE VALIDATION (authoritative, Pydantic)                │
│     • Re-validate (never trust client)                              │
│     • Cross-reference with database                                 │
│     • Check business rules                                          │
│     │                                                               │
│     ├── Invalid? → Return detailed errors                           │
│     │                                                               │
│     ▼                                                               │
│  5. Store cleaned data, ready for calculation                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What to Validate Where

| Check | Client | Server |
|-------|--------|--------|
| File parseable | ✓ | ✓ |
| Required columns exist | ✓ | ✓ |
| Data types valid | ✓ | ✓ |
| Values cleaned (commas, $, etc.) | ✓ | ✓ |
| Cross-reference DB (item exists?) | ✗ | ✓ |
| Business rules (department matches?) | ✗ | ✓ |
| Duplicate detection | Basic | Full |
| Row-level error reporting | Basic | Detailed |

---

## Server-Side Validation with Pydantic

Pydantic integrates seamlessly with FastAPI and supports custom validators/parsers for complex field types.

### Custom Annotated Types

Using `Annotated` with `BeforeValidator` for automatic cleaning:

```python
from typing import Annotated
from pydantic import BaseModel, BeforeValidator, ValidationError
from pydantic.functional_validators import AfterValidator
import re

# ============================================================
# CUSTOM TYPE PARSERS
# ============================================================

def parse_numeric(value: any) -> float | None:
    """
    Parse numeric values, handling:
    - Commas: "1,234.56" → 1234.56
    - Whitespace: " 123 " → 123
    - Empty strings: "" → None
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).strip()
    if cleaned == "":
        return None

    # Remove commas and whitespace
    cleaned = re.sub(r"[,\s]", "", cleaned)

    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"Cannot parse '{value}' as numeric")


def parse_integer(value: any) -> int | None:
    """
    Parse integer values, handling:
    - Commas: "1,234" → 1234
    - Floats: "123.0" → 123
    - Whitespace
    """
    numeric = parse_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def parse_usd(value: any) -> float | None:
    """
    Parse USD currency values, handling:
    - Dollar signs: "$1,234.56" → 1234.56
    - Commas: "1,234.56" → 1234.56
    - Accounting negatives: "(1,234.56)" → -1234.56
    - Whitespace
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).strip()
    if cleaned == "":
        return None

    # Check for accounting negative format: (1,234.56)
    is_negative = cleaned.startswith("(") and cleaned.endswith(")")
    if is_negative:
        cleaned = cleaned[1:-1]

    # Remove $, commas, whitespace
    cleaned = re.sub(r"[$,\s]", "", cleaned)

    try:
        result = float(cleaned)
        return -result if is_negative else result
    except ValueError:
        raise ValueError(f"Cannot parse '{value}' as USD")


def parse_percentage(value: any) -> float | None:
    """
    Parse percentage values, handling:
    - Percent signs: "45.5%" → 0.455
    - Already decimal: "0.455" → 0.455
    - Commas, whitespace
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Assume already decimal if < 1, otherwise treat as percentage
        return float(value) if value <= 1 else float(value) / 100

    cleaned = str(value).strip()
    if cleaned == "":
        return None

    has_percent = "%" in cleaned
    cleaned = re.sub(r"[%,\s]", "", cleaned)

    try:
        result = float(cleaned)
        return result / 100 if has_percent or result > 1 else result
    except ValueError:
        raise ValueError(f"Cannot parse '{value}' as percentage")


# ============================================================
# ANNOTATED TYPES FOR PYDANTIC
# ============================================================

# Use these in your Pydantic models for automatic parsing
CleanedNumeric = Annotated[float | None, BeforeValidator(parse_numeric)]
CleanedInteger = Annotated[int | None, BeforeValidator(parse_integer)]
CleanedUSD = Annotated[float | None, BeforeValidator(parse_usd)]
CleanedPercentage = Annotated[float | None, BeforeValidator(parse_percentage)]

# Non-nullable versions (will fail if None)
RequiredNumeric = Annotated[float, BeforeValidator(parse_numeric)]
RequiredInteger = Annotated[int, BeforeValidator(parse_integer)]
RequiredUSD = Annotated[float, BeforeValidator(parse_usd)]
```

### Row Models

Define Pydantic models for each file type:

```python
from pydantic import BaseModel, Field, model_validator

class SupplierInventoryRow(BaseModel):
    """A single row from the Supplier Inventory file."""

    item_number: RequiredInteger = Field(
        ...,
        alias="WMT Prime Item Nbr",
        description="Walmart item number"
    )
    department: RequiredInteger = Field(
        ...,
        alias="Department Nbr",
        description="Department number"
    )
    on_hand: RequiredNumeric = Field(
        ...,
        alias="Supplier Has On Hand to Ship",
        ge=0,  # Must be >= 0
        description="Available inventory at supplier"
    )
    max_qty: CleanedNumeric = Field(
        None,
        alias="Max quantity we are willing to send right now",
        description="Optional cap on quantity to send"
    )

    class Config:
        populate_by_name = True  # Allow both alias and field name


class SSOSalesRow(BaseModel):
    """A single row from the SSO Sales file."""

    item_number: RequiredInteger = Field(..., alias="all_links_item_number")
    store_number: RequiredInteger = Field(..., alias="store_number")
    department: RequiredInteger = Field(..., alias="omni_department_number")

    on_hand_qty: CleanedNumeric = Field(0, alias="store_on_hand_quantity_this_year")
    in_transit_qty: CleanedNumeric = Field(0, alias="store_in_transit_quantity_this_year")
    in_warehouse_qty: CleanedNumeric = Field(0, alias="store_in_warehouse_quantity_this_year")
    on_order_qty: CleanedNumeric = Field(0, alias="store_on_order_quantity_this_year")

    warehouse_pack_qty: RequiredInteger = Field(..., alias="warehouse_pack_quantity")
    retail_amount: CleanedUSD = Field(None, alias="base_unit_retail_amount")
    traited_count: CleanedInteger = Field(0, alias="traited_store_count_this_year")


class DemandForecastRow(BaseModel):
    """A single row from the Demand Forecast file."""

    item_number: RequiredInteger = Field(..., alias="all_links_item_nbr")
    store_number: RequiredInteger = Field(..., alias="store_nbr")
    department: RequiredInteger = Field(..., alias="omni_dept_nbr")
    week: RequiredInteger = Field(..., alias="walmart_calendar_week")
    forecast_qty: RequiredNumeric = Field(..., alias="final_forecast_each_quantity", ge=0)
```

### File Validation Service

```python
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Generic, Type
import polars as pl

T = TypeVar("T", bound=BaseModel)


class RowError(BaseModel):
    """Error details for a single row."""
    row_number: int
    field: str
    value: any
    message: str


class ValidationResult(BaseModel, Generic[T]):
    """Result of validating a file."""
    valid: bool
    rows: list[T] = []
    row_count: int = 0
    valid_count: int = 0
    error_count: int = 0
    errors: list[RowError] = []
    warnings: list[str] = []


def validate_file(
    df: pl.DataFrame,
    row_model: Type[T],
    max_errors: int = 100,
) -> ValidationResult[T]:
    """
    Validate a DataFrame against a Pydantic model.

    Args:
        df: Polars DataFrame to validate
        row_model: Pydantic model class for row validation
        max_errors: Stop collecting errors after this many

    Returns:
        ValidationResult with valid rows and detailed errors
    """
    valid_rows: list[T] = []
    errors: list[RowError] = []

    # Convert to list of dicts for row-by-row validation
    rows = df.to_dicts()

    for i, row_data in enumerate(rows, start=2):  # Start at 2 (Excel row numbers)
        try:
            validated = row_model.model_validate(row_data)
            valid_rows.append(validated)
        except ValidationError as e:
            if len(errors) < max_errors:
                for error in e.errors():
                    field = error["loc"][0] if error["loc"] else "unknown"
                    errors.append(RowError(
                        row_number=i,
                        field=str(field),
                        value=row_data.get(str(field)),
                        message=error["msg"]
                    ))

    return ValidationResult(
        valid=len(errors) == 0,
        rows=valid_rows,
        row_count=len(rows),
        valid_count=len(valid_rows),
        error_count=len(rows) - len(valid_rows),
        errors=errors,
        warnings=[f"{len(rows) - len(valid_rows)} rows failed validation"] if errors else []
    )
```

### FastAPI Integration

```python
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import polars as pl
import io

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/supplier-inventory")
async def upload_supplier_inventory(
    file: UploadFile,
    department: int,  # Expected department for validation
):
    """
    Upload and validate supplier inventory file.

    Returns validation results with cleaned data or detailed errors.
    """
    # Read file
    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = pl.read_csv(io.BytesIO(contents))
        else:
            df = pl.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to parse file: {e}")

    # Validate with Pydantic
    result = validate_file(df, SupplierInventoryRow)

    if not result.valid:
        return JSONResponse(
            status_code=400,
            content={
                "message": "Validation failed",
                "valid_count": result.valid_count,
                "error_count": result.error_count,
                "errors": [e.model_dump() for e in result.errors[:20]],  # First 20 errors
            }
        )

    # Additional business rule: check department matches
    wrong_dept = [r for r in result.rows if r.department != department]
    if wrong_dept:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Found {len(wrong_dept)} rows with wrong department (expected {department})",
                "valid_count": result.valid_count - len(wrong_dept),
                "error_count": len(wrong_dept),
            }
        )

    # Success - return summary
    return {
        "message": "File validated successfully",
        "row_count": result.row_count,
        "valid_count": result.valid_count,
        "preview": [r.model_dump() for r in result.rows[:10]],
    }


@router.post("/sso-sales")
async def upload_sso_sales(file: UploadFile, department: int):
    """Upload and validate SSO Sales file."""
    contents = await file.read()

    try:
        df = pl.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to parse file: {e}")

    result = validate_file(df, SSOSalesRow)

    if not result.valid:
        return JSONResponse(status_code=400, content={
            "message": "Validation failed",
            "errors": [e.model_dump() for e in result.errors[:20]],
        })

    # Filter to requested department
    filtered = [r for r in result.rows if r.department == department]

    return {
        "message": "File validated successfully",
        "row_count": result.row_count,
        "valid_count": len(filtered),
        "filtered_out": result.valid_count - len(filtered),
    }
```

### Detailed Error Response Example

```json
{
  "message": "Validation failed",
  "valid_count": 142,
  "error_count": 3,
  "errors": [
    {
      "row_number": 15,
      "field": "Supplier Has On Hand to Ship",
      "value": "N/A",
      "message": "Cannot parse 'N/A' as numeric"
    },
    {
      "row_number": 28,
      "field": "WMT Prime Item Nbr",
      "value": "",
      "message": "Field required"
    },
    {
      "row_number": 45,
      "field": "Supplier Has On Hand to Ship",
      "value": "-50",
      "message": "Input should be greater than or equal to 0"
    }
  ]
}
```

---

## Client-Side Validation (Next.js)

Client-side validation provides instant feedback before upload.

### Cleaning Functions (TypeScript)

Mirror the Python cleaners for consistency:

```typescript
// lib/cleaners.ts

export const cleaners = {
  integer: (value: unknown): number | null => {
    if (value === null || value === undefined || String(value).trim() === '') {
      return null;
    }
    const cleaned = String(value).replace(/[,\s]/g, '');
    const num = parseInt(cleaned, 10);
    return isNaN(num) ? null : num;
  },

  numeric: (value: unknown): number | null => {
    if (value === null || value === undefined || String(value).trim() === '') {
      return null;
    }
    const cleaned = String(value).replace(/[,\s]/g, '');
    const num = parseFloat(cleaned);
    return isNaN(num) ? null : num;
  },

  usd: (value: unknown): number | null => {
    if (value === null || value === undefined || String(value).trim() === '') {
      return null;
    }
    let cleaned = String(value).trim();

    // Handle accounting negative format: (1,234.56)
    const isNegative = cleaned.startsWith('(') && cleaned.endsWith(')');
    if (isNegative) {
      cleaned = cleaned.slice(1, -1);
    }

    // Remove $, commas, whitespace
    cleaned = cleaned.replace(/[$,\s]/g, '');
    const num = parseFloat(cleaned);

    if (isNaN(num)) return null;
    return isNegative ? -num : num;
  },

  percentage: (value: unknown): number | null => {
    if (value === null || value === undefined || String(value).trim() === '') {
      return null;
    }
    const str = String(value).trim();
    const hasPercent = str.includes('%');
    const cleaned = str.replace(/[%,\s]/g, '');
    const num = parseFloat(cleaned);

    if (isNaN(num)) return null;
    return hasPercent || num > 1 ? num / 100 : num;
  },

  text: (value: unknown): string | null => {
    if (value === null || value === undefined) return null;
    const trimmed = String(value).trim();
    return trimmed === '' ? null : trimmed;
  },
};

export type CleanerType = keyof typeof cleaners;
```

### Schema Definition

```typescript
// lib/schemas.ts

export interface ColumnSchema {
  name: string;              // Column name in file
  type: CleanerType;         // Cleaner to apply
  required: boolean;         // Is this column required?
  renameTo?: string;         // Rename after cleaning
  validate?: (value: any) => boolean;  // Additional validation
  errorMessage?: string;     // Custom error message
}

export interface FileSchema {
  name: string;
  columns: ColumnSchema[];
}

export const SUPPLIER_INVENTORY_SCHEMA: FileSchema = {
  name: 'Supplier Inventory',
  columns: [
    {
      name: 'WMT Prime Item Nbr',
      type: 'integer',
      required: true,
      renameTo: 'item_number'
    },
    {
      name: 'Department Nbr',
      type: 'integer',
      required: true,
      renameTo: 'department'
    },
    {
      name: 'Supplier Has On Hand to Ship',
      type: 'numeric',
      required: true,
      renameTo: 'on_hand',
      validate: (v) => v >= 0,
      errorMessage: 'Must be non-negative'
    },
    {
      name: 'Max quantity we are willing to send right now',
      type: 'numeric',
      required: false,
      renameTo: 'max_qty'
    },
  ],
};

export const SSO_SALES_SCHEMA: FileSchema = {
  name: 'SSO Sales',
  columns: [
    { name: 'all_links_item_number', type: 'integer', required: true, renameTo: 'item_number' },
    { name: 'store_number', type: 'integer', required: true },
    { name: 'omni_department_number', type: 'integer', required: true, renameTo: 'department' },
    { name: 'store_on_hand_quantity_this_year', type: 'numeric', required: false, renameTo: 'on_hand_qty' },
    { name: 'store_in_transit_quantity_this_year', type: 'numeric', required: false, renameTo: 'in_transit_qty' },
    { name: 'store_in_warehouse_quantity_this_year', type: 'numeric', required: false, renameTo: 'in_warehouse_qty' },
    { name: 'store_on_order_quantity_this_year', type: 'numeric', required: false, renameTo: 'on_order_qty' },
    { name: 'warehouse_pack_quantity', type: 'integer', required: true, renameTo: 'warehouse_pack_qty' },
    { name: 'base_unit_retail_amount', type: 'usd', required: false, renameTo: 'retail_amount' },
    { name: 'traited_store_count_this_year', type: 'integer', required: false, renameTo: 'traited_count' },
  ],
};

export const DEMAND_FORECAST_SCHEMA: FileSchema = {
  name: 'Demand Forecast',
  columns: [
    { name: 'all_links_item_nbr', type: 'integer', required: true, renameTo: 'item_number' },
    { name: 'store_nbr', type: 'integer', required: true, renameTo: 'store_number' },
    { name: 'omni_dept_nbr', type: 'integer', required: true, renameTo: 'department' },
    { name: 'walmart_calendar_week', type: 'integer', required: true, renameTo: 'week' },
    { name: 'final_forecast_each_quantity', type: 'numeric', required: true, renameTo: 'forecast_qty' },
  ],
};
```

### Validation Hook

```typescript
// hooks/useFileValidator.ts

import { useState, useCallback } from 'react';
import * as XLSX from 'xlsx';
import { cleaners, CleanerType } from '@/lib/cleaners';
import { FileSchema, ColumnSchema } from '@/lib/schemas';

export interface RowError {
  rowNumber: number;
  field: string;
  value: unknown;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  data: Record<string, unknown>[] | null;
  errors: RowError[];
  warnings: string[];
  preview: Record<string, unknown>[];
  rowCount: number;
  validCount: number;
}

export function useFileValidator(schema: FileSchema) {
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  const validate = useCallback(async (file: File): Promise<ValidationResult> => {
    setIsValidating(true);

    try {
      // Parse file with SheetJS
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer);
      const sheetName = workbook.SheetNames[0];
      const rawData = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);

      const errors: RowError[] = [];
      const warnings: string[] = [];

      // Check required columns exist
      const fileColumns = Object.keys(rawData[0] || {});
      const requiredCols = schema.columns
        .filter((c) => c.required)
        .map((c) => c.name);
      const missingCols = requiredCols.filter((c) => !fileColumns.includes(c));

      if (missingCols.length > 0) {
        const result: ValidationResult = {
          valid: false,
          data: null,
          errors: [{
            rowNumber: 0,
            field: 'columns',
            value: fileColumns,
            message: `Missing required columns: ${missingCols.join(', ')}`,
          }],
          warnings: [],
          preview: [],
          rowCount: rawData.length,
          validCount: 0,
        };
        setResult(result);
        return result;
      }

      // Clean and validate each row
      const cleanedData: Record<string, unknown>[] = [];

      for (let i = 0; i < rawData.length; i++) {
        const row = rawData[i] as Record<string, unknown>;
        const cleanedRow: Record<string, unknown> = {};
        let rowValid = true;

        for (const col of schema.columns) {
          if (!(col.name in row)) {
            if (col.required) {
              errors.push({
                rowNumber: i + 2,  // Excel row number (1-indexed + header)
                field: col.name,
                value: undefined,
                message: 'Required field missing',
              });
              rowValid = false;
            }
            continue;
          }

          const rawValue = row[col.name];
          const cleaner = cleaners[col.type];
          const cleanedValue = cleaner(rawValue);

          // Check if required field is null after cleaning
          if (cleanedValue === null && col.required) {
            errors.push({
              rowNumber: i + 2,
              field: col.name,
              value: rawValue,
              message: `Cannot parse as ${col.type}`,
            });
            rowValid = false;
            continue;
          }

          // Run custom validation if provided
          if (col.validate && cleanedValue !== null && !col.validate(cleanedValue)) {
            errors.push({
              rowNumber: i + 2,
              field: col.name,
              value: rawValue,
              message: col.errorMessage || 'Validation failed',
            });
            rowValid = false;
            continue;
          }

          const finalName = col.renameTo || col.name;
          cleanedRow[finalName] = cleanedValue;
        }

        if (rowValid) {
          cleanedData.push(cleanedRow);
        }
      }

      const invalidCount = rawData.length - cleanedData.length;
      if (invalidCount > 0) {
        warnings.push(`${invalidCount} rows had invalid data and were excluded`);
      }

      const validationResult: ValidationResult = {
        valid: errors.length === 0 && cleanedData.length > 0,
        data: cleanedData,
        errors: errors.slice(0, 50),  // Limit errors returned
        warnings,
        preview: cleanedData.slice(0, 10),
        rowCount: rawData.length,
        validCount: cleanedData.length,
      };

      setResult(validationResult);
      return validationResult;

    } catch (e) {
      const errorResult: ValidationResult = {
        valid: false,
        data: null,
        errors: [{
          rowNumber: 0,
          field: 'file',
          value: null,
          message: `Failed to parse file: ${e}`,
        }],
        warnings: [],
        preview: [],
        rowCount: 0,
        validCount: 0,
      };
      setResult(errorResult);
      return errorResult;
    } finally {
      setIsValidating(false);
    }
  }, [schema]);

  const reset = useCallback(() => {
    setResult(null);
  }, []);

  return { validate, result, isValidating, reset };
}
```

### File Upload Component

```typescript
// components/FileUploader.tsx

'use client';

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useFileValidator, ValidationResult } from '@/hooks/useFileValidator';
import { FileSchema } from '@/lib/schemas';

interface FileUploaderProps {
  schema: FileSchema;
  onValidated: (data: Record<string, unknown>[]) => void;
}

export function FileUploader({ schema, onValidated }: FileUploaderProps) {
  const { validate, result, isValidating, reset } = useFileValidator(schema);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const validationResult = await validate(file);

    if (validationResult.valid && validationResult.data) {
      onValidated(validationResult.data);
    }
  }, [validate, onValidated]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
  });

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
          ${result?.valid ? 'border-green-500 bg-green-50' : ''}
          ${result && !result.valid ? 'border-red-500 bg-red-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        <p className="text-gray-600">
          {isDragActive
            ? 'Drop the file here...'
            : `Drag & drop ${schema.name} file, or click to select`}
        </p>
        <p className="text-sm text-gray-400 mt-2">
          Accepts .xlsx, .xls, .csv
        </p>
      </div>

      {/* Loading */}
      {isValidating && (
        <div className="text-blue-600">Validating...</div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Status */}
          <div className={`p-4 rounded ${result.valid ? 'bg-green-100' : 'bg-red-100'}`}>
            {result.valid ? (
              <p className="text-green-800">
                ✓ Valid: {result.validCount} of {result.rowCount} rows
              </p>
            ) : (
              <p className="text-red-800">
                ✗ Invalid: {result.errors.length} errors found
              </p>
            )}
          </div>

          {/* Warnings */}
          {result.warnings.map((warning, i) => (
            <p key={i} className="text-yellow-700 bg-yellow-50 p-2 rounded">
              ⚠ {warning}
            </p>
          ))}

          {/* Errors */}
          {result.errors.length > 0 && (
            <div className="bg-red-50 p-4 rounded max-h-60 overflow-auto">
              <h4 className="font-semibold text-red-800 mb-2">Errors:</h4>
              <ul className="text-sm space-y-1">
                {result.errors.map((err, i) => (
                  <li key={i} className="text-red-700">
                    Row {err.rowNumber}: {err.field} - {err.message}
                    {err.value !== undefined && (
                      <span className="text-gray-500"> (value: "{String(err.value)}")</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Preview */}
          {result.preview.length > 0 && (
            <div>
              <h4 className="font-semibold mb-2">Preview (first 10 rows, cleaned):</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm border">
                  <thead className="bg-gray-100">
                    <tr>
                      {Object.keys(result.preview[0]).map((col) => (
                        <th key={col} className="px-3 py-2 text-left border-b">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="px-3 py-2 border-b">
                            {val === null ? <span className="text-gray-400">null</span> : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Reset button */}
          <button
            onClick={reset}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            Clear and upload different file
          </button>
        </div>
      )}
    </div>
  );
}
```

---

## Shared Type Definitions

Keep client and server in sync with shared types:

```typescript
// shared/types.ts (or generate from Pydantic models)

export interface SupplierInventoryRow {
  item_number: number;
  department: number;
  on_hand: number;
  max_qty: number | null;
}

export interface SSOSalesRow {
  item_number: number;
  store_number: number;
  department: number;
  on_hand_qty: number;
  in_transit_qty: number;
  in_warehouse_qty: number;
  on_order_qty: number;
  warehouse_pack_qty: number;
  retail_amount: number | null;
  traited_count: number;
}

export interface DemandForecastRow {
  item_number: number;
  store_number: number;
  department: number;
  week: number;
  forecast_qty: number;
}
```

---

## Summary

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Client | SheetJS + TypeScript | Parse files, instant validation, preview |
| Server | Pydantic + FastAPI | Authoritative validation, business rules |
| Shared | Type definitions | Keep client/server in sync |

### Benefits of This Approach

1. **Pydantic integration**: Custom validators play nicely with FastAPI
2. **Automatic cleaning**: `BeforeValidator` handles parsing before validation
3. **Detailed errors**: Row-level error reporting with values
4. **Consistent logic**: Same cleaning rules on client and server
5. **Type safety**: Full TypeScript/Python type hints
6. **Preview before upload**: Users see cleaned data before submitting
