# Using the `tax_filer` Parameter in OG-Core

## Overview

The `tax_filer` parameter allows you to model income tax non-filers in OG-Core. This feature is useful for analyzing:

- **Filing thresholds**: Model the effects of standard deductions and filing requirements
- **Tax compliance**: Study the impact of tax filing policies
- **Low-income tax treatment**: Analyze economic effects when low-income groups face no income tax

## How It Works

Non-filers in OG-Core:
- Pay **zero income tax** (income tax liability = 0)
- Face **zero marginal tax rates** on both labor and capital income
- Still pay **payroll taxes** (Social Security and Medicare)
- Experience no tax distortions on labor supply and savings decisions

This is economically consistent: both average tax rates (ATR) and marginal tax rates (MTR) are zero for non-filers.

## Parameter Specification

The `tax_filer` parameter is a J-length vector where each element represents the filing status of lifetime income group j:

- **`tax_filer[j] = 0.0`**: Non-filer (no income tax, zero MTRs)
- **`tax_filer[j] = 1.0`**: Full filer (normal income tax treatment)
- **`tax_filer[j] = 0.5`**: Partial filer (50% of the group files, 50% scaling of taxes and MTRs)

### Default Value
```python
tax_filer = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # All groups are filers
```

## Example Usage

### Example 1: Lowest Income Group as Non-Filers

```python
from ogcore.parameters import Specifications

# Create specifications object
p = Specifications()

# Set lowest income group (j=0) as non-filers
p.update_specifications({
    "tax_filer": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
})

# j=0 now pays zero income tax and faces zero MTRs
```

### Example 2: Multiple Non-Filer Groups

```python
# Set first two income groups as non-filers
p.update_specifications({
    "tax_filer": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
})
```

### Example 3: Partial Filing

```python
# 50% of group j=0 files taxes
p.update_specifications({
    "tax_filer": [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
})

# Group j=0 pays 50% of normal income taxes and faces 50% of normal MTRs
```

### Example 4: Modeling Filing Threshold Policy Reform

```python
# Baseline: Groups j=0 and j=1 are non-filers (low income)
baseline_spec = {
    "tax_filer": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}

# Reform: Lower filing threshold, only j=0 is non-filer
reform_spec = {
    "tax_filer": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}

# Compare economic effects of requiring j=1 to file
```

## Complete Example Script

See `examples/run_ogcore_nonfiler_example.py` for a complete working example that:
- Sets up a baseline with non-filers
- Runs a reform where all groups file
- Compares macroeconomic and household-level results
- Provides economic interpretation

Run it with:
```bash
cd examples
python run_ogcore_nonfiler_example.py
```

## Implementation Details

### What Gets Modified

When you set `tax_filer[j] = 0.0`, the following functions are affected:

1. **`ogcore.tax.income_tax_liab()`**: Returns zero income tax (but still returns payroll tax)
2. **`ogcore.tax.MTR_income()`**: Returns zero marginal tax rates on both labor and capital income
3. **`ogcore.household.FOC_labor()`**: Uses zero MTR in first-order condition for labor supply
4. **`ogcore.household.FOC_savings()`**: Uses zero MTR in Euler equation for savings

### What Stays the Same

- **Payroll taxes**: Non-filers still pay payroll taxes (Social Security, Medicare)
- **Wealth taxes**: If applicable, wealth taxes are unaffected
- **Consumption taxes**: Consumption taxes are unaffected
- **Bequest taxes**: Bequest taxes are unaffected
- **Government transfers**: Transfers and UBI are unaffected

## Economic Interpretation

### Effects of Non-Filer Status

**For the non-filing income group:**
- Higher labor supply (no income tax distortion on labor-leisure choice)
- Higher savings (no income tax distortion on savings decision)
- Higher consumption (higher after-tax income)

**General equilibrium effects:**
- Lower tax revenue
- Potentially higher GDP (less tax distortion)
- Lower interest rate (higher capital stock)
- Higher wage rate (higher capital-labor ratio)

### Policy Applications

**1. Standard Deduction Analysis**
Model the economic effects of the standard deduction by setting low-income groups as non-filers.

**2. Filing Threshold Reforms**
Analyze proposals to change filing thresholds by comparing different `tax_filer` configurations.

**3. Tax Compliance Policies**
Study the impact of policies that increase or decrease the share of filers using partial filing (0 < `tax_filer[j]` < 1).

**4. Distributional Analysis**
Examine how filing requirements affect different lifetime income groups.

## Technical Notes

### Numerical Optimization

The implementation ensures smooth optimization by:
- Applying `tax_filer` scaling within each j-group (no discontinuities within optimization)
- Allowing different behavior across j-groups (which are optimized separately)

### Backward Compatibility

The default value (`tax_filer = [1.0, 1.0, ...]`) preserves the original OG-Core behavior where all households file taxes. Existing models will run unchanged.

### Validation

The implementation has been validated through:
- 85 existing OG-Core tests (all pass)
- Custom verification tests for tax liabilities and MTRs
- Full model runs comparing non-filer and filer scenarios

## Questions or Issues?

If you have questions about using the `tax_filer` parameter or encounter any issues, please:
1. Check the example script: `examples/run_ogcore_nonfiler_example.py`
2. Review the test cases in `tests/test_tax.py` and `tests/test_household.py`
3. Open an issue on the OG-Core GitHub repository

## References

- **Parameter definition**: `ogcore/default_parameters.json` (lines 4251-4278)
- **Tax implementation**: `ogcore/tax.py`
  - `income_tax_liab()` function (lines 296-411)
  - `MTR_income()` function (lines 113-190)
- **Household FOCs**: `ogcore/household.py`
  - `FOC_labor()` function (lines 561-724)
  - `FOC_savings()` function (lines 373-558)
