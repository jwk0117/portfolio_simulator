# Portfolio Simulator - Dependency Audit Report

**Date**: 2026-01-19
**Status**: ✅ No Critical Issues Found

## Executive Summary

This audit analyzed all dependencies in `requirements.txt` for security vulnerabilities, outdated packages, and unnecessary bloat. Overall, the project maintains a lean dependency list with no known security vulnerabilities.

## Security Status

✅ **No known security vulnerabilities detected** (verified with pip-audit)

All packages passed security scanning with no CVEs or security advisories.

## Dependency Analysis

### Current Dependencies

| Package | Current Version | Latest Version | Status | Used? |
|---------|----------------|----------------|---------|-------|
| streamlit | ≥1.34 | 1.53.0 | Outdated | ✅ Yes |
| pandas | ≥2.0 | 2.3.3 | Outdated | ✅ Yes |
| numpy | ≥1.24 | 2.4.1 | Outdated | ✅ Yes |
| yfinance | ≥0.2 | 1.0 | Outdated | ✅ Yes |
| plotly | ≥5.18 | 6.5.2 | Outdated | ✅ Yes |
| matplotlib | ≥3.8 | 3.10.8 | Outdated | ✅ Yes |
| openpyxl | ≥3.1 | 3.1.5 | Current | ❌ **UNUSED** |
| pyarrow | ≥14.0 | 23.0.0 | Outdated | ✅ Yes |

### Dependency Usage Details

**✅ Essential Dependencies:**
- **streamlit** (portfolio_gui_app.py:34) - Core GUI framework
- **pandas** (all modules) - DataFrame operations and data handling
- **numpy** (portfolio_sim.py:6, portfolio_gui_core.py:25, portfolio_gui_plotting.py:11) - Numerical computations
- **yfinance** (portfolio_sim.py:8) - Market data fetching from Yahoo Finance
- **plotly** (portfolio_gui_plotting.py) - Interactive visualizations with fallback support
- **matplotlib** (portfolio_sim.py:9, portfolio_gui_app.py:36) - Static plotting and fallback
- **pyarrow** (portfolio_gui_core.py:699) - Parquet file export via `df.to_parquet()`

**❌ Unnecessary Dependencies:**
- **openpyxl** - NOT USED
  - No Excel file operations found (`.xlsx`, `.xls`, `read_excel`, `to_excel`)
  - Only CSV and Pickle formats are used for I/O
  - Recommendation: **REMOVE**

## Recommendations

### 1. 🔴 HIGH PRIORITY - Remove Unnecessary Bloat

**Remove openpyxl** - This package is not used anywhere in the codebase and adds unnecessary installation size and complexity.

```diff
- openpyxl>=3.1
```

**Impact:**
- Reduces installation size by ~1.5 MB
- Simplifies dependency tree
- Faster deployment to Streamlit Cloud

### 2. 🟡 MEDIUM PRIORITY - Update Outdated Packages

All packages are significantly behind latest versions. Consider updating minimum versions to benefit from bug fixes, performance improvements, and new features.

**Recommended updates:**

```diff
# Core framework
- streamlit>=1.34
+ streamlit>=1.53.0

# Data handling
- pandas>=2.0
+ pandas>=2.3.0
- numpy>=1.24
+ numpy>=2.4.0

# Market data
- yfinance>=0.2
+ yfinance>=1.0

# Visualization
- plotly>=5.18
+ plotly>=6.5.0
- matplotlib>=3.8
+ matplotlib>=3.10.0

# File format support
- pyarrow>=14.0
+ pyarrow>=23.0.0
```

**Considerations:**
- **numpy 2.x** is a major version upgrade - test thoroughly for breaking changes
- **yfinance 1.0** is a stable release with improved API
- **streamlit 1.53** includes performance improvements and bug fixes

### 3. 🟢 LOW PRIORITY - Version Pinning Strategy

Current strategy uses minimum versions (`>=`), which is flexible but can lead to inconsistent deployments.

**Options:**
1. **Keep flexible** (current) - Allows automatic updates, may break unexpectedly
2. **Pin exact versions** (`==`) - Reproducible builds, requires manual updates
3. **Use constraints** (`>=X.Y,<X+1`) - Balance between stability and updates

**Recommendation:** Consider using `~=` (compatible release) for more predictable updates:
```python
streamlit~=1.53.0  # Allows 1.53.x but not 1.54.0
```

## Version Compatibility Matrix

| Package | Python 3.11 | Python 3.12 | Python 3.13 |
|---------|-------------|-------------|-------------|
| streamlit 1.53 | ✅ | ✅ | ✅ |
| pandas 2.3 | ✅ | ✅ | ✅ |
| numpy 2.4 | ✅ | ✅ | ✅ |
| yfinance 1.0 | ✅ | ✅ | ⚠️ Check |
| plotly 6.5 | ✅ | ✅ | ✅ |
| matplotlib 3.10 | ✅ | ✅ | ✅ |
| pyarrow 23.0 | ✅ | ✅ | ✅ |

## Implementation Plan

### Immediate Action (Recommended)
```bash
# Remove unused dependency
# Edit requirements.txt to remove openpyxl line
# Test deployment to ensure no issues
```

### Optional Updates (Test First)
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install updated dependencies
pip install streamlit>=1.53.0 pandas>=2.3.0 numpy>=2.4.0 yfinance>=1.0 \
            plotly>=6.5.0 matplotlib>=3.10.0 pyarrow>=23.0.0

# Run application and test all features
streamlit run portfolio_gui_app.py

# If successful, update requirements.txt
```

## Testing Checklist

Before deploying updated dependencies:

- [ ] Data loading from pickle files works
- [ ] Market data fetching via yfinance works
- [ ] Portfolio simulations run without errors
- [ ] Plotly interactive charts render correctly
- [ ] Matplotlib fallback charts work
- [ ] Parquet export functionality works
- [ ] CSV export functionality works
- [ ] All numerical calculations produce expected results
- [ ] Streamlit Cloud deployment succeeds

## Conclusion

**Overall Assessment:** 🟢 Good

The project maintains a clean dependency list with only one unnecessary package (openpyxl). No security vulnerabilities were found. While packages are behind latest versions, the current minimum versions are still maintained and secure.

**Priority Actions:**
1. ✅ Remove openpyxl (immediate)
2. ⚠️ Test and update to latest package versions (optional but recommended)
3. 📋 Establish a regular dependency review schedule (quarterly recommended)

---

**Audit Tool:** pip-audit 2.8.0
**Python Version:** 3.11.14
**Audited by:** Claude Code Agent
