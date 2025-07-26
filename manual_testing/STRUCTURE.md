# Manual Testing Structure

## 📁 Directory Organization

The `manual_testing/` directory is organized into clear, purpose-driven folders:

```
manual_testing/
├── 📋 tests/                    # All test files
│   ├── 🤖 automated/           # Automated test suites
│   │   ├── test_step1_core_agentic.py
│   │   ├── run_step1_test.py
│   │   ├── comprehensive_test.py
│   │   ├── test_bug_demonstration.py
│   │   └── test_config.py
│   └── 🧪 manual/              # Manual tests with real API
│       ├── manual_test_step1.py
│       ├── manual_test_step1_challenging.py
│       └── interactive_test_step1.py
├── 📚 docs/                     # Documentation
│   ├── MANUAL_TESTING_SUMMARY.md
│   └── TEST_STEP1_SUMMARY.md
├── 🔧 examples/                 # Example implementations
│   ├── cohort_pipeline.py
│   └── main.py
├── 📄 README.md                 # Main documentation
├── ⚙️  flujo.toml              # Configuration
├── 🚀 run_tests.py              # Main test runner
└── 📄 STRUCTURE.md              # This file
```

## 🎯 Purpose of Each Directory

### 📋 `tests/` - Test Files
Contains all test-related files organized by type:

#### 🤖 `tests/automated/` - Automated Test Suites
- **Purpose**: Comprehensive automated testing with mock agents
- **Use Case**: Regression testing, CI/CD, validation
- **Files**:
  - `test_step1_core_agentic.py` - 11 comprehensive tests
  - `run_step1_test.py` - Test runner
  - `comprehensive_test.py` - Agent compatibility tests
  - `test_bug_demonstration.py` - FSD-11 bug demo
  - `test_config.py` - Configuration validation

#### 🧪 `tests/manual/` - Manual Tests (Real API)
- **Purpose**: Real API testing with actual cohort definitions
- **Use Case**: Learning, exploration, real-world validation
- **Files**:
  - `manual_test_step1.py` - Basic examples
  - `manual_test_step1_challenging.py` - Challenging cases
  - `interactive_test_step1.py` - Interactive input

### 📚 `docs/` - Documentation
- **Purpose**: Detailed documentation and summaries
- **Files**:
  - `MANUAL_TESTING_SUMMARY.md` - Complete manual testing guide
  - `TEST_STEP1_SUMMARY.md` - Step 1 test documentation

### 🔧 `examples/` - Example Implementations
- **Purpose**: Reference implementations and examples
- **Files**:
  - `cohort_pipeline.py` - Step 1 pipeline implementation
  - `main.py` - Basic pipeline runner

## 🚀 How to Use

### Quick Start
```bash
cd manual_testing
python3 run_tests.py
```

### Direct Access
```bash
# Automated tests
python3 tests/automated/run_step1_test.py

# Manual tests
python3 tests/manual/interactive_test_step1.py

# Examples
python3 examples/main.py
```

### Module Access
```bash
# Automated tests
python3 -m tests.automated.run_step1_test

# Manual tests
python3 -m tests.manual.interactive_test_step1

# Examples
python3 -m examples.main
```

## 🔄 Migration from Old Structure

### Old Files → New Locations
- `test_step1_core_agentic.py` → `tests/automated/`
- `run_step1_test.py` → `tests/automated/`
- `comprehensive_test.py` → `tests/automated/`
- `test_bug_demonstration.py` → `tests/automated/`
- `test_config.py` → `tests/automated/`
- `manual_test_step1.py` → `tests/manual/`
- `manual_test_step1_challenging.py` → `tests/manual/`
- `interactive_test_step1.py` → `tests/manual/`
- `cohort_pipeline.py` → `examples/`
- `main.py` → `examples/`
- `MANUAL_TESTING_SUMMARY.md` → `docs/`
- `TEST_STEP1_SUMMARY.md` → `docs/`

### Updated Imports
All import statements have been updated to reflect the new structure:
- `from manual_testing.cohort_pipeline` → `from manual_testing.examples.cohort_pipeline`
- `from manual_testing.main` → `from manual_testing.examples.main`

## 🎯 Benefits of New Structure

### ✅ **Clear Organization**
- Tests separated by type (automated vs manual)
- Documentation centralized
- Examples isolated

### ✅ **Easy Navigation**
- Intuitive folder names
- Clear purpose for each directory
- Logical file grouping

### ✅ **Scalable Structure**
- Easy to add new test types
- Simple to extend with new steps
- Clear separation of concerns

### ✅ **Multiple Access Methods**
- Interactive menu runner
- Direct file execution
- Module-based access

## 🔮 Future Extensions

This structure can easily accommodate future steps:

```
tests/
├── automated/
│   ├── test_step1_*.py
│   ├── test_step2_*.py
│   ├── test_step3_*.py
│   └── ...
├── manual/
│   ├── manual_test_step1_*.py
│   ├── manual_test_step2_*.py
│   └── ...
docs/
├── STEP1_*.md
├── STEP2_*.md
└── ...
examples/
├── step1_*.py
├── step2_*.py
└── ...
```

Each step can follow the same pattern, making the structure scalable and maintainable. 