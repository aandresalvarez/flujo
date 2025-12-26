# High-Assurance Gene Extraction with Real LLMs

This example demonstrates **production-ready** invariant-based tree search using real OpenAI and Anthropic API calls to extract gene-disease relationships from biomedical literature.

## 🎯 What This Example Demonstrates

1. **Real LLM Integration**: Actual API calls to GPT-4o and Claude (not mocks)
2. **Static Invariants**: Pre-defined rules that prevent contamination
3. **Discovery Agents**: LLMs that deduce safety rules at runtime
4. **Consensus Panel**: Parallel verification before expensive search
5. **Cost Tracking**: Real-time monitoring of API costs
6. **Separate Evaluation**: Dedicated script for analyzing results

## 📋 Requirements

### API Keys

You need API keys from both providers:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Dependencies

The example uses `flujo.toml` for configuration and dependencies:

```bash
# Dependencies are specified in flujo.toml
# Install them with:
pip install openai anthropic rich
```

Or if you have the Flujo CLI:

```bash
flujo install
```

## 💰 Cost Estimates

| Component | Model | Calls/Run | Cost/Call | Total |
|-----------|-------|-----------|-----------|-------|
| Proposer | GPT-4o | 10-20 | $0.005 | $0.05-0.10 |
| Evaluator | Claude 3.5 | 10-20 | $0.003 | $0.03-0.06 |
| Verification | GPT-4o + Claude 3.5 | 2 | $0.002 | $0.004 |
| Discovery | GPT-4o | 1 | $0.002 | $0.002 |
| **TOTAL** | | | | **$0.084-0.164** |

> **Note**: Actual costs may be **lower** due to invariant pruning catching contaminated candidates before evaluation.

## 🚀 Quick Start

### 1. Set Up Environment

```bash
cd examples/high_assurance_realistic
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Run Extraction

**Method 1: Direct Python** (Interactive with cost confirmation)
```bash
python main.py
```

**Method 2: Flujo CLI** (Recommended - better state management)
```bash
flujo run main.py --input "Extract MED13 gene facts"
```

**Pipeline composition mode** (Full Flujo DSL):
```bash
# Direct Python
python main.py --use-pipeline

# Flujo CLI (pipeline mode is default when using flujo run)
flujo run main.py --input "Extract MED13 facts" --json
```

**Advanced options**:
```bash
# Adjust search parameters (Python)
python main.py --max-depth 5 --beam-width 4

# Disable discovery agent (Python)
python main.py --no-discovery

# Flujo CLI with live progress
flujo run main.py --input "Extract facts" --live

# Flujo CLI with JSON output
flujo run main.py --input "Extract facts" --json

# Flujo CLI with debug mode
flujo run main.py --input "Extract facts" --debug
```

**Why use `flujo run`?**
- ✅ Better state persistence
- ✅ Live progress indicators
- ✅ JSON output mode
- ✅ Debug trace export
- ✅ Automatic flujo.toml detection

### 3. Evaluate Results

```bash
python evaluate_results.py
```

This produces a detailed report covering:
- Search efficiency
- Invariant effectiveness
- Cost analysis
- Quality metrics

## 📁 Project Structure

```
high_assurance_realistic/
├── README.md              # This file
├── DSL_SYNTAX.md          # Complete Flujo DSL syntax guide
├── flujo.toml             # Configuration and dependencies
├── main.py                # Main execution script (500+ lines)
├── evaluate_results.py    # Evaluation script (300+ lines)
├── agents.py              # Real LLM agents + @step decorators + Pipeline composition
├── models.py              # Domain models (Triplet, SearchResults)
├── invariants.py          # Invariant definitions
├── sample_data.py         # 6 real PubMed abstracts
└── results/               # Output directory
    ├── run_TIMESTAMP.json
    ├── latest.json
    └── report_TIMESTAMP.txt
```

## 🎓 Understanding the Code

### Main Execution Flow (`main.py`)

The script is organized into clear sections with detailed comments:

1. **Setup & Configuration**
   - API key verification
   - Cost estimation
   - Parameter configuration

2. **Tree Search Execution**
   - Configure `TreeSearchStep` with real LLM agents
   - Add static invariants (pre-defined rules)
   - Optionally add discovery agent (runtime rules)
   - Execute search with progress tracking

3. **Results Analysis**
   - Extract best triplet
   - Save complete results to JSON
   - Display summary

### Key Components

#### Proposer Agent (GPT-4o)

Extracts candidate triplets from text:

```python
proposer_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="Extract MED13 gene-disease triplets...",
    output_type=list[Triplet],
)
```

#### Evaluator Agent (Claude 3.5)

Scores triplet quality:

```python
evaluator_agent = make_agent_async(
    model="anthropic:claude-3-5-sonnet",
    system_prompt="Evaluate triplet quality (0.0-1.0)...",
    output_type=EvaluationReport,
)
```

#### Static Invariants

Pre-defined rules checked before evaluation:

```python
STRICT_INVARIANTS = [
    no_med13l_contamination,  # Callable
    must_mention_med13,        # Callable
    non_empty_output,          # Callable
    has_evidence_quote,        # Callable
]
```

#### Discovery Agent (GPT-4o)

Deduces invariants from the goal:

```python
discovery_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="Analyze goal and deduce safety rules...",
    output_type=list[str],
)
```

## 📊 Example Output

### During Execution

```
================================================================================
ENVIRONMENT VERIFICATION
================================================================================
✅ API keys verified
✅ Results directory: .../results

================================================================================
COST ESTIMATE
================================================================================
Estimated nodes: 12
Proposer cost: $0.0600
Evaluator cost: $0.0360
Discovery cost: $0.0020
TOTAL ESTIMATE: $0.0980

Proceed with extraction? (y/n): y

================================================================================
TREE SEARCH EXECUTION
================================================================================
Goal: Extract MED13 gene-disease relationships...
Max Depth: 4
Beam Width: 3
Discovery Agent: Enabled

✅ TreeSearchStep configured
✅ Loaded 6 PubMed abstracts
Starting tree search...

✅ Search completed
  - Execution time: 12.34s

================================================================================
RESULTS SUMMARY
================================================================================
Run ID: 20251225_155700
Total Nodes Explored: 14
Invariant Violations: 6
Goal Reached: ✅ Yes
Total Cost: $0.0842
```

### Evaluation Report

```
================================================================================
EVALUATION REPORT
================================================================================
Run ID: 20251225_155700

SEARCH EFFICIENCY
--------------------------------------------------------------------------------
Total Nodes Explored: 14
Execution Time: 12.34s
Goal Reached: ✅ Yes
Efficiency Rating: Excellent

INVARIANT EFFECTIVENESS
--------------------------------------------------------------------------------
Total Violations Caught: 6
Unique Rules Violated: 2
Estimated Pruning Rate: 14.3%
Effectiveness Rating: High (caught many contaminations)

Top Violated Rules:
  - no_med13l_contamination: 5 times
  - must_mention_med13: 1 times

COST ANALYSIS
--------------------------------------------------------------------------------
Total Cost: $0.0842
Total Tokens: 8,420
Cost per Node: $0.0060
Estimated Savings from Pruning: $0.0180
Savings Percentage: 17.6%

QUALITY METRICS
--------------------------------------------------------------------------------
Total Triplets Extracted: 3
Best Triplet Found: ✅ Yes
Best Triplet Confidence: 0.95
Quality Rating: Excellent
```

## 🔧 Configuration Options

### Command-Line Arguments

```bash
# Adjust search depth
python main.py --max-depth 5

# Adjust beam width
python main.py --beam-width 5

# Disable discovery agent
python main.py --no-discovery

# Combine options
python main.py --max-depth 6 --beam-width 4
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Override default models
export FLUJO_PROPOSER_MODEL="openai:gpt-4o-mini"
export FLUJO_EVALUATOR_MODEL="anthropic:claude-3-opus"
```

## 🐛 Troubleshooting

### "Missing API keys" Error

**Problem**: Script exits with missing API key error.

**Solution**:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "No valid triplet found" Result

**Problem**: Search completes but finds no valid triplets.

**Possible Causes**:
1. Invariants are too strict (rejecting all candidates)
2. Sample data doesn't contain target information
3. LLM extraction is failing

**Solutions**:
- Check `results/latest.json` for violation details
- Review invariant definitions in `invariants.py`
- Try with `--no-discovery` to use only static invariants

### High Costs

**Problem**: Costs are higher than expected.

**Solutions**:
- Reduce `--max-depth` (default: 4)
- Reduce `--beam-width` (default: 3)
- Check that invariants are working (should prune bad branches)

## 📚 Further Reading

- [Main Mock Example](../high_assurance_med13_extractor.py) - No API keys required
- [Tutorial](../README_high_assurance_reasoning.md) - Comprehensive guide
- [TreeSearchStep API](../../docs/user_guide/pipeline_dsl.md#treesearchstep)
- [Invariants Deep Dive](../../docs/cookbook/reasoning_with_astar.md)

## 🤝 Contributing

To adapt this example for your use case:

1. **Define your domain models** in `models.py`
2. **Add your data** in `sample_data.py`
3. **Define your invariants** in `invariants.py`
4. **Update agent prompts** in `agents.py`
5. **Run and evaluate** with `main.py` and `evaluate_results.py`

## 📝 License

This example is part of the Flujo framework and follows the same license.
