# 🔧 Summary Aggregation Bug - Workaround Examples

## 📋 **Overview**

This document provides comprehensive workarounds for the summary aggregation bug. These workarounds allow users to access accurate cost and token information while waiting for the framework fix.

**⚠️  Important**: These are temporary solutions. The root cause needs to be fixed by the Flujo team.

---

## 🚨 **The Problem**

Flujo summary tables fail to aggregate costs and tokens from nested steps, showing only wrapper step data with $0.0000 costs and 0 tokens. This makes it impossible to track actual pipeline costs and resource usage from the summary table.

**Impact**:
- Breaks cost tracking and monitoring
- Shows misleading information in summary
- Requires workarounds for accurate data
- Affects all pipelines with nested workflows

---

## 🔧 **Workaround 1: JSON Output Flag (Recommended)**

### **Basic Usage**
```bash
flujo run --json pipeline.yaml
```

### **With Input Piping**
```bash
echo "your input" | flujo run --json pipeline.yaml
```

### **With Output Capture**
```bash
flujo run --json pipeline.yaml > output.json
```

### **Advantages**
- ✅ Simple and reliable
- ✅ Provides complete, accurate data
- ✅ Works in all environments
- ✅ No additional dependencies

### **Disadvantages**
- ❌ Requires manual parsing
- ❌ Not user-friendly for quick viewing
- ❌ Requires JSON processing knowledge

### **Example Output Structure**
```json
{
  "step_history": [
    {
      "name": "wrapper_workflow",
      "total_cost_usd": 0.0,
      "total_tokens": 0,
      "branch_context": {
        "wrapper_workflow": {
          "total_cost_usd": 0.00057525,
          "total_tokens": 1746,
          "step_history": [
            {
              "name": "inner_step_1",
              "cost_usd": 0.0000726,
              "token_counts": 370
            }
          ]
        }
      }
    }
  ]
}
```

---

## 🔧 **Workaround 2: Manual JSON Parsing**

### **Python Script for Cost Extraction**
```python
#!/usr/bin/env python3
"""
Flujo Cost Extractor - Workaround for Summary Aggregation Bug
"""

import json
import subprocess
import sys

def extract_costs_from_json(json_output):
    """Extract cost and token information from Flujo JSON output."""
    try:
        data = json.loads(json_output)
        
        # Extract top-level summary
        top_cost = data.get('total_cost_usd', 0)
        top_tokens = data.get('total_tokens', 0)
        top_steps = len(data.get('step_history', []))
        
        print(f"📊 Top-Level Summary:")
        print(f"   Total cost: ${top_cost}")
        print(f"   Total tokens: {top_tokens}")
        print(f"   Steps: {top_steps}")
        
        # Extract nested workflow data
        if 'step_history' in data and len(data['step_history']) > 0:
            first_step = data['step_history'][0]
            if 'branch_context' in first_step:
                print(f"\n🔍 Nested Workflow Data:")
                
                nested_data = first_step['branch_context']
                for key, value in nested_data.items():
                    if isinstance(value, dict) and 'total_cost_usd' in value:
                        nested_cost = value['total_cost_usd']
                        nested_tokens = value.get('total_tokens', 0)
                        nested_steps = len(value.get('step_history', []))
                        
                        print(f"   Workflow: {key}")
                        print(f"     Total cost: ${nested_cost}")
                        print(f"     Total tokens: {nested_tokens}")
                        print(f"     Steps: {nested_steps}")
                        
                        # Show individual step costs
                        if 'step_history' in value:
                            print(f"     Individual steps:")
                            for step in value['step_history']:
                                step_cost = step.get('cost_usd', 0)
                                step_tokens = step.get('token_counts', 0)
                                step_name = step.get('name', 'unknown')
                                print(f"       {step_name}: ${step_cost} ({step_tokens} tokens)")
                        
                        # Check for mismatch
                        if nested_cost != top_cost or nested_tokens != top_tokens:
                            print(f"     ⚠️  MISMATCH: Nested data differs from top-level!")
                            print(f"        This confirms the summary aggregation bug!")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        return False

def run_pipeline_with_cost_extraction(pipeline_file, input_data=None):
    """Run a Flujo pipeline and extract cost information."""
    print(f"🚀 Running pipeline: {pipeline_file}")
    
    cmd = ['flujo', 'run', '--json', pipeline_file]
    
    if input_data:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        stdout, stderr = process.communicate(input=input_data, timeout=60)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
    
    if process.returncode == 0:
        print("✅ Pipeline executed successfully")
        print("📊 Extracting cost information...")
        extract_costs_from_json(stdout)
    else:
        print(f"❌ Pipeline execution failed: {stderr}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cost_extractor.py <pipeline.yaml> [input_data]")
        sys.exit(1)
    
    pipeline_file = sys.argv[1]
    input_data = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_pipeline_with_cost_extraction(pipeline_file, input_data)

if __name__ == "__main__":
    main()
```

### **Usage**
```bash
# Extract costs from pipeline
python cost_extractor.py pipeline.yaml

# Extract costs with input
python cost_extractor.py pipeline.yaml "test input"
```

---

## 🔧 **Workaround 3: Shell Script Wrapper**

### **Cost Extraction Script**
```bash
#!/bin/bash
# Flujo Cost Extractor Script
# Workaround for Summary Aggregation Bug

set -e

PIPELINE_FILE="$1"
INPUT_DATA="${2:-test input}"

if [ -z "$PIPELINE_FILE" ]; then
    echo "Usage: $0 <pipeline.yaml> [input_data]"
    exit 1
fi

echo "🚀 Running pipeline: $PIPELINE_FILE"
echo "📥 Input: $INPUT_DATA"

# Run pipeline with JSON output
echo "$INPUT_DATA" | flujo run --json "$PIPELINE_FILE" > temp_output.json

# Extract and display cost information
echo ""
echo "📊 Cost and Token Information:"
echo "=============================="

# Extract top-level summary
TOP_COST=$(jq -r '.total_cost_usd // 0' temp_output.json)
TOP_TOKENS=$(jq -r '.total_tokens // 0' temp_output.json)
TOP_STEPS=$(jq -r '.step_history | length // 0' temp_output.json)

echo "Top-Level Summary:"
echo "  Total cost: \$$TOP_COST"
echo "  Total tokens: $TOP_TOKENS"
echo "  Steps: $TOP_STEPS"

# Extract nested workflow data
echo ""
echo "Nested Workflow Data:"

NESTED_COST=$(jq -r '.step_history[0].branch_context | to_entries[] | select(.value.total_cost_usd != null) | .value.total_cost_usd // 0' temp_output.json 2>/dev/null || echo "0")
NESTED_TOKENS=$(jq -r '.step_history[0].branch_context | to_entries[] | select(.value.total_tokens != null) | .value.total_tokens // 0' temp_output.json 2>/dev/null || echo "0")
NESTED_STEPS=$(jq -r '.step_history[0].branch_context | to_entries[] | select(.value.step_history != null) | .value.step_history | length // 0' temp_output.json 2>/dev/null || echo "0")

echo "  Nested workflow totals:"
echo "    Total cost: \$$NESTED_COST"
echo "    Total tokens: $NESTED_TOKENS"
echo "    Steps: $NESTED_STEPS"

# Check for mismatch
if [ "$NESTED_COST" != "$TOP_COST" ] || [ "$NESTED_TOKENS" != "$TOP_TOKENS" ]; then
    echo ""
    echo "⚠️  MISMATCH DETECTED:"
    echo "   Top-level: \$$TOP_COST, $TOP_TOKENS tokens"
    echo "   Nested: \$$NESTED_COST, $NESTED_TOKENS tokens"
    echo "   This confirms the summary aggregation bug!"
fi

# Cleanup
rm -f temp_output.json

echo ""
echo "✅ Cost extraction completed"
```

### **Usage**
```bash
# Make executable
chmod +x cost_extractor.sh

# Extract costs
./cost_extractor.sh pipeline.yaml

# Extract costs with input
./cost_extractor.sh pipeline.yaml "custom input"
```

---

## 🔧 **Workaround 4: Python Class-Based Solution**

### **FlujoCostTracker Class**
```python
#!/usr/bin/env python3
"""
Flujo Cost Tracker - Advanced Workaround for Summary Aggregation Bug
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple

class FlujoCostTracker:
    """Track and analyze Flujo pipeline costs and tokens."""
    
    def __init__(self):
        self.pipeline_results = {}
    
    def run_pipeline(self, pipeline_file: str, input_data: str = "test input") -> Dict:
        """Run a Flujo pipeline and capture cost data."""
        print(f"🚀 Running pipeline: {pipeline_file}")
        
        cmd = ['flujo', 'run', '--json', pipeline_file]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        
        stdout, stderr = process.communicate(input=input_data, timeout=60)
        
        if process.returncode != 0:
            raise RuntimeError(f"Pipeline execution failed: {stderr}")
        
        return self.parse_pipeline_output(stdout)
    
    def parse_pipeline_output(self, json_output: str) -> Dict:
        """Parse Flujo JSON output and extract cost information."""
        try:
            data = json.loads(json_output)
            return self.extract_cost_data(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON output: {e}")
    
    def extract_cost_data(self, data: Dict) -> Dict:
        """Extract comprehensive cost and token data."""
        result = {
            'top_level': {
                'total_cost_usd': data.get('total_cost_usd', 0),
                'total_tokens': data.get('total_tokens', 0),
                'step_count': len(data.get('step_history', []))
            },
            'nested_workflows': [],
            'individual_steps': [],
            'aggregation_bug_detected': False
        }
        
        # Extract nested workflow data
        if 'step_history' in data and len(data['step_history']) > 0:
            for step in data['step_history']:
                if 'branch_context' in step:
                    nested_data = step['branch_context']
                    for key, value in nested_data.items():
                        if isinstance(value, dict) and 'total_cost_usd' in value:
                            nested_workflow = {
                                'name': key,
                                'total_cost_usd': value.get('total_cost_usd', 0),
                                'total_tokens': value.get('total_tokens', 0),
                                'step_count': len(value.get('step_history', [])),
                                'steps': []
                            }
                            
                            # Extract individual step data
                            if 'step_history' in value:
                                for nested_step in value['step_history']:
                                    step_data = {
                                        'name': nested_step.get('name', 'unknown'),
                                        'cost_usd': nested_step.get('cost_usd', 0),
                                        'token_counts': nested_step.get('token_counts', 0),
                                        'latency_s': nested_step.get('latency_s', 0),
                                        'success': nested_step.get('success', False)
                                    }
                                    nested_workflow['steps'].append(step_data)
                                    result['individual_steps'].append(step_data)
                            
                            result['nested_workflows'].append(nested_workflow)
        
        # Check for aggregation bug
        top_cost = result['top_level']['total_cost_usd']
        nested_cost = sum(wf['total_cost_usd'] for wf in result['nested_workflows'])
        
        if abs(top_cost - nested_cost) > 0.000001:  # Account for floating point precision
            result['aggregation_bug_detected'] = True
        
        return result
    
    def generate_report(self, cost_data: Dict) -> str:
        """Generate a formatted cost report."""
        report = []
        report.append("📊 Flujo Pipeline Cost Report")
        report.append("=" * 40)
        
        # Top-level summary
        top = cost_data['top_level']
        report.append(f"Top-Level Summary:")
        report.append(f"  Total cost: ${top['total_cost_usd']}")
        report.append(f"  Total tokens: {top['total_tokens']}")
        report.append(f"  Steps: {top['step_count']}")
        
        # Nested workflow summary
        if cost_data['nested_workflows']:
            report.append("")
            report.append("Nested Workflows:")
            total_nested_cost = 0
            total_nested_tokens = 0
            total_nested_steps = 0
            
            for wf in cost_data['nested_workflows']:
                report.append(f"  {wf['name']}:")
                report.append(f"    Cost: ${wf['total_cost_usd']}")
                report.append(f"    Tokens: {wf['total_tokens']}")
                report.append(f"    Steps: {wf['step_count']}")
                
                total_nested_cost += wf['total_cost_usd']
                total_nested_tokens += wf['total_tokens']
                total_nested_steps += wf['step_count']
            
            report.append("")
            report.append(f"Total Nested: ${total_nested_cost} ({total_nested_tokens} tokens, {total_nested_steps} steps)")
        
        # Individual step breakdown
        if cost_data['individual_steps']:
            report.append("")
            report.append("Individual Steps:")
            for step in cost_data['individual_steps']:
                status = "✅" if step['success'] else "❌"
                report.append(f"  {status} {step['name']}: ${step['cost_usd']} ({step['token_counts']} tokens, {step['latency_s']:.3f}s)")
        
        # Bug detection
        if cost_data['aggregation_bug_detected']:
            report.append("")
            report.append("🚨 SUMMARY AGGREGATION BUG DETECTED!")
            report.append("   Top-level summary does not match nested workflow totals.")
            report.append("   Use this report for accurate cost tracking.")
        
        return "\n".join(report)
    
    def save_report(self, cost_data: Dict, output_file: str):
        """Save cost data to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(cost_data, f, indent=2)
        print(f"💾 Cost data saved to: {output_file}")

def main():
    """Example usage of FlujoCostTracker."""
    tracker = FlujoCostTracker()
    
    if len(sys.argv) < 2:
        print("Usage: python cost_tracker.py <pipeline.yaml> [input_data]")
        sys.exit(1)
    
    pipeline_file = sys.argv[1]
    input_data = sys.argv[2] if len(sys.argv) > 2 else "test input"
    
    try:
        # Run pipeline and extract costs
        cost_data = tracker.run_pipeline(pipeline_file, input_data)
        
        # Generate and display report
        report = tracker.generate_report(cost_data)
        print(report)
        
        # Save detailed data
        tracker.save_report(cost_data, "pipeline_costs.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### **Usage**
```bash
# Run cost tracker
python cost_tracker.py pipeline.yaml

# Run with custom input
python cost_tracker.py pipeline.yaml "custom input"
```

---

## 🔧 **Workaround 5: CI/CD Integration**

### **GitHub Actions Example**
```yaml
name: Run Flujo Pipeline with Cost Tracking
on: [push, pull_request]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Flujo
        run: |
          pip install flujo

      - name: Run Pipeline with Cost Extraction
        run: |
          # Run pipeline with JSON output
          echo "test input" | flujo run --json pipeline.yaml > output.json
          
          # Extract cost information
          TOP_COST=$(jq -r '.total_cost_usd // 0' output.json)
          NESTED_COST=$(jq -r '.step_history[0].branch_context | to_entries[] | select(.value.total_cost_usd != null) | .value.total_cost_usd // 0' output.json 2>/dev/null || echo "0")
          
          echo "Top-level cost: $TOP_COST"
          echo "Nested cost: $NESTED_COST"
          
          # Check for aggregation bug
          if [ "$TOP_COST" != "$NESTED_COST" ]; then
            echo "⚠️  Summary aggregation bug detected!"
            echo "Using nested cost data for accurate tracking."
            ACTUAL_COST=$NESTED_COST
          else
            echo "✅ Summary aggregation working correctly"
            ACTUAL_COST=$TOP_COST
          fi
          
          echo "Actual pipeline cost: $ACTUAL_COST"
          
          # Fail if cost exceeds threshold
          if (( $(echo "$ACTUAL_COST > 0.01" | bc -l) )); then
            echo "❌ Pipeline cost ($ACTUAL_COST) exceeds threshold (0.01)"
            exit 1
          fi
```

### **Jenkins Pipeline Example**
```groovy
pipeline {
    agent any
    
    stages {
        stage('Run Flujo Pipeline') {
            steps {
                script {
                    // Run pipeline with JSON output
                    def result = sh(
                        script: 'echo "test input" | flujo run --json pipeline.yaml',
                        returnStdout: true
                    ).trim()
                    
                    // Parse JSON and extract costs
                    def jsonData = readJSON text: result
                    def topCost = jsonData.total_cost_usd ?: 0
                    def nestedCost = 0
                    
                    // Extract nested workflow costs
                    if (jsonData.step_history && jsonData.step_history[0].branch_context) {
                        jsonData.step_history[0].branch_context.each { key, value ->
                            if (value.total_cost_usd) {
                                nestedCost += value.total_cost_usd
                            }
                        }
                    }
                    
                    echo "Top-level cost: ${topCost}"
                    echo "Nested cost: ${nestedCost}"
                    
                    // Use actual cost for monitoring
                    def actualCost = (topCost != nestedCost) ? nestedCost : topCost
                    echo "Actual pipeline cost: ${actualCost}"
                    
                    // Fail if cost exceeds threshold
                    if (actualCost > 0.01) {
                        error "Pipeline cost (${actualCost}) exceeds threshold (0.01)"
                    }
                }
            }
        }
    }
}
```

---

## 📊 **Workaround Comparison**

| Workaround | Ease of Use | Reliability | Automation | Complexity |
|------------|-------------|-------------|------------|------------|
| JSON Output Flag | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Manual JSON Parsing | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Shell Script Wrapper | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Python Class | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| CI/CD Integration | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 **Recommendations by Use Case**

### **For Development**
- **Primary**: JSON output flag
- **Secondary**: Manual JSON parsing
- **Reason**: Simple, reliable, immediate access to data

### **For Production**
- **Primary**: Python class-based solution
- **Secondary**: CI/CD integration
- **Reason**: Robust, configurable, production-ready

### **For Automation**
- **Primary**: CI/CD integration
- **Secondary**: Python class-based solution
- **Reason**: Automated, reliable, integrated

### **For Quick Testing**
- **Primary**: Shell script wrapper
- **Secondary**: JSON output flag
- **Reason**: Fast, simple, immediate results

---

## 🚀 **Implementation Priority**

### **Immediate (Today)**
1. **Use JSON output flag** for all pipeline runs
2. **Implement basic cost extraction** scripts
3. **Document workarounds** for team members

### **Short-term (This Week)**
1. **Create comprehensive cost tracking** solutions
2. **Update CI/CD pipelines** with cost monitoring
3. **Test all workarounds** in your environment

### **Long-term (This Month)**
1. **Implement production cost tracking** systems
2. **Create automated cost monitoring** dashboards
3. **Integrate with existing monitoring** tools

---

## 📝 **Monitoring and Alerts**

### **Cost Threshold Monitoring**
```bash
# Check pipeline costs and alert if too high
pipeline_cost=$(echo "input" | flujo run --json pipeline.yaml | jq -r '.step_history[0].branch_context | to_entries[] | select(.value.total_cost_usd != null) | .value.total_cost_usd // 0')

if (( $(echo "$pipeline_cost > 0.01" | bc -l) )); then
    echo "ALERT: Pipeline cost ($pipeline_cost) exceeds threshold!"
    # Send alert, notify team, etc.
fi
```

### **Token Usage Monitoring**
```bash
# Monitor token usage
token_count=$(echo "input" | flujo run --json pipeline.yaml | jq -r '.step_history[0].branch_context | to_entries[] | select(.value.total_tokens != null) | .value.total_tokens // 0')

echo "Pipeline used $token_count tokens"
```

---

## 🔄 **Migration Path**

### **Phase 1: Immediate Protection**
- Implement JSON output workarounds everywhere
- Update documentation and procedures
- Train team on workarounds

### **Phase 2: Enhanced Workarounds**
- Implement comprehensive cost tracking
- Add CI/CD integration
- Create automated monitoring

### **Phase 3: Production Ready**
- Deploy production cost tracking systems
- Implement comprehensive monitoring
- Create automated alerting

### **Phase 4: Framework Fix**
- Remove workarounds when Flujo team provides fix
- Update procedures and documentation
- Test fix thoroughly before deployment

---

## 📞 **Support and Updates**

### **Workaround Maintenance**
- **Monitor effectiveness** of each workaround
- **Update scripts** based on Flujo version changes
- **Refine monitoring** based on real-world usage

### **Framework Fix Tracking**
- **Monitor Flujo releases** for bug fixes
- **Test fixes** in development environment
- **Plan migration** from workarounds to fixes

### **Community Support**
- **Share workarounds** with other Flujo users
- **Report additional issues** to Flujo team
- **Contribute improvements** to workaround scripts

---

**Remember**: These workarounds are temporary solutions. The root cause needs to be fixed by the Flujo team. Use them to maintain accurate cost tracking and resource monitoring while waiting for the framework fix.
