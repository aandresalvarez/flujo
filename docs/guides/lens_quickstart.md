# 🚀 Flujo Lens Quick Start

**New in Lens v2.0** - Faster, smarter, and more powerful debugging.

---

## 🎯 Quick Commands

```bash
# List all runs
flujo lens list

# Find run by partial ID (new!)
flujo lens get abc123

# Show run details (supports partial IDs)
flujo lens show abc123

# Show with final output (new!)
flujo lens show abc123 --final-output

# Show as JSON for automation (new!)
flujo lens show abc123 --json

# Show everything
flujo lens show abc123 --verbose

# View execution trace
flujo lens trace abc123

# Replay a run
flujo lens replay abc123
```

---

## ✨ What's New

### 1. **Partial ID Matching**
No more copy-pasting long IDs!
```bash
# Before
flujo lens show run_ec00798feed049fb8b1e1c8bcb97eb17

# After
flujo lens show ec00798f
```

### 2. **Fuzzy Search with `get`**
Find runs quickly:
```bash
flujo lens get abc   # Shows all matching runs
```

### 3. **Final Output Display**
See what your pipeline produced:
```bash
flujo lens show abc123 --final-output
```

### 4. **JSON Output**
Perfect for CI/CD:
```bash
flujo lens show abc123 --json | jq '.details.status'
```

### 5. **No More Hanging!**
Fixed critical bug - commands complete in < 1 second.

---

## 📊 Common Workflows

### Debug a Failed Run
```bash
# 1. Find recent failures
flujo lens list --status failed

# 2. Investigate with partial ID
flujo lens show def456 --verbose

# 3. Check specific step outputs
flujo lens show def456 --show-error
```

### Export Run Data
```bash
# Export to JSON
flujo lens show abc123 --json > run_details.json

# Extract specific information
flujo lens show abc123 --json | jq '.steps[] | {name: .step_name, status: .status}'
```

### Quick Result Check
```bash
# Just see the final output
flujo lens show abc123 --final-output

# Or the full context
flujo lens show abc123 --verbose --final-output
```

---

## 🔧 Troubleshooting

### Timeout Issues
```bash
# Increase timeout
flujo lens show abc123 --timeout 30

# Or set permanently
export FLUJO_LENS_TIMEOUT=30
```

### Database Locked
```bash
# Check for other Flujo processes
ps aux | grep flujo

# Or switch to memory backend temporarily
FLUJO_STATE_URI=memory:// flujo lens list
```

### Can't Find Run
```bash
# List all runs
flujo lens list --limit 100

# Try fuzzy search
flujo lens get <any_part_of_id>
```

---

## 💡 Pro Tips

1. **Use short partial IDs**: First 8-12 characters are usually unique
2. **Combine flags**: `--verbose --final-output --json` works!
3. **Pipe to jq**: Perfect for extracting specific data
4. **Set FLUJO_LENS_TIMEOUT**: If your DB is slow
5. **Use `get` for exploration**: When you don't know the exact ID

---

## 📖 Full Documentation

- **Complete Guide**: [Lens Improvements](lens_improvements.md)
- **CLI Reference**: [CLI Reference](../reference/cli.md)
- **Tracing Guide**: [Tracing Guide](tracing_guide.md)
- **Replay Guide**: [Debugging with Replay](debugging_with_replay.md)

---

**Questions?** Open an issue or ask in Discord!

