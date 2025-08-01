#!/usr/bin/env python3
"""
Script to find the problematic test by running tests one by one.
This will help identify exactly which test is causing the hang.
"""

import subprocess
import time


def get_fast_tests():
    """Get list of fast tests using pytest collection."""
    cmd = [
        "uv", "run", "pytest", "tests/", 
        "-m", "not slow and not serial and not benchmark",
        "--collect-only", "-q"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Error collecting tests: {result.stderr}")
            return []
        
        # Parse the output to get test names
        lines = result.stdout.strip().split('\n')
        tests = []
        for line in lines:
            if line.strip() and not line.startswith('='):
                tests.append(line.strip())
        
        return tests
    except subprocess.TimeoutExpired:
        print("Timeout collecting tests")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def run_single_test(test_name, timeout=30):
    """Run a single test with timeout."""
    cmd = ["uv", "run", "pytest", test_name, "-v", "--tb=no"]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED: {test_name} ({duration:.2f}s)")
            return True
        else:
            print(f"❌ FAILED: {test_name} ({duration:.2f}s)")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {test_name} (>={timeout}s) - LIKELY PROBLEMATIC!")
        return False
    except Exception as e:
        print(f"💥 ERROR: {test_name} - {e}")
        return False


def main():
    print("🔍 Finding problematic test...")
    print("=" * 60)
    
    # Get list of fast tests
    print("📋 Collecting test list...")
    tests = get_fast_tests()
    
    if not tests:
        print("❌ No tests found!")
        return
    
    print(f"📊 Found {len(tests)} tests to check")
    print("=" * 60)
    
    # Run tests one by one
    passed = 0
    failed = 0
    timed_out = 0
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Testing: {test}")
        
        result = run_single_test(test)
        
        if result is True:
            passed += 1
        elif result is False:
            if "TIMEOUT" in test:
                timed_out += 1
                print(f"🚨 FOUND PROBLEMATIC TEST: {test}")
                print("   This test is likely causing the hang!")
                break
            else:
                failed += 1
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Timed out: {timed_out}")
    print(f"   Total: {len(tests)}")


if __name__ == "__main__":
    main() 