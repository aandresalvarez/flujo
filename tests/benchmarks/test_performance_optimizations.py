"""
Performance benchmark tests for Flujo optimizations.

This module benchmarks the performance improvements from:
1. uvloop event loop optimization
2. time.perf_counter_ns() precision improvements
3. bytearray buffer reuse for serialization
4. orjson and blake3 optimizations
"""

import asyncio
import time
import pytest
from typing import List, Dict, Any

from flujo import Step
from flujo.testing.utils import StubAgent
from flujo.utils.performance import time_perf_ns, time_perf_ns_to_seconds, measure_time, measure_time_async
from tests.conftest import create_test_flujo


class TestPerformanceOptimizations:
    """Test the performance optimizations implemented in Flujo."""

    @pytest.mark.benchmark
    def test_perf_counter_ns_precision(self, benchmark):
        """Benchmark the precision improvement of perf_counter_ns vs perf_counter."""
        
        def measure_with_perf_counter():
            start = time.perf_counter()
            # Simulate some work
            sum(range(1000))
            end = time.perf_counter()
            return end - start
        
        def measure_with_perf_counter_ns():
            start = time_perf_ns()
            # Simulate some work
            sum(range(1000))
            end = time_perf_ns()
            return time_perf_ns_to_seconds(end - start)
        
        # Benchmark both methods
        perf_counter_time = benchmark(measure_with_perf_counter)
        perf_counter_ns_time = benchmark(measure_with_perf_counter_ns)
        
        print(f"\nPrecision Benchmark Results:")
        print(f"  time.perf_counter():     {perf_counter_time:.6f}s")
        print(f"  time.perf_counter_ns():  {perf_counter_ns_time:.6f}s")
        
        # Both should be similar, but ns version should be more precise
        assert abs(perf_counter_time - perf_counter_ns_time) < 0.001

    @pytest.mark.benchmark
    def test_serialization_performance(self, benchmark):
        """Benchmark serialization performance with optimizations."""
        
        # Test data
        test_data = {
            "nested": {
                "list": [{"item": i} for i in range(100)],
                "dict": {f"key_{i}": f"value_{i}" for i in range(100)},
            },
            "complex": {"list": [1, 2, 3], "dict": {"a": 1}},
        }
        
        def serialize_with_json():
            import json
            return json.dumps(test_data, sort_keys=True)
        
        def serialize_with_orjson():
            try:
                import orjson
                return orjson.dumps(test_data, option=orjson.OPT_SORT_KEYS)
            except ImportError:
                return serialize_with_json()
        
        # Benchmark both methods
        json_time = benchmark(serialize_with_json)
        orjson_time = benchmark(serialize_with_orjson)
        
        print(f"\nSerialization Benchmark Results:")
        print(f"  json.dumps():    {json_time:.6f}s")
        print(f"  orjson.dumps():  {orjson_time:.6f}s")
        
        # orjson should be faster if available
        if hasattr(orjson_time, 'stats'):
            print(f"  Speedup: {json_time.stats.mean / orjson_time.stats.mean:.2f}x")

    @pytest.mark.benchmark
    def test_hashing_performance(self, benchmark):
        """Benchmark hashing performance with optimizations."""
        
        # Test data
        test_data = b"test_data_for_hashing" * 1000
        
        def hash_with_hashlib():
            import hashlib
            return hashlib.blake2b(test_data, digest_size=32).hexdigest()
        
        def hash_with_blake3():
            try:
                import blake3
                return str(blake3.blake3(test_data).hexdigest())
            except ImportError:
                return hash_with_hashlib()
        
        # Benchmark both methods
        hashlib_time = benchmark(hash_with_hashlib)
        blake3_time = benchmark(hash_with_blake3)
        
        print(f"\nHashing Benchmark Results:")
        print(f"  hashlib.blake2b(): {hashlib_time:.6f}s")
        print(f"  blake3.blake3():   {blake3_time:.6f}s")
        
        # blake3 should be faster if available
        if hasattr(blake3_time, 'stats'):
            print(f"  Speedup: {hashlib_time.stats.mean / blake3_time.stats.mean:.2f}x")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_async_performance_with_uvloop(self, benchmark):
        """Benchmark async performance with uvloop optimization."""
        
        # Create a simple pipeline
        agent = StubAgent(["output"] * 100)  # Multiple outputs for benchmark
        pipeline = Step.solution(agent)
        runner = create_test_flujo(pipeline)
        
        async def run_pipeline():
            result = await runner.run_async("test_input")
            return result
        
        # Benchmark async execution
        execution_time = benchmark(lambda: asyncio.run(run_pipeline()))
        
        print(f"\nAsync Performance Benchmark:")
        print(f"  Pipeline execution: {execution_time:.6f}s")
        
        # Should complete in reasonable time
        assert execution_time < 5.0, f"Async execution too slow: {execution_time:.3f}s"

    @pytest.mark.benchmark
    def test_measure_time_decorator(self, benchmark):
        """Test the measure_time decorator performance."""
        
        @measure_time
        def test_function():
            return sum(range(10000))
        
        # Benchmark the decorated function
        result_time = benchmark(test_function)
        
        print(f"\nMeasure Time Decorator Benchmark:")
        print(f"  Decorated function: {result_time:.6f}s")
        
        # Should complete quickly
        assert result_time < 0.1, f"Decorated function too slow: {result_time:.3f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_measure_time_async_decorator(self, benchmark):
        """Test the measure_time_async decorator performance."""
        
        @measure_time_async
        async def test_async_function():
            await asyncio.sleep(0.001)  # Simulate async work
            return sum(range(1000))
        
        # Benchmark the decorated async function
        result_time = benchmark(lambda: asyncio.run(test_async_function()))
        
        print(f"\nMeasure Time Async Decorator Benchmark:")
        print(f"  Decorated async function: {result_time:.6f}s")
        
        # Should complete in reasonable time
        assert result_time < 1.0, f"Decorated async function too slow: {result_time:.3f}s"

    def test_scratch_buffer_reuse(self):
        """Test that scratch buffer reuse works correctly."""
        from flujo.utils.performance import clear_scratch_buffer, get_scratch_buffer
        
        # Get initial buffer
        buffer1 = get_scratch_buffer()
        initial_size = len(buffer1)
        
        # Add some data
        buffer1.extend(b"test_data")
        
        # Clear and reuse
        clear_scratch_buffer()
        buffer2 = get_scratch_buffer()
        
        # Should be the same buffer object
        assert buffer1 is buffer2
        assert len(buffer2) == 0
        
        # Add more data
        buffer2.extend(b"more_data")
        
        # Clear again
        clear_scratch_buffer()
        assert len(buffer2) == 0

    @pytest.mark.benchmark
    def test_buffer_reuse_performance(self, benchmark):
        """Benchmark the performance improvement from buffer reuse."""
        
        def without_buffer_reuse():
            # Simulate creating new buffers
            buffers = []
            for i in range(1000):
                buffer = bytearray(1024)
                buffer.extend(f"data_{i}".encode())
                buffers.append(buffer)
            return len(buffers)
        
        def with_buffer_reuse():
            # Simulate reusing the same buffer
            from flujo.utils.performance import clear_scratch_buffer, get_scratch_buffer
            count = 0
            for i in range(1000):
                clear_scratch_buffer()
                buffer = get_scratch_buffer()
                buffer.extend(f"data_{i}".encode())
                count += 1
            return count
        
        # Benchmark both approaches
        without_reuse_time = benchmark(without_buffer_reuse)
        with_reuse_time = benchmark(with_buffer_reuse)
        
        print(f"\nBuffer Reuse Benchmark Results:")
        print(f"  Without buffer reuse: {without_reuse_time:.6f}s")
        print(f"  With buffer reuse:    {with_reuse_time:.6f}s")
        
        # With reuse should be faster (less memory allocation)
        if hasattr(with_reuse_time, 'stats'):
            speedup = without_reuse_time.stats.mean / with_reuse_time.stats.mean
            print(f"  Speedup: {speedup:.2f}x")
            assert speedup > 1.0, "Buffer reuse should provide performance improvement"


class TestOptimizationImpact:
    """Test the overall impact of optimizations on real-world scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_end_to_end_performance(self, benchmark):
        """Test end-to-end performance with all optimizations."""
        
        # Create a more complex pipeline
        agent1 = StubAgent(["step1_output"] * 50)
        agent2 = StubAgent(["step2_output"] * 50)
        agent3 = StubAgent(["step3_output"] * 50)
        
        pipeline = (
            Step.solution(agent1) >>
            Step.solution(agent2) >>
            Step.solution(agent3)
        )
        
        runner = create_test_flujo(pipeline)
        
        async def run_complex_pipeline():
            result = await runner.run_async("complex_input")
            return result
        
        # Benchmark the complex pipeline
        execution_time = benchmark(lambda: asyncio.run(run_complex_pipeline()))
        
        print(f"\nEnd-to-End Performance Benchmark:")
        print(f"  Complex pipeline execution: {execution_time:.6f}s")
        
        # Should complete in reasonable time
        assert execution_time < 10.0, f"Complex pipeline too slow: {execution_time:.3f}s"
        
        # Verify the result
        result = asyncio.run(run_complex_pipeline())
        assert result is not None
        assert len(result.step_history) == 3 