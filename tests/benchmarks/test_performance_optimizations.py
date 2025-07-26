"""Performance optimization benchmark tests.

These tests verify that our performance optimizations are working correctly
and provide measurable improvements over standard implementations.
"""

import asyncio
import time

from flujo import Step
from flujo.utils.performance import (
    time_perf_ns,
    time_perf_ns_to_seconds,
    measure_time,
    measure_time_async,
)
from tests.conftest import create_test_flujo


class TestPerformanceOptimizations:
    """Test performance optimizations are working correctly."""

    def test_perf_counter_ns_precision(self, benchmark):
        """Test that perf_counter_ns provides higher precision timing."""
        
        def measure_with_perf_counter():
            start = time.perf_counter()
            time.sleep(0.001)  # 1ms
            end = time.perf_counter()
            return end - start
        
        def measure_with_perf_counter_ns():
            start = time_perf_ns()
            time.sleep(0.001)  # 1ms
            end = time_perf_ns()
            return time_perf_ns_to_seconds(end - start)
        
        perf_counter_time = benchmark(measure_with_perf_counter)
        perf_counter_ns_time = benchmark(measure_with_perf_counter_ns)

        print("\nPrecision Benchmark Results:")
        print(f"  time.perf_counter():     {perf_counter_time:.6f}s")
        print(f"  time.perf_counter_ns():  {perf_counter_ns_time:.6f}s")
        
        # Both should be close to 0.001s (1ms)
        assert 0.0005 < perf_counter_time < 0.002
        assert 0.0005 < perf_counter_ns_time < 0.002

    def test_serialization_performance(self, benchmark):
        """Test that orjson provides faster JSON serialization."""
        
        test_data = {
            "string": "test" * 1000,
            "number": 42,
            "boolean": True,
            "array": list(range(1000)),
            "object": {f"key{i}": f"value{i}" for i in range(100)}
        }
        
        def serialize_with_json():
            import json
            return json.dumps(test_data)
        
        def serialize_with_orjson():
            import orjson
            return orjson.dumps(test_data)
        
        json_time = benchmark(serialize_with_json)
        orjson_time = benchmark(serialize_with_orjson)

        print("\nSerialization Benchmark Results:")
        print(f"  json.dumps():    {json_time:.6f}s")
        print(f"  orjson.dumps():  {orjson_time:.6f}s")
        
        # orjson should be significantly faster
        assert orjson_time < json_time * 0.5  # At least 2x faster

    def test_hashing_performance(self, benchmark):
        """Test that blake3 provides faster hashing."""
        
        test_data = b"test_data" * 10000
        
        def hash_with_hashlib():
            import hashlib
            return hashlib.blake2b(test_data).hexdigest()
        
        def hash_with_blake3():
            import blake3
            return blake3.blake3(test_data).hexdigest()
        
        hashlib_time = benchmark(hash_with_hashlib)
        blake3_time = benchmark(hash_with_blake3)

        print("\nHashing Benchmark Results:")
        print(f"  hashlib.blake2b(): {hashlib_time:.6f}s")
        print(f"  blake3.blake3():   {blake3_time:.6f}s")
        
        # blake3 should be significantly faster
        assert blake3_time < hashlib_time * 0.3  # At least 3x faster

    def test_async_performance_with_uvloop(self, benchmark):
        """Test that uvloop provides better async performance."""
        
        async def run_pipeline():
            flujo = create_test_flujo()
            
            @Step()
            async def test_step(context):
                await asyncio.sleep(0.001)  # Simulate work
                return {"result": "test"}
            
            pipeline = flujo.pipeline([test_step])
            result = await pipeline.run()
            return result
        
        # Use a different approach for async benchmarking
        def run_async_pipeline():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_pipeline())
            finally:
                loop.close()
        
        execution_time = benchmark(run_async_pipeline)

        print("\nAsync Performance Benchmark:")
        print(f"  Pipeline execution: {execution_time:.6f}s")
        
        # Should complete in reasonable time
        assert execution_time < 0.1

    def test_measure_time_decorator(self, benchmark):
        """Test that the measure_time decorator works correctly."""
        
        @measure_time
        def test_function():
            time.sleep(0.001)  # 1ms
            return "test"
        
        # Benchmark the decorated function
        result_time = benchmark(test_function)

        print("\nMeasure Time Decorator Benchmark:")
        print(f"  Decorated function: {result_time:.6f}s")
        
        # Should be reasonable (decorator adds minimal overhead)
        assert result_time < 0.1

    def test_measure_time_async_decorator(self, benchmark):
        """Test that the measure_time_async decorator works correctly."""
        
        @measure_time_async
        async def test_async_function():
            await asyncio.sleep(0.001)  # 1ms
            return "test"
        
        # Use a different approach for async benchmarking
        def run_async_function():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(test_async_function())
            finally:
                loop.close()
        
        # Benchmark the decorated async function
        result_time = benchmark(run_async_function)

        print("\nMeasure Time Async Decorator Benchmark:")
        print(f"  Decorated async function: {result_time:.6f}s")
        
        # Should be reasonable (decorator adds minimal overhead)
        assert result_time < 0.1

    def test_scratch_buffer_reuse(self):
        """Test that scratch buffer reuse works correctly."""
        from flujo.utils.performance import clear_scratch_buffer, get_scratch_buffer

        # Get initial buffer
        buffer1 = get_scratch_buffer()

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
        buffer3 = get_scratch_buffer()

        # Should still be the same buffer
        assert buffer1 is buffer3
        assert len(buffer3) == 0

    def test_buffer_reuse_performance(self, benchmark):
        """Test that buffer reuse provides performance benefits."""
        
        def without_buffer_reuse():
            # Create new buffer each time
            buffer = bytearray()
            for i in range(1000):
                buffer.extend(f"data{i}".encode())
            return len(buffer)
        
        def with_buffer_reuse():
            from flujo.utils.performance import clear_scratch_buffer, get_scratch_buffer
            
            # Reuse buffer
            buffer = get_scratch_buffer()
            for i in range(1000):
                buffer.extend(f"data{i}".encode())
            result = len(buffer)
            clear_scratch_buffer()
            return result
        
        without_reuse_time = benchmark(without_buffer_reuse)
        with_reuse_time = benchmark(with_buffer_reuse)

        print("\nBuffer Reuse Benchmark Results:")
        print(f"  Without buffer reuse: {without_reuse_time:.6f}s")
        print(f"  With buffer reuse:    {with_reuse_time:.6f}s")
        
        # Buffer reuse should be faster due to reduced allocations
        assert with_reuse_time < without_reuse_time * 0.8  # At least 20% faster


class TestOptimizationImpact:
    """Test the overall impact of optimizations on real-world scenarios."""

    def test_end_to_end_performance(self, benchmark):
        """Test end-to-end performance with all optimizations enabled."""
        
        async def run_complex_pipeline():
            flujo = create_test_flujo()
            
            @Step()
            async def step1(context):
                await asyncio.sleep(0.001)
                return {"step1": "done"}
            
            @Step()
            async def step2(context):
                await asyncio.sleep(0.001)
                return {"step2": "done"}
            
            @Step()
            async def step3(context):
                await asyncio.sleep(0.001)
                return {"step3": "done"}
            
            pipeline = flujo.pipeline([step1, step2, step3])
            result = await pipeline.run()
            return result
        
        # Use a different approach for async benchmarking
        def run_complex_pipeline_sync():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_complex_pipeline())
            finally:
                loop.close()
        
        # Benchmark the complex pipeline
        execution_time = benchmark(run_complex_pipeline_sync)

        print("\nEnd-to-End Performance Benchmark:")
        print(f"  Complex pipeline execution: {execution_time:.6f}s")
        
        # Should complete in reasonable time
        assert execution_time < 0.1
