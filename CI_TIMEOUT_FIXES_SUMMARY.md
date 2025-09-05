🚨 FAILURE SUMMARY

❌ tests/unit/test_cli_performance_edge_cases.py
   Status  : TIMEOUT
   Duration: 181.10s
   Error   : TEST TIMED OUT — outer timeout 180s (per-test timeout 60s)

❌ tests/benchmarks/test_conversational_overhead.py
   Status  : FAIL
   Duration: 2.79s
   Error   : FAILED tests/benchmarks/test_conversational_overhead.py::test_history_manager_overhead_benchmark - assert 1.1076273740000033 < 1.0
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!

❌ tests/integration/test_conversational_loop_nested.py
   Status  : FAIL
   Duration: 1.93s
   Error   : FAILED tests/integration/test_conversational_loop_nested.py::test_nested_conversation_inner_scoped - assert False
 +  where False = any(<generator object test_nested_conversation_inner_scoped.<locals>.<genexpr> at 0x7ac5c3af3440>)
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!

❌ tests/integration/test_conversational_loop_parallel.py
   Status  : FAIL
   Duration: 1.92s
   Error   : FAILED tests/integration/test_conversational_loop_parallel.py::test_conversational_loop_parallel_all_agents - assert False
 +  where False = any(<generator object test_conversational_loop_parallel_all_agents.<locals>.<genexpr> at 0x78cd972a3ac0>)
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!

❌ tests/integration/test_hitl_trace_resume_event.py
   Status  : TIMEOUT
   Duration: 361.10s
   Error   : TEST TIMED OUT — outer timeout 360s (per-test timeout 120s)

❌ tests/unit/test_cli_performance_edge_cases.py
   Status  : TIMEOUT
   Duration: 361.07s
   Error   : TEST TIMED OUT — outer timeout 360s (per-test timeout 120s)

Total failures: 6