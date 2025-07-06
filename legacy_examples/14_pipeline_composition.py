"""
Example 14: Enhanced Pipeline Composition and Sequencing (FSD 2.1)

This example demonstrates how to use the new `Pipeline >> Pipeline` operator
to create clean, modular, multi-stage workflows by chaining independent pipelines.

The example shows:
1. Building independent pipelines for different concerns
2. Chaining them together using the >> operator
3. Maintaining type safety across pipeline boundaries
4. Sharing context and observability across the entire workflow

This matches the use case described in the FSD where you want to break complex
workflows into logical, independent pipelines and then chain them together.
"""

import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.testing import StubAgent


class ConceptResolutionContext(BaseModel):
    """Context for concept resolution pipeline."""
    resolved_concepts: List[str] = []
    confidence_scores: Dict[str, float] = {}


class SQLGenerationContext(BaseModel):
    """Context for SQL generation pipeline."""
    generated_sql: str = ""
    validation_errors: List[str] = []


class MasterContext(BaseModel):
    """Combined context for the master pipeline."""
    resolved_concepts: List[str] = []
    confidence_scores: Dict[str, float] = {}
    generated_sql: str = ""
    validation_errors: List[str] = []


class ConceptResolutionAgent:
    """Agent that resolves concepts from text input."""
    
    async def run(self, data: str, **kwargs) -> Dict[str, Any]:
        # Simulate concept resolution using an LLM
        # In a real application, this would be a powerful LLM agent
        concepts = ["users", "orders", "products"]
        scores = {"users": 0.95, "orders": 0.87, "products": 0.92}
        
        return {
            "concepts": concepts,
            "confidence_scores": scores,
            "input_text": data
        }


class SQLGenerationAgent:
    """Agent that generates SQL from resolved concepts."""
    
    async def run(self, data: Dict[str, Any], **kwargs) -> str:
        # Simulate SQL generation based on resolved concepts
        # In a real application, this would be an LLM agent
        concepts = data.get("concepts", [])
        sql = f"SELECT * FROM {', '.join(concepts)} WHERE 1=1;"
        return sql


class SQLValidationAgent:
    """Agent that validates generated SQL."""
    
    async def run(self, data: str, **kwargs) -> Dict[str, Any]:
        # Simulate SQL validation
        # In a real application, this could use a SQL parser or LLM
        is_valid = "SELECT" in data and "FROM" in data
        return {
            "sql": data,
            "is_valid": is_valid,
            "errors": [] if is_valid else ["Invalid SQL syntax"]
        }


def build_concept_pipeline() -> Pipeline[str, Dict[str, Any]]:
    """
    Build the concept resolution pipeline.
    
    This pipeline takes natural language text and outputs resolved concepts
    with confidence scores. It's a complete, independent pipeline that can
    be tested and deployed separately.
    """
    concept_agent = ConceptResolutionAgent()
    
    # Step 1: Resolve concepts from text
    resolve_step = Step.model_validate({
        "name": "resolve_concepts",
        "agent": concept_agent,
        "updates_context": True,
        "persist_feedback_to_context": "concept_resolution_feedback"
    })
    
    return Pipeline.from_step(resolve_step)


def build_sql_pipeline() -> Pipeline[Dict[str, Any], Dict[str, Any]]:
    """
    Build the SQL generation and validation pipeline.
    
    This pipeline takes resolved concepts and outputs validated SQL.
    It's a complete, independent pipeline that can be tested and deployed separately.
    """
    sql_gen_agent = SQLGenerationAgent()
    sql_val_agent = SQLValidationAgent()
    
    # Step 1: Generate SQL from resolved concepts
    generate_step = Step.model_validate({
        "name": "generate_sql",
        "agent": sql_gen_agent,
        "updates_context": True,
        "persist_feedback_to_context": "sql_generation_feedback"
    })
    
    # Step 2: Validate the generated SQL
    validate_step = Step.model_validate({
        "name": "validate_sql",
        "agent": sql_val_agent,
        "updates_context": True,
        "persist_validation_results_to": "sql_validation_results"
    })
    
    return generate_step >> validate_step


def build_master_pipeline() -> Pipeline[str, Dict[str, Any]]:
    """
    Build the master pipeline by chaining concept and SQL pipelines.
    
    This demonstrates the key feature from the FSD: using the >> operator
    to chain complete pipelines together, creating a clean, readable workflow
    that matches the mental model of the process.
    """
    # 1. Build each independent pipeline
    concept_pipeline = build_concept_pipeline()
    sql_pipeline = build_sql_pipeline()
    
    # 2. Chain them together using the enhanced >> operator
    # The resulting pipeline takes text and outputs validated SQL
    master_pipeline = concept_pipeline >> sql_pipeline
    
    return master_pipeline


async def main() -> None:
    print("🧠 Enhanced Pipeline Composition and Sequencing Example")
    print("=" * 60)
    
    # Build the master pipeline using the new composition feature
    master_pipeline = build_master_pipeline()
    
    print(f"\n📋 Master Pipeline Structure:")
    print(f"   Input Type: str (natural language query)")
    print(f"   Output Type: Dict[str, Any] (validated SQL)")
    print(f"   Steps: {[step.name for step in master_pipeline.steps]}")
    
    # Create a Flujo runner for our composed pipeline
    runner = Flujo(master_pipeline, context_model=MasterContext)
    
    # Test queries
    test_queries = [
        "Find all users and their orders",
        "Show me products with high ratings",
        "Get customer data with purchase history"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: Processing query: '{query}'")
        
        # Execute the pipeline
        result = None
        async for item in runner.run_async(query):
            result = item
        
        if result and result.success:
            # Inspect the results from the pipeline's step_history
            concept_result = result.step_history[0]
            sql_gen_result = result.step_history[1]
            sql_val_result = result.step_history[2]
            
            print(f"   ✅ Concept Resolution ('{concept_result.name}'):")
            print(f"      - Concepts: {concept_result.output.get('concepts', [])}")
            print(f"      - Confidence: {concept_result.output.get('confidence_scores', {})}")
            
            print(f"   ✅ SQL Generation ('{sql_gen_result.name}'):")
            print(f"      - Generated SQL: {sql_gen_result.output}")
            
            print(f"   ✅ SQL Validation ('{sql_val_result.name}'):")
            print(f"      - Valid: {sql_val_result.output.get('is_valid', False)}")
            print(f"      - Final SQL: {sql_val_result.output.get('sql', '')}")
            
            # Show context sharing
            context = result.final_pipeline_context
            print(f"   📊 Context Sharing:")
            print(f"      - Concept feedback persisted: {hasattr(context, 'concept_resolution_feedback')}")
            print(f"      - SQL feedback persisted: {hasattr(context, 'sql_generation_feedback')}")
            print(f"      - Validation results persisted: {hasattr(context, 'sql_validation_results')}")
        else:
            print(f"   ❌ Pipeline failed: {result.feedback if result else 'Unknown error'}")
    
    print(f"\n🎉 Pipeline composition example completed!")
    print(f"\nKey Benefits Demonstrated:")
    print(f"   ✅ Clean separation of concerns with independent pipelines")
    print(f"   ✅ Simple composition using the >> operator")
    print(f"   ✅ Unified context and observability across all steps")
    print(f"   ✅ Type safety maintained across pipeline boundaries")
    print(f"   ✅ Backward compatibility with existing Step >> Step operations")


if __name__ == "__main__":
    asyncio.run(main()) 