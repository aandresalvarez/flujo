from __future__ import annotations

import pytest
from typing import List

from flujo import Step
from flujo.domain.models import BaseModel as FlujoBaseModel
from flujo.testing.utils import SimpleDummyRemoteBackend as DummyRemoteBackend, gather_result
from flujo.utils.serialization import register_custom_serializer
from flujo.state.backends.base import _serialize_for_json
import json
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class UserContext(FlujoBaseModel):
    """Context model for user data with mixed scalar and list fields."""

    user_id: str
    name: str
    age: int
    is_active: bool
    preferences: List[str]
    scores: List[float]
    metadata: JSONObject


class ProductContext(FlujoBaseModel):
    """Context model for product data."""

    product_id: str
    name: str
    price: float
    categories: List[str]
    tags: List[str]
    in_stock: bool
    ratings: List[float]


class OrderContext(FlujoBaseModel):
    """Context model for order data with nested structures."""

    order_id: str
    customer: UserContext
    items: List[JSONObject]
    total_amount: float
    status: str
    shipping_address: dict[str, str]


class TestReconstructionIntegration:
    """Integration tests for reconstruction logic in realistic pipeline scenarios."""

    @pytest.mark.asyncio
    async def test_user_profile_pipeline_with_scalar_values(self):
        """Test a pipeline that processes user profiles with scalar values."""

        class UserProfileAgent:
            async def run(self, data: JSONObject, *, context: UserContext) -> JSONObject:
                # Verify that scalar values are preserved correctly
                assert isinstance(context.user_id, str)
                assert isinstance(context.name, str)
                assert isinstance(context.age, int)
                assert isinstance(context.is_active, bool)

                return {
                    "processed": True,
                    "user_id": context.user_id,
                    "name": context.name,
                    "age": context.age,
                    "is_active": context.is_active,
                }

        # Create a user context with scalar values
        user_context = UserContext(
            user_id="user123",
            name="John Doe",
            age=30,
            is_active=True,
            preferences=["music", "sports"],
            scores=[85.5, 92.3, 78.9],
            metadata={"last_login": "2024-01-01", "login_count": 42},
        )

        # Create the pipeline
        step = Step.model_validate({"name": "process_user_profile", "agent": UserProfileAgent()})

        backend = DummyRemoteBackend()
        runner = create_test_flujo(
            step,
            backend=backend,
            context_model=UserContext,
            initial_context_data=user_context.model_dump(),
        )

        # Execute the pipeline
        result = await gather_result(runner, {"action": "process_profile"})

        # Verify the context was reconstructed correctly
        assert result.final_pipeline_context is not None
        assert isinstance(result.final_pipeline_context, UserContext)
        assert result.final_pipeline_context.user_id == "user123"
        assert result.final_pipeline_context.name == "John Doe"
        assert result.final_pipeline_context.age == 30
        assert result.final_pipeline_context.is_active is True
        assert result.final_pipeline_context.preferences == ["music", "sports"]
        assert result.final_pipeline_context.scores == [85.5, 92.3, 78.9]
        assert result.final_pipeline_context.metadata == {
            "last_login": "2024-01-01",
            "login_count": 42,
        }

    @pytest.mark.asyncio
    async def test_product_catalog_pipeline_with_mixed_data_types(self):
        """Test a pipeline that processes product catalog with mixed data types."""

        class ProductCatalogAgent:
            async def run(self, data: JSONObject, *, context: ProductContext) -> JSONObject:
                # Verify that mixed data types are preserved correctly
                assert isinstance(context.product_id, str)
                assert isinstance(context.name, str)
                assert isinstance(context.price, float)
                assert isinstance(context.categories, list)
                assert isinstance(context.tags, list)
                assert isinstance(context.in_stock, bool)
                assert isinstance(context.ratings, list)

                return {
                    "processed": True,
                    "product_id": context.product_id,
                    "name": context.name,
                    "price": context.price,
                    "categories": context.categories,
                    "tags": context.tags,
                    "in_stock": context.in_stock,
                    "ratings": context.ratings,
                }

        # Create a product context with mixed data types
        product_context = ProductContext(
            product_id="prod456",
            name="Laptop Computer",
            price=999.99,
            categories=["Electronics", "Computers"],
            tags=["laptop", "computer", "electronics"],
            in_stock=True,
            ratings=[4.5, 4.2, 4.8, 4.1],
        )

        # Create the pipeline
        step = Step.model_validate(
            {"name": "process_product_catalog", "agent": ProductCatalogAgent()}
        )

        backend = DummyRemoteBackend()
        runner = create_test_flujo(
            step,
            backend=backend,
            context_model=ProductContext,
            initial_context_data=product_context.model_dump(),
        )

        # Execute the pipeline
        result = await gather_result(runner, {"action": "process_catalog"})

        # Verify the context was reconstructed correctly
        assert result.final_pipeline_context is not None
        assert isinstance(result.final_pipeline_context, ProductContext)
        assert result.final_pipeline_context.product_id == "prod456"
        assert result.final_pipeline_context.name == "Laptop Computer"
        assert result.final_pipeline_context.price == 999.99
        assert result.final_pipeline_context.categories == ["Electronics", "Computers"]
        assert result.final_pipeline_context.tags == ["laptop", "computer", "electronics"]
        assert result.final_pipeline_context.in_stock is True
        assert result.final_pipeline_context.ratings == [4.5, 4.2, 4.8, 4.1]

    @pytest.mark.asyncio
    async def test_order_processing_pipeline_with_nested_structures(self):
        """Test a pipeline that processes orders with nested structures."""

        class OrderProcessingAgent:
            async def run(self, data: JSONObject, *, context: OrderContext) -> JSONObject:
                # Verify that nested structures are preserved correctly
                assert isinstance(context.order_id, str)
                assert isinstance(context.customer, UserContext)
                assert isinstance(context.items, list)
                assert isinstance(context.total_amount, float)
                assert isinstance(context.status, str)
                assert isinstance(context.shipping_address, dict)

                # Verify nested customer data
                customer = context.customer
                assert isinstance(customer.user_id, str)
                assert isinstance(customer.name, str)
                assert isinstance(customer.age, int)
                assert isinstance(customer.is_active, bool)
                assert isinstance(customer.preferences, list)
                assert isinstance(customer.scores, list)
                assert isinstance(customer.metadata, dict)

                return {
                    "processed": True,
                    "order_id": context.order_id,
                    "customer_id": context.customer.user_id,
                    "total_amount": context.total_amount,
                    "status": context.status,
                }

        # Create nested context structures
        customer = UserContext(
            user_id="user789",
            name="Jane Smith",
            age=28,
            is_active=True,
            preferences=["books", "travel"],
            scores=[88.5, 91.2, 85.7],
            metadata={"member_since": "2023-06-15", "order_count": 15},
        )

        order_context = OrderContext(
            order_id="order123",
            customer=customer,
            items=[
                {"product_id": "prod1", "quantity": 2, "price": 29.99},
                {"product_id": "prod2", "quantity": 1, "price": 49.99},
            ],
            total_amount=109.97,
            status="pending",
            shipping_address={
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip": "12345",
            },
        )

        # Create the pipeline
        step = Step.model_validate({"name": "process_order", "agent": OrderProcessingAgent()})

        backend = DummyRemoteBackend()
        runner = create_test_flujo(
            step,
            backend=backend,
            context_model=OrderContext,
            initial_context_data=order_context.model_dump(),
        )

        # Execute the pipeline
        result = await gather_result(runner, {"action": "process_order"})

        # Verify the context was reconstructed correctly
        assert result.final_pipeline_context is not None
        assert isinstance(result.final_pipeline_context, OrderContext)
        assert result.final_pipeline_context.order_id == "order123"
        assert result.final_pipeline_context.total_amount == 109.97
        assert result.final_pipeline_context.status == "pending"

        # Verify nested customer data
        customer = result.final_pipeline_context.customer
        assert isinstance(customer, UserContext)
        assert customer.user_id == "user789"
        assert customer.name == "Jane Smith"
        assert customer.age == 28
        assert customer.is_active is True
        assert customer.preferences == ["books", "travel"]
        assert customer.scores == [88.5, 91.2, 85.7]
        assert customer.metadata == {"member_since": "2023-06-15", "order_count": 15}

        # Verify items and shipping address
        assert len(result.final_pipeline_context.items) == 2
        assert result.final_pipeline_context.items[0]["product_id"] == "prod1"
        assert result.final_pipeline_context.items[0]["quantity"] == 2
        assert result.final_pipeline_context.items[0]["price"] == 29.99

        assert result.final_pipeline_context.shipping_address["street"] == "123 Main St"
        assert result.final_pipeline_context.shipping_address["city"] == "Anytown"
        assert result.final_pipeline_context.shipping_address["state"] == "CA"
        assert result.final_pipeline_context.shipping_address["zip"] == "12345"

    @pytest.mark.asyncio
    async def test_multi_step_pipeline_with_context_updates(self):
        """Test a multi-step pipeline that updates context between steps."""

        class Step1Agent:
            async def run(self, data: JSONObject, *, context: "UserContext") -> JSONObject:
                # Update context with new data
                context.age += 1
                context.scores.append(95.0)
                context.metadata["step1_processed"] = True

                return {"step": 1, "processed": True}

        class Step2Agent:
            async def run(self, data: JSONObject, *, context: "UserContext") -> JSONObject:
                # Verify context updates from previous step
                assert context.age == 31  # Should be incremented from 30
                assert len(context.scores) == 4  # Should have one more score
                assert context.metadata["step1_processed"] is True

                # Add more updates
                context.preferences.append("gaming")
                context.metadata["step2_processed"] = True

                return {"step": 2, "processed": True}

        class Step3Agent:
            async def run(self, data: JSONObject, *, context: "UserContext") -> JSONObject:
                # Verify all context updates are preserved
                assert context.age == 31
                assert len(context.scores) == 4
                assert context.scores[-1] == 95.0
                assert len(context.preferences) == 3  # Should have "gaming" added
                assert context.metadata["step1_processed"] is True
                assert context.metadata["step2_processed"] is True

                return {"step": 3, "processed": True}

        # Create initial user context
        user_context = UserContext(
            user_id="user456",
            name="Alice Johnson",
            age=30,
            is_active=True,
            preferences=["music", "sports"],
            scores=[85.5, 92.3, 78.9],
            metadata={"initial": True},
        )

        # Create the multi-step pipeline
        step1 = Step.model_validate({"name": "step1", "agent": Step1Agent()})
        step2 = Step.model_validate({"name": "step2", "agent": Step2Agent()})
        step3 = Step.model_validate({"name": "step3", "agent": Step3Agent()})

        backend = DummyRemoteBackend()
        runner = create_test_flujo(
            step1 >> step2 >> step3,
            backend=backend,
            context_model=UserContext,
            initial_context_data=user_context.model_dump(),
        )

        # Execute the pipeline
        result = await gather_result(runner, {"action": "multi_step_process"})

        # Verify final context state
        assert result.final_pipeline_context is not None
        assert isinstance(result.final_pipeline_context, UserContext)
        assert result.final_pipeline_context.user_id == "user456"
        assert result.final_pipeline_context.name == "Alice Johnson"
        assert result.final_pipeline_context.age == 31  # Should be incremented
        assert result.final_pipeline_context.is_active is True
        assert result.final_pipeline_context.preferences == ["music", "sports", "gaming"]
        assert result.final_pipeline_context.scores == [85.5, 92.3, 78.9, 95.0]
        assert result.final_pipeline_context.metadata["initial"] is True
        assert result.final_pipeline_context.metadata["step1_processed"] is True
        assert result.final_pipeline_context.metadata["step2_processed"] is True

    @pytest.mark.asyncio
    async def test_regression_bug_integration_test(self):
        """Integration test specifically for the regression bug fix."""

        class RegressionTestAgent:
            async def run(self, data: JSONObject, *, context: "UserContext") -> JSONObject:
                # This test specifically checks that scalar values are NOT wrapped in lists
                # The bug was that scalar values were being converted to lists

                # Verify scalar values are preserved as their original types
                assert isinstance(context.user_id, str)  # Should be str, not List[str]
                assert isinstance(context.name, str)  # Should be str, not List[str]
                assert isinstance(context.age, int)  # Should be int, not List[int]
                assert isinstance(context.is_active, bool)  # Should be bool, not List[bool]

                # Verify list values are preserved as lists
                assert isinstance(context.preferences, list)  # Should be List[str]
                assert isinstance(context.scores, list)  # Should be List[float]
                assert isinstance(context.metadata, dict)

                # Verify the actual values
                assert context.user_id == "test_user"
                assert context.name == "Test User"
                assert context.age == 25
                assert context.is_active is True
                assert context.preferences == ["pref1", "pref2"]
                assert context.scores == [85.5, 92.3]
                assert context.metadata == {"key": "value"}

                return {"test_passed": True}

        # Create context with mixed scalar and list values
        user_context = UserContext(
            user_id="test_user",  # Scalar string
            name="Test User",  # Scalar string
            age=25,  # Scalar int
            is_active=True,  # Scalar bool
            preferences=["pref1", "pref2"],  # List of strings
            scores=[85.5, 92.3],  # List of floats
            metadata={"key": "value"},  # Dict
        )

        # Create the pipeline
        step = Step.model_validate({"name": "regression_test", "agent": RegressionTestAgent()})

        backend = DummyRemoteBackend()
        runner = create_test_flujo(
            step,
            backend=backend,
            context_model=UserContext,
            initial_context_data=user_context.model_dump(),
        )

        # Execute the pipeline
        result = await gather_result(runner, {"test": "data"})

        # Verify the test passed (no exceptions raised)
        assert result.step_history[0].output is not None, (
            "Step output is None (step may have failed)"
        )
        assert result.step_history[0].output["test_passed"] is True

        # Verify final context is still correct
        final_context = result.final_pipeline_context
        assert final_context is not None
        assert isinstance(final_context, UserContext)
        assert final_context.user_id == "test_user"
        assert final_context.name == "Test User"
        assert final_context.age == 25
        assert final_context.is_active is True
        assert final_context.preferences == ["pref1", "pref2"]
        assert final_context.scores == [85.5, 92.3]
        assert final_context.metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_circular_reference_in_dict_keys_in_pipeline():
    """Integration test: circular/self-referential objects as dict keys in pipeline context/output."""

    class Node:
        def __init__(self, name):
            self.name = name
            self.ref = None

    # Register a custom serializer for Node
    register_custom_serializer(Node, lambda obj: {"name": obj.name, "has_ref": obj.ref is not None})

    node1 = Node("node1")
    node2 = Node("node2")
    node1.ref = node2
    node2.ref = node1

    test_dict = {node1: "a", node2: "b", "plain": "c"}

    # Simulate pipeline context/output serialization
    normalized = _serialize_for_json({"context": test_dict})
    result = json.loads(json.dumps(normalized, ensure_ascii=False))
    assert isinstance(result, dict)
    assert "context" in result
    # Should not raise RecursionError or stack overflow
    assert any("node1" in str(k) or "node2" in str(k) for k in result["context"].keys())
