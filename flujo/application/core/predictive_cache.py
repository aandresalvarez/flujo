"""Predictive caching for step execution optimization."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

from ...domain.models import StepResult


@dataclass
class CachePrediction:
    """Prediction for cache preloading."""

    step_name: str
    confidence: float
    expected_input: Any
    priority: int = 0


class PredictiveCache:
    """Predictive caching system for step execution."""

    def __init__(self, max_predictions: int = 10):
        self.max_predictions = max_predictions
        self.execution_history: deque[tuple[str, Any, float]] = deque(maxlen=1000)
        self.step_patterns: Dict[str, List[str]] = defaultdict(list)
        self.prediction_accuracy: Dict[str, float] = defaultdict(lambda: 0.5)
        self.current_predictions: List[CachePrediction] = []

    def record_execution(self, step_name: str, input_data: Any, execution_time: float) -> None:
        """Record a step execution for pattern analysis."""
        self.execution_history.append((step_name, input_data, execution_time))

        # Update step patterns
        if len(self.execution_history) >= 2:
            prev_step = self.execution_history[-2][0]
            current_step = step_name
            if prev_step != current_step:
                self.step_patterns[prev_step].append(current_step)

    def predict_next_steps(self, current_step: str, current_input: Any) -> List[CachePrediction]:
        """Predict which steps are likely to be executed next."""
        predictions = []

        # Pattern-based prediction
        if current_step in self.step_patterns:
            for next_step in self.step_patterns[current_step]:
                confidence = self.prediction_accuracy.get(next_step, 0.5)
                predictions.append(
                    CachePrediction(
                        step_name=next_step,
                        confidence=confidence,
                        expected_input=current_input,
                        priority=int(confidence * 100),
                    )
                )

        # Input-based prediction (for similar inputs, predict similar step sequences)
        similar_inputs = self._find_similar_inputs(current_input)
        for step_name, input_data, _ in similar_inputs:
            if step_name != current_step:
                confidence = 0.7  # Higher confidence for input-based prediction
                predictions.append(
                    CachePrediction(
                        step_name=step_name,
                        confidence=confidence,
                        expected_input=input_data,
                        priority=int(confidence * 100),
                    )
                )

        # Sort by priority and limit
        predictions.sort(key=lambda p: p.priority, reverse=True)
        return predictions[: self.max_predictions]

    def _find_similar_inputs(self, current_input: Any) -> List[tuple[str, Any, float]]:
        """Find historical inputs similar to the current input."""
        similar_inputs = []

        # Simple similarity based on type and structure
        current_type = type(current_input)
        current_str = str(current_input)[:100]  # First 100 chars for comparison

        for step_name, input_data, execution_time in self.execution_history:
            if isinstance(input_data, current_type):
                input_str = str(input_data)[:100]
                similarity = self._calculate_similarity(current_str, input_str)
                if similarity > 0.8:  # 80% similarity threshold
                    similar_inputs.append((step_name, input_data, execution_time))

        return similar_inputs

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple algorithm."""
        if not str1 or not str2:
            return 0.0

        # Simple character-based similarity
        common_chars = sum(1 for c in str1 if c in str2)
        total_chars = len(str1) + len(str2)
        return common_chars / total_chars if total_chars > 0 else 0.0

    def update_prediction_accuracy(self, predicted_step: str, was_executed: bool) -> None:
        """Update prediction accuracy based on actual execution."""
        current_accuracy = self.prediction_accuracy[predicted_step]

        if was_executed:
            # Increase accuracy for correct predictions
            new_accuracy = min(1.0, current_accuracy + 0.1)
        else:
            # Decrease accuracy for incorrect predictions
            new_accuracy = max(0.0, current_accuracy - 0.05)

        self.prediction_accuracy[predicted_step] = new_accuracy

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about prediction accuracy."""
        if not self.prediction_accuracy:
            return {}

        accuracies = list(self.prediction_accuracy.values())
        return {
            "total_predictions": len(self.prediction_accuracy),
            "average_accuracy": sum(accuracies) / len(accuracies),
            "best_predictions": sorted(
                self.prediction_accuracy.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "worst_predictions": sorted(self.prediction_accuracy.items(), key=lambda x: x[1])[:5],
        }


class AsyncPredictiveCache(PredictiveCache):
    """Asynchronous version of predictive cache with background preloading."""

    def __init__(self, max_predictions: int = 10):
        super().__init__(max_predictions)
        self.preload_tasks: Dict[str, asyncio.Task[Any]] = {}
        self.cache_store: Dict[str, StepResult] = {}

    async def preload_predicted_steps(
        self, predictions: List[CachePrediction], step_executor: Any
    ) -> None:
        """Preload predicted steps in the background."""
        for prediction in predictions:
            if prediction.step_name not in self.preload_tasks:
                # Create background task for preloading
                task = asyncio.create_task(self._preload_step(prediction, step_executor))
                self.preload_tasks[prediction.step_name] = task

    async def _preload_step(self, prediction: CachePrediction, step_executor: Any) -> None:
        """Preload a single step in the background."""
        try:
            # This would need the actual step object, which we don't have here
            # In a real implementation, you'd need to get the step from the pipeline
            pass
        except Exception as e:
            # Log error but don't fail
            print(f"Preload failed for {prediction.step_name}: {e}")
        finally:
            # Clean up task
            if prediction.step_name in self.preload_tasks:
                del self.preload_tasks[prediction.step_name]

    def get_cached_result(self, step_name: str) -> Optional[StepResult]:
        """Get a cached result if available."""
        return self.cache_store.get(step_name)

    def set_cached_result(self, step_name: str, result: StepResult) -> None:
        """Set a cached result."""
        self.cache_store[step_name] = result

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache_store.clear()
        # Cancel all preload tasks
        for task in self.preload_tasks.values():
            task.cancel()
        self.preload_tasks.clear()
