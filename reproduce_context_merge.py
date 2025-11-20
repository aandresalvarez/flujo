from pydantic import BaseModel
from flujo.application.core.context_manager import ContextManager
import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.DEBUG)


class MyContext(BaseModel):
    run_id: str

    def __setattr__(self, name, value):
        if name == "run_id":
            super().__setattr__(name, value)
        # Note: Missing handling for other attributes!


def test_merge():
    original = MyContext(run_id="original")
    isolated = ContextManager.isolate(original)

    print(f"Original run_id: {original.run_id}")
    print(f"Isolated run_id: {isolated.run_id}")

    isolated.run_id = "modified"
    print(f"Isolated run_id after modification: {isolated.run_id}")

    merged = ContextManager.merge(original, isolated)

    print(f"Merged run_id: {merged.run_id}")

    if merged.run_id == "modified":
        print("SUCCESS: Context merged correctly")
    else:
        print("FAILURE: Context NOT merged correctly")


if __name__ == "__main__":
    test_merge()
