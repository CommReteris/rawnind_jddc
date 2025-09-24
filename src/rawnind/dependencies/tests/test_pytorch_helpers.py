import pytest
import torch
from torch.optim import AdamW
from torch.nn import Linear

from src.rawnind.dependencies.pytorch_helpers import get_basic_linear_schedule_with_warmup

def test_get_basic_linear_schedule_with_warmup():
    """
    Test linear learning rate scheduler with warmup for correct behavior.

    Objective: Verify that the scheduler provides correct learning rate progression with warmup and decay.
    Test criteria: Initial LR is 0, reaches base LR after warmup, decays to 0 by end of training.
    How testing for this criteria fulfills purpose: Ensures proper learning rate scheduling for model training.
    What components are mocked, monkeypatched, or are fixtures: None - direct PyTorch scheduler operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure scheduler logic testing without external dependencies.
    """
    model = Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = 100
    num_warmup_steps = 10

    scheduler = get_basic_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Check initial LR (should be 0 because LambdaLR sets initial LR based on lr_lambda at step -1 by default)
    # The first step() call will then apply the LR for step 0.
    assert scheduler.get_last_lr()[0] == 0.0

    # Simulate warmup phase
    for _ in range(num_warmup_steps):
        scheduler.step()
    
    # Check LR after warmup (should reach base_lr)
    # At current_step == num_warmup_steps, lr_lambda returns num_warmup_steps / num_warmup_steps = 1.0
    assert pytest.approx(scheduler.get_last_lr()[0], abs=1e-6) == 1e-5

    # Simulate decay phase
    # Total steps are num_training_steps. Warmup is num_warmup_steps.
    # Remaining steps for decay are num_training_steps - num_warmup_steps.
    # The loop should iterate from num_warmup_steps up to num_training_steps - 1.
    for _ in range(num_warmup_steps, num_training_steps - 1):
        scheduler.step()
    
    # Check LR at the end (should be close to 0)
    scheduler.step() # Final step corresponding to num_training_steps - 1
    assert pytest.approx(scheduler.get_last_lr()[0], abs=1e-6) == 0.0