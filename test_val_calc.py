#!/usr/bin/env python3
"""Test script to verify validation interval calculations"""

def test_val_calculation():
    # Test case 1: 1000 samples, batch size 4, val_every_fraction 0.05
    train_ds_len = 1000
    batch_size = 4
    val_every_fraction = 0.05

    total_iters = max(1, train_ds_len // batch_size)
    val_every = max(1, int(total_iters * val_every_fraction))
    val_every_percent = (val_every / total_iters) * 100

    print(f"Test Case 1:")
    print(f"  Dataset: {train_ds_len} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Validation every: {val_every} iterations ({val_every_percent:.1f}%)")
    print(f"  Validation at iterations: {list(range(val_every, total_iters+1, val_every))}")
    print()

    # Test case 2: Small dataset
    train_ds_len = 50
    batch_size = 4
    val_every_fraction = 0.2

    total_iters = max(1, train_ds_len // batch_size)
    val_every = max(1, int(total_iters * val_every_fraction))
    val_every_percent = (val_every / total_iters) * 100

    print(f"Test Case 2:")
    print(f"  Dataset: {train_ds_len} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Validation every: {val_every} iterations ({val_every_percent:.1f}%)")
    print(f"  Validation at iterations: {list(range(val_every, total_iters+1, val_every))}")
    print()

    # Test next_val calculation
    print("Next validation calculation test:")
    for global_iter in [0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25]:
        if global_iter % 5 == 0 or global_iter == 25:  # Print every 5th iter and final
            next_val = ((global_iter // 5) + 1) * 5
            if next_val > 25:
                next_val_str = "end"
            else:
                next_val_str = f"{next_val}"
            print(f"  iter {global_iter:2d}/25  [next val at iter {next_val_str}]")

if __name__ == "__main__":
    test_val_calculation()