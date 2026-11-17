import warp as wp


@wp.kernel
def increment(values: wp.array(dtype=wp.int32)):
    index = wp.tid()
    values[index] += 1


def main():
    wp.init()
    values = wp.zeros(257, dtype=wp.int32, device="cuda:0")
    wp.launch(increment, dim=values.size, inputs=[values], device="cuda:0")
    wp.synchronize()
    result = values.numpy()
    assert result.shape == (257,)
    assert (result == 1).all()
    print(f"PASS: {result.size} gfx936 values incremented")


if __name__ == "__main__":
    main()
