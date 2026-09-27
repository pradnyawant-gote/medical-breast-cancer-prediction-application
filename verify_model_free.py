"""Check converted model output against the source on deterministic test inputs."""

from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent


def main():
    source = tf.keras.models.load_model(ROOT / "model3.h5", compile=False)
    lite = tf.lite.Interpreter(model_path=str(ROOT / "model_free.tflite"), num_threads=1)
    lite.allocate_tensors()
    input_index = lite.get_input_details()[0]["index"]
    output_index = lite.get_output_details()[0]["index"]

    rng = np.random.default_rng(2026)
    inputs = [
        np.zeros((1, 128, 128, 3), dtype=np.float32),
        np.ones((1, 128, 128, 3), dtype=np.float32),
    ]
    inputs.extend(rng.random((1, 128, 128, 3), dtype=np.float32) for _ in range(8))
    largest_delta = 0.0
    for batch in inputs:
        expected = np.asarray(source(batch, training=False))
        lite.set_tensor(input_index, batch)
        lite.invoke()
        actual = lite.get_tensor(output_index)
        largest_delta = max(largest_delta, float(np.max(np.abs(expected - actual))))
        if int(np.argmax(expected)) != int(np.argmax(actual)):
            raise AssertionError(f"Class mismatch: {expected} versus {actual}")
    print(f"10 test inputs agreed on class; largest score difference: {largest_delta:.6f}")
    if largest_delta > 0.02:
        raise AssertionError("Converted scores differ by more than 0.02")


if __name__ == "__main__":
    main()
