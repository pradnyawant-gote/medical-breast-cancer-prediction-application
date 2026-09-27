"""Convert model3.h5 to a smaller LiteRT inference artifact.

Run with the development environment: python convert_model_free.py
The source model and the output should be versioned together.
"""

from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model3.h5"
OUTPUT = ROOT / "model_free.tflite"


def main():
    model = tf.keras.models.load_model(SOURCE, compile=False)
    if model.input_shape != (None, 128, 128, 3) or model.output_shape[-1] != 3:
        raise ValueError("Unexpected model input or output shape")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converted = converter.convert()
    OUTPUT.write_bytes(converted)
    print(f"Saved {OUTPUT.name}: {len(converted):,} bytes")


if __name__ == "__main__":
    main()
