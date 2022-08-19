import os
import argparse
import tensorflow as tf


unweighted_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
weighting = 1  # Not used for inference, just set to 1


def weighted_bce(y_true, y_pred):
    weights = (y_true * (weighting - 1)) + 1
    bce = tf.expand_dims(unweighted_bce(y_true, y_pred), -1)
    return tf.reduce_mean(bce * weights)


def save_model_weights(model_path):
    if os.path.splitext(model_path)[1] == '.pb':
        model_path = os.path.split(model_path)[0]
    output_path = f'{os.path.splitext(model_path)[0]}_weights.h5'
    model = tf.keras.models.load_model(model_path, custom_objects={'weighted_bce': weighted_bce, 'weighted_cce': weighted_cce})
    model.save_weights(output_path)
    model.summary()
    
    filters = []
    for layer in model.layers:
        try:
            if 'conv2d_transpose' in layer.get_config()['name']:
                filters.append(layer.get_config()['filters'])
        except NotImplementedError:
            continue  # Skip layers for which the get_config method is not implemented
        
    with open(f'{os.path.splitext(model_path)[0]}.config', 'w') as f:
        f.write(f"{model.layers[0].get_config()['batch_input_shape'][1:]};{model.outputs[0].shape[-1]};{filters[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a model and save its weights in hdf5 format')
    parser.add_argument('-model_path', type=str, default="Model.h5", help='Path to the model.')

    args = parser.parse_args()
    model_path = args.model_path

    save_model_weights(model_path)
    