import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import keras
import math

import model_architecture

def convert_model(weights_path, input_shape=(384, 384, 1), output_channels=1, filters=16):
    output_model_path, output_model_name = os.path.split(weights_path)
    output_model_name = f'{os.path.splitext(weights_path)[0]}.pb'.replace('_weights', '')
    
    # Make sure the input has a size that can be downsampled at least 4 times (input for inference can be different from input during training for FCNs such as UNets)
    input_height = math.ceil(input_shape[0]/(2**4)) * 2**4
    input_width = math.ceil(input_shape[1]/(2**4)) * 2**4
    input_channels = input_shape[2]
    model = model_architecture.create_model((input_height, input_width, input_channels), output_channels, filters)
    model.load_weights(weights_path)

    with keras.backend.get_session() as sess:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [node.op.name for node in model.outputs])
        graph_io.write_graph(constant_graph, output_model_path, output_model_name, as_text=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert build a MultiRes UNet model, load weights from file, and save as .pb for tensorflow 1.12')
    parser.add_argument('-weights_path', type=str, help='Path to the model weights file.')
    parser.add_argument('--input_shape', type=str, default="(384,384,1)", help='Input shape for the exported model (since it is an FCN, it can be different from the input shape of the original model).')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels of the original model from which the weights were saved.')
    parser.add_argument('--filters', type=int, default=16, help='Number of filters of the original model from which the weights were saved.')

    args = parser.parse_args()
    weights_path = args.weights_path
    config_path = f'{os.path.splitext(weights_path)[0]}.config'.replace('_weights', '')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            params = f.read()
        params = params.split(';')
        input_shape = [int(i) for i in params[0].replace('(', '').replace(')', '').replace(' ', '').split(',')]
        output_channels = int(params[1])
        filters = int(params[2])
        print(f'Config file found. Using parameters input_shape={input_shape}, output_channels={output_channels}, filters={filters}.')
    else:
        input_shape = [int(i) for i in args.input_shape.replace('(', '').replace(')', '').replace(' ', '').split(',')]
        output_channels = args.output_channels
        filters = args.filters

    convert_model(weights_path, input_shape, output_channels, filters)
    