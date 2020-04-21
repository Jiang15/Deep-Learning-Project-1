


class Sequential(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        input_layer = input
        for layer in self.layers:
            input_layer = layer.forward(input_layer)
        return input_layer

    def backward(self, output):
        output_layer = output
        for layer in reversed(self.layers):
            output_layer = layer.backward(output_layer)
        return output_layer

    def param(self):
        out_param =[]
        for layer in self.layers:
            out_param.append(layer.param())
        return out_param
