import _ini_
class Layer(Module):
    def __init__(self, input_size, output_size, active_func):
        super(Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.active_func = active_func

        # 初始化w和b
        std = np.sqrt(2. / (input_size + output_size))
        self.weights = torch.empty(input_size, output_size, dtype=torch.float32).normal(0, std)
        self.bias = torch.empty(output_size, dtype=torch.float32).normal(0, std)

    def forward(self, input_data):
        result = data_input.mm(self.weights).add(self.bias)
        return result

    def backward(self, grad):
        return input