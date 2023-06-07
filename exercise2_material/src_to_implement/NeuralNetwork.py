import copy


class NeuralNetwork:
    def __init__(self,optimizer,weights_initializer,bias_initializer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.loss_layer = None
        self.data_layer = None

        self.weight_initializer = weights_initializer
        self.bias_initializer = bias_initializer



    def forward(self):

        input_tensor,label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

        data_tensor = input_tensor.copy()
        for layer in self.layers:
            data_tensor = layer.forward(data_tensor)

        loss = self.loss_layer.forward(data_tensor,label_tensor)

        return loss

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_layer(self,layer):

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weight_initializer,self.bias_initializer)

        self.layers.append(layer)


    def train(self,iterations):

        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)


    def test(self,input_tensor):
        data_tensor = input_tensor.copy()

        for layer in self.layers:
            data_tensor = layer.forward(data_tensor)

        return data_tensor










