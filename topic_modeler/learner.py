import torch
from topic_modeler.modeler import Modeler


class LinearLogisticClassifier(torch.nn.Module):
    def __init__(self, n, o):
        super().__init__()
        self.input_layer = torch.nn.Linear(n, o, bias=True)
        self.output_layer = torch.nn.Softmax(1)

    def forward(self, xb):
        xb = self.input_layer(xb)
        return self.output_layer(xb)


class Learner():
    def __init__(self, modeler: Modeler):
        self.data_modeler = modeler
        self.input_size = len(modeler.word_mapping)
        self.output_size = len(modeler.topic_to_index)
        self.model = LinearLogisticClassifier(
            self.input_size, self.output_size)
        self.model = self.model.float()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.losses = []

    def learn(self, iterations, learning_rate):
        self.model.train()

        for i in range(iterations):
            output = self.model(self.data_modeler.feature_matrix)
            loss = self.loss_func(output, self.data_modeler.train_targets)
            loss.backward()

            with torch.no_grad():
                for p in self.model.parameters():
                    p.sub_(learning_rate * p.grad)
                    p.grad.zero_()

            self.losses.append(loss)

    def get_preds(self, sentence_series):

        pred_feature_matrix = self.data_modeler.get_feature_matrix(
            sentence_series)
        pred_feature_matrix = torch.from_numpy(pred_feature_matrix)
        pred_feature_matrix = pred_feature_matrix.float()

        self.model.train(mode=False)
        with torch.no_grad():
            probs = self.model(pred_feature_matrix)

        preds = []

        for i in range(probs.shape[0]):
            li = list(probs[i])
            index = li.index(max(li))
            preds.append(self.data_modeler.index_to_topic[index])

        return preds, probs
