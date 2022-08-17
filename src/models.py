import torch
from transformers import BertModel

class BERT_baseline(torch.nn.Module):
    def __init__(self, config):
        super(BERT_baseline, self).__init__()

        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        #         self.linear1 = torch.nn.Linear(bert_output_shape, 128)
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(config.output_layer_params.n_inputs, config.output_layer_params.n_outputs)
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
            'last_hidden_state']
        x = x[:, 0, :]
        #         x = self.linear1(x)
        #         x = self.activation(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class BERT_with_metadata(torch.nn.Module):
    def __init__(self, config):
        super(BERT_with_metadata, self).__init__()

        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.sources_linear = torch.nn.Linear(config.source_layer_params.n_inputs, config.source_layer_params.n_neurons)
        self.sources_relu = torch.nn.ReLU()

        self.subjects_linear = torch.nn.Linear(config.subject_layer_params.n_inputs, config.subject_layer_params.n_neurons)
        self.subjects_relu = torch.nn.ReLU()

        self.bert_linear = torch.nn.Linear(config.bert_layer_params.n_inputs, config.bert_layer_params.n_neurons)
        self.bert_relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(config.bert_layer_params.n_neurons + config.source_layer_params.n_neurons + config.subject_layer_params.n_neurons,
                                      config.output_layer_params.n_outputs)
        self.softmax = torch.nn.Softmax()

    def forward(self, input_ids, token_type_ids, attention_mask, sources, subjects):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
            'last_hidden_state']
        bert_output = bert_output[:, 0, :]
        bert_linear_output = self.bert_linear(bert_output)
        bert_linear_output = self.bert_relu(bert_linear_output)

        sources_output = self.sources_linear(sources)
        sources_output = self.sources_relu(sources_output)

        subjects_output = self.subjects_linear(subjects)
        subjects_output = self.subjects_relu(subjects_output)

        combined_output = torch.cat((bert_linear_output, sources_output, subjects_output), dim=1)
        final_output = self.output(combined_output)
        final_output = self.softmax(final_output)
        return final_output

def get_model(config):
    if config.metadata_in_input:
        return BERT_with_metadata(config)
    else:
        return BERT_baseline(config)