import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel


class AutoModelForSlabt(nn.Module):
    def __init__(self, pretrained_path, config, args=None, cls_num=3):
        super(AutoModelForSlabt, self).__init__()

        self.cls_num = cls_num
        self.bert_config = config
        self.bert = BertModel.from_pretrained(pretrained_path)
        print(self.bert.config)
        # self.fc1 = nn.Linear(self.bert_config.hidden_size, cls_num)
        self.fc1 = nn.Linear(self.bert_config.hidden_size, 1)
        self.fc2 = nn.Linear(self.bert_config.hidden_size, 1)
        self.fc3 = nn.Linear(self.bert_config.hidden_size, 1)

        # for tables
        # for different levels
        self.register_buffer('pre_features', torch.zeros(args.n_feature, args.feature_dim))
        self.register_buffer('pre_weight1', torch.ones(args.n_feature, 1))


    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[1]
        flatten_features = sequence_output # (batch_size, hidden_size)


        if self.cls_num == 3:
            xc = self.fc1(flatten_features)
            xe = self.fc2(flatten_features)
            xn = self.fc3(flatten_features)
            x_enable = torch.cat((xc, xe, xn), dim=1)
        elif self.cls_num == 2:
            xc = self.fc1(flatten_features)
            xe = self.fc2(flatten_features)
            x_enable = torch.cat((xc, xe), dim=1)
        else:
            raise KeyError       


        # detach_features = flatten_features.detach()
        
        # x_unabled = self.fc1(detach_features)
        # return x_enable, flatten_features, x_unabled

        return x_enable, flatten_features
