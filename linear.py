import torch.nn as nn


class Linear_(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act="ReLU", is_folded=True):
        super(Linear_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.act_type = act
        self.is_folded = is_folded
        self.linear = nn.Linear(in_features=self.in_features,
                                out_features=self.out_features,
                                bias=self.bias)
        self.act = _act(self.act_type)

    def forward(self, inputs):
        result_linear = self.linear(inputs)
        result = self.act(result_linear)
        return result

    @property
    def multiply_adds(self):
        result = self.in_features * self.out_features
        return result

    @property
    def params(self):
        params = self.in_features * self.out_features
        # TODO 不考虑fold方式，需要计算bias的参数量
        if self.bias is True and self.is_folded is False:
            params += self.out_features
            # print("%d" % self.out_features)
        return params


class Identity_(nn.Module):
    """
    skip connect
    """

    def __init__(self):
        super(Identity_, self).__init__()

    def forward(self, inputs):
        return inputs

def _act(act_type, **kwargs):
    if act_type is None or act_type == "Identity":
        return Identity_()
    elif act_type == "ReLU":
        result = nn.ReLU(inplace=True)
        return result
    else:
        raise Exception("Not implemented !!!")
    

    raise Exception("_act: Go here ..., maybe don\'t define some act")
