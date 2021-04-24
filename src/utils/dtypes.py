import torch

torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)
float_type = torch.float64

int_type = torch.int64
eps = 1e-7