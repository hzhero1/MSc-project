import torch
import numpy as np

# t = torch.tensor([[1, 2, 3], [2, 3, 4], [4, 4, 5]])
# t = t.float()
# # print(t.diagonal().sum())
# # print(t.sum())
# print(t)
# print(t.diagonal())
# # print(t.diagonal() / t.sum(0))
# # print(t.diagonal() / t.sum(1))
# print(t.sum(0))
# print(t.sum(1))
# # print(t.sum(0))
# # print(t.sum(1))
# print(t.sum(0) + t.sum(1))
# print(t.sum(0) * t.sum(1))
# print((torch.div(t.sum(0) + t.sum(1), (t.sum(0) * t.sum(1)))))
# # print('\n-----------------------\nEvaluation on test data\n-----------------------')
# # print('{:20}{}'.format('1', 2))
# # print()

y_ = (torch.rand(5, 1) * 2).type(torch.LongTensor).squeeze()
print(y_)
