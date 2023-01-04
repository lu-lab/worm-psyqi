# from __future__ import absolute_import
# from lovasz_losses import lovasz_hinge
import torch


# def Compute_combine(model, CR, input_tensor, label_tensor):
#     if input_tensor.ndim < 5:
#         input_tensor = input_tensor.unsqueeze(dim=1)
#     input_tensor = input_tensor.cuda()
#     label_tensor = label_tensor.clone().detach().long().cuda()
#     label_tensor_class_1 = label_tensor.squeeze(1).cuda()
#     label_tensor_class_0 = torch.ones_like(label_tensor_class_1).cuda() - label_tensor_class_1
#     result = model(input_tensor)
#     cr_loss = CR(result, label_tensor)
#     result_class_0 = result.narrow(dim=1, start=0, length=1).squeeze(dim=1)
#     result_class_1 = result.narrow(dim=1, start=1, length=1).squeeze(dim=1)
#     batch_loss_class_0 = cri(result_class_0, label_tensor_class_0)
#     batch_loss_class_1 = cri(result_class_1, label_tensor_class_1)
#     batch_loss = batch_loss_class_1 + batch_loss_class_0 + cr_loss
#     return batch_loss, result


def Compute_Lov(model, criteria, input_tensor, label_tensor):
    if input_tensor.ndim < 5:
        input_tensor = input_tensor.unsqueeze(dim=1)
    input_tensor = input_tensor.cuda()
    label_tensor = label_tensor.clone().detach().long()
    label_tensor_class_1 = label_tensor.squeeze(1).cuda()
    # label_tensor_class_0 = torch.ones_like(label_tensor_class_1).cuda() - label_tensor_class_1
    result = model(input_tensor)
    # result_class_0 = result.narrow(dim=1, start=0, length=1).squeeze(dim=1)
    result_class_1 = result.narrow(dim=1, start=1, length=1).squeeze(dim=1)
    # batch_loss_class_0 = cri(result_class_0, label_tensor_class_0)
    batch_loss_class_1 = criteria(result_class_1, label_tensor_class_1)
    batch_loss = batch_loss_class_1
    return batch_loss, result


def Compute(model, criteria, input_tensor, label_tensor, *args):
    if input_tensor.ndim < 5:
        input_tensor = input_tensor.unsqueeze(dim=1)
    input_tensor = input_tensor.cuda()
    label_tensor = label_tensor.squeeze(1).long().cuda()
    result = model(input_tensor)
    batch_loss = criteria(result, label_tensor)
    return batch_loss, result


def Compute_MultiScale(model, criteria, input_tensor, label_tensor):
    assert input_tensor.ndim == 5 and input_tensor.shape[1] == 2, 'Squeeze 2 channels together, follow red-green manner'
    red_tensor, green_tensor = [tensor.cuda() for tensor in torch.split(input_tensor, 1, dim=1)]
    red_label, green_label = [label.squeeze(dim=1).long().cuda() for label in torch.split(label_tensor, 1, dim=1)]
    red_result, green_result = model(red_tensor, green_tensor)
    # print(red_tensor.requires_grad)
    red_loss, green_loss = criteria(red_result, red_label), criteria(green_result, green_label)
    return red_result, green_result, red_loss, green_loss


if __name__ == '__main__':
    pass







