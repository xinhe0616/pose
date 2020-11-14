from pose_resnet import get_pose_net
from torchsummary import summary
# from lstm_pm import  LSTM_PM
# from Vgg16 import  VGG16
# import torch
# # net = get_pose_net().cuda()
# # summary(net,(1, 260, 260))
#
# # net = LSTM_PM(T=4)
# # a = torch.randn(12, 260, 260)  # batch size = 2
# # c = torch.randn(1, 260, 260)
# # summary(net,(a,c))
#
# net = VGG16().cuda()
# summary(net, (3,224,224))



import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet_161()

# vgg11 = models.vgg11().cuda()
# summary(vgg11,(3,224,224))

# vgg16 = models.vgg16().cuda()
# summary(vgg16,(3,224,224))

# vgg19 = models.vgg19().cuda()
# summary(vgg19,(3,224,224))

# alexnet = models.alexnet().cuda()
# summary(alexnet,(3,224,224))

# resnet18 = models.resnet18().cuda()
# summary(resnet18,(3,224,224))

# resnet34= models.resnet34().cuda()
# summary(resnet34,(3,224,224))

# from torch import  nn
# lstm = nn.LSTM(4,10).cuda()
# print(lstm.)
# nn.

# net = get_pose_net().cuda()
# summary(net,(1, 368, 368))
# print(net)
import torch
net = get_pose_net().cuda()
# a = torch.randn(1, 3, 224, 224).cuda()  # batch size = 2
a = torch.randn(1,1, 224, 224).cuda()  # batch size = 2
net(a)