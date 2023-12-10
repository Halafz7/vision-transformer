from models.vit import vit
from brevitas.export import export_onnx_qcdq
import torch
import torch.nn as nn

test = False
export = True

def get_model(model_name):
    if model_name == "ViT_B_16":
        return vit.get_vit_b_16()
    if model_name == "ViT_L_16":
        return vit.get_vit_l_16()
    if model_name == "ViT_H_14":
        return vit.get_vit_h_14()
    
model_name = "ViT_B_16"
model = get_model(model_name)

if export:
    export_onnx_qcdq(model, input_shape=([1,3,224,224]), export_path=(model_name + "_quant.onnx"), opset_version=13)

if test:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(25):
        print(str(epoch + 1) + "/" + str(25))
        batch_size = 1 
        random_input = torch.randn(batch_size, 3, 224, 224)  # Assuming 3-channel RGB images

        labels = torch.randint(0, 1000, (batch_size,))

        optimizer.zero_grad()

        outputs = model(random_input)
        """
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        """

        loss = criterion(outputs[2], labels)

        loss.backward()
        optimizer.step()

    model.eval()
    for test in range(25):
        print(str(test + 1) + "/" + str(25))
        random_input = torch.randn(batch_size, 3, 224, 224)  # Assuming 3-channel RGB images
        outputs = model(random_input)
