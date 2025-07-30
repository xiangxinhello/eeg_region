import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

# 
real_images_folder = '/home/results/eval/18-12-2023-14-38-20/test0-0.png'
generated_images_folder = '/home/results/eval/18-12-2023-14-38-20/test0-1.png'

# 
inception_model = torchvision.models.inception_v3(pretrained=True)

# 
transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# # 
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 inception_model,cuda = True, dims=2048)
print('FID value:', fid_value)
