import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything_main.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# original_tensor = torch.randn(4, 4, 64, 64)
# new_tensor = original_tensor[4:5, :, :, :]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask

    img1 = img[..., :3]
    img2 = ((img1 - np.min(img1)) / (np.max(img1) - np.min(img1))*255).astype(np.uint8)
    gray_img = cv2.cvtColor(img2, cv2.COLOR_RGBA2GRAY)
    return gray_img,img1
    # ax.imshow(img)



sam_checkpoint='/root/autodl-tmp/DreamDiffusion/code/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth'
model_type='vit_h'
# folder_path = '/home/lab505/mind/segment_anything_main/output1'
folder_path = '/root/autodl-tmp/DreamDiffusion/code/segment_anything_main/eeg_jpeg'
save_folder_path = '/root/autodl-tmp/DreamDiffusion/code/segment_anything_main/output2'
label_path = '/root/autodl-tmp/DreamDiffusion/code/segment_anything_main/imageNet_class.json'


device = "cuda:0"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

def test():
    for filename in os.listdir(folder_path):
       if filename.endswith(".JPEG") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image1 = os.path.join(save_folder_path, filename)
            if not os.path.exists(image1):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(512,512))
                predictor = SamPredictor(sam)
                predictor.set_image(image)
                # parts = filename.split('_')
                # label_num = parts[0]
                # print(label_num)
                # with open(label_path, 'r') as file:
                #     json_data = file.read()
                # 解析JSON内容
                # data = json.loads(json_data)
                # promet = ""
                # for index, value in data.items():
                #     if label_num == value[0]:
                #         promet = value[1]
                #         break


                # for i in range(0,39):
                #     x = data['i'][0]
                #     if label_num ==data[i][0]:
                #      promet = data['i'][1]

                # masks, _, _ = predictor.predict(promet)


                mask_generator = SamAutomaticMaskGenerator(sam)
                masks = mask_generator.generate(image)
                image_with_mask = np.concatenate((image, np.zeros_like(image[..., :1])), axis=-1)
                # print(len(masks))
                # print(masks[0].keys())
                # plt.figure(figsize=(20, 20))
                # plt.imshow(image)
                gray_img, img=show_anns(masks)
                image_with_mask[:, :, 3] = gray_img
                # plt.figure(figsize=(20, 20))
                # plt.imshow(img)
                # plt.axis('off')
                # plt.savefig(os.path.join(save_folder_path, filename))
                cv2.imwrite(os.path.join(save_folder_path, filename),  image_with_mask)
                # plt.show()
                # plt.close()
# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)







