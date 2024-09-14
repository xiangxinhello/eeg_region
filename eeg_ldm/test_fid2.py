from pytorch_fid import fid_score

# 定义真实图像和生成图像的文件夹路径
real_images_folder = '/root/autodl-tmp/DreamDiffusion/dreamdiffusion/results/eval/18-12-2023-14-38-20/test0-0.png'
generated_images_folder = '/root/autodl-tmp/DreamDiffusion/dreamdiffusion/results/eval/18-12-2023-14-38-20/test0-1.png'

# 计算 FID
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=50, dims=2048,
                    cuda=True)

print(f"FID: {fid_value}")
