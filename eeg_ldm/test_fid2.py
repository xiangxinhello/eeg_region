from pytorch_fid import fid_score

# 
real_images_folder = '/home/results/eval/18-12-2023-14-38-20/test0-0.png'
generated_images_folder = '/home/results/eval/18-12-2023-14-38-20/test0-1.png'

# 
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=50, dims=2048,
                    cuda=True)

print(f"FID: {fid_value}")
