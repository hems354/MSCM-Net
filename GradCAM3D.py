import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.CNN_Mamba.LowTransformer import LowTransformer
from scipy.ndimage import zoom

def load_3d(img_path):
    if img_path.endswith('.npy'):
        img_3d = np.load(img_path)
        # 检查原始维度（确保是3维）
        assert len(img_3d.shape) == 3, f"输入必须是3维数组(d,h,w)，当前维度：{img_3d.shape}"
        original_d, original_h, original_w = img_3d.shape
        print(f"原始3D数组尺寸: (d={original_d}, h={original_h}, w={original_w})")

        # 计算各维度的缩放因子（目标尺寸256/原始尺寸）
        scale_d = 256 / original_d
        scale_h = 256 / original_h
        scale_w = 256 / original_w
        scaling_factors = (scale_d, scale_h, scale_w)

        # 三维立方插值（order=3对应立方插值，order=1是线性插值）
        img_3d_resized = zoom(
            img_3d,
            zoom=scaling_factors,
            order=3,  # 3=立方插值，最接近你要的立方插值效果
            mode='nearest'  # 边界填充方式，避免边缘值异常
        )
        img=img_3d_resized
    elif img_path.endswith(('.nii', '.nii.gz')):
        try:
            import SimpleITK as sitk
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        except ImportError:
            raise ImportError("请先安装SimpleITK: pip install SimpleITK")
    else:
        raise ValueError("仅支持.npy或.nii/.nii.gz格式的3D MRI数据")

    assert img.ndim == 3
    return img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    stride      = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    padding     = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    size = [32, 16, 8, 4]
    embed_dims = [16, 32, 64, 128, 256, 320, 512]
    model = LowTransformer(kernel_size=kernel_size, stride=stride, padding=padding, embed_dims=embed_dims, size=size).to(device)
    print(model)
    weights = torch.load('/home/wangyh/Codes/pCR/results/checkpoints/PCM_checkpoint_1180.pt',map_location='cuda')['net']
    model.load_state_dict(weights, strict=False)
    model.eval()

    target_layers = [model.layers[3].blocks[-1].conv_block.conv2.conv]

    img_path = "/home/wangyh/Codes/pCR/data/Z12B.npy" 
    assert os.path.exists(img_path), f"文件不存在: {img_path}"
    
    img_3d = load_3d(img_path)
 
    img_3d = img_3d.astype(np.float32)
    img_3d = (img_3d - img_3d.min()) / (img_3d.max() - img_3d.min() + 1e-8)
    
    input_tensor = torch.from_numpy(img_3d).unsqueeze(0).unsqueeze(0)

    
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = None  
    grayscale_cam_3d = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam_3d = grayscale_cam_3d[0, :]  

    slice_idx = img_3d.shape[0] // 2 
    img_slice = img_3d[slice_idx, :, :]  
    cam_slice = grayscale_cam_3d[slice_idx, :, :]  
    

    img_slice_rgb = np.stack([img_slice]*3, axis=-1)
    
   
    visualization = show_cam_on_image(
        img_slice_rgb, 
        cam_slice, 
        use_rgb=True,
        image_weight=0.5  
    )
    
    plt.figure(figsize=(10, 10))

    plt.imshow(visualization)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig("3d_mri_gradcam.png", dpi=300)


if __name__ == '__main__':
    main()
