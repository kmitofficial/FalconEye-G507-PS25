# main.py
import torch
import cv2
import matplotlib.pyplot as plt
from local_sam import build_sam_vit_h

# 1. Build SAM model
sam = build_sam_vit_h()
checkpoint = torch.load("weights/sam_vit_h_4b8939.pth", map_location="cpu")
sam.load_state_dict(checkpoint, strict=True)
sam.eval()

print("✅ SAM ready to use")

# 2. Load an image
image = cv2.imread("test.jpg")  # put any sample image here
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

# 3. Create a prompt (single click at center of image)
H, W, _ = image.shape
point = torch.tensor([[[W // 2, H // 2]]])  # (B, N, 2)
label = torch.tensor([[1]])  # 1 = foreground point
points = (point, label)

# 4. Run SAM forward
with torch.no_grad():
    masks, iou_preds = sam(image_tensor, points=points)

print("Mask shape:", masks.shape)

# 5. Visualize first predicted mask
mask = masks[0][0].cpu().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.imshow(mask, alpha=0.5)
plt.axis("off")
plt.show()
