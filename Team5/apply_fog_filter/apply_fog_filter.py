import cv2
import numpy as np
import os

# ---------- 유틸: fBM 스타일 노이즈 ----------
def fbm_noise(h, w, octaves=4, persistence=0.5, seed=0):
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), np.float32)
    amp = 1.0
    total_amp = 0.0
    for i in range(octaves):
        fh, fw = max(1, h // (2**(octaves - 1 - i))), max(1, w // (2**(octaves - 1 - i)))
        base = rng.random((fh, fw)).astype(np.float32)
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        k = max(1, int(3 * (i + 1)))
        if k % 2 == 0: k += 1
        base = cv2.GaussianBlur(base, (k, k), 0)
        noise += base * amp
        total_amp += amp
        amp *= persistence
    noise = noise / (total_amp + 1e-6)
    return np.clip(noise, 0, 1)

# ---------- 유틸: 라이트 샤프트 ----------
def light_rays(mask, light_pos, ray_length=200, ray_decay=0.95, blur_ksize=25):
    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    vx, vy = (cx - light_pos[0], cy - light_pos[1])
    vlen = np.hypot(vx, vy) + 1e-6
    dx, dy = vx / vlen, vy / vlen

    acc = np.zeros_like(mask, dtype=np.float32)
    tmp = mask.copy().astype(np.float32)

    if blur_ksize % 2 == 0: blur_ksize += 1
    tmp = cv2.GaussianBlur(tmp, (blur_ksize, blur_ksize), 0)

    strength = 1.0
    for _ in range(int(ray_length)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        tmp = cv2.warpAffine(tmp, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc += tmp * strength
        strength *= ray_decay

    acc = acc - acc.min()
    if acc.max() > 1e-6:
        acc = acc / acc.max()
    acc = cv2.GaussianBlur(acc, (blur_ksize, blur_ksize), 0)
    return np.clip(acc, 0, 1)

# ---------- 볼류메트릭 안개 ----------
def apply_volumetric_fog(image_bgr,
                         strength=0.55,
                         fog_color=(245, 248, 250),
                         height_bias=0.1,
                         height_falloff=1.8,
                         noise_scale=0.5,
                         noise_octaves=4,
                         desaturate=0.65,
                         contrast_fade=0.45,
                         light_pos=None,
                         ray_strength=0.7,
                         ray_length=180,
                         ray_decay=0.965,
                         seed=7):
    img = image_bgr.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    depth_grad = np.clip((yy - height_bias) * height_falloff, 0, 1)

    noise = fbm_noise(h, w, octaves=noise_octaves, persistence=0.5, seed=seed)
    noise = (noise - 0.5) * 1.4 + 0.5
    noise = np.clip(noise, 0, 1)

    fog_density = (1 - noise_scale) * depth_grad + (noise_scale) * (depth_grad * noise)
    fog_density = cv2.GaussianBlur(fog_density, (0, 0), sigmaX=3, sigmaY=3)
    fog_density = np.clip(fog_density, 0, 1)

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)
    S = S * (1.0 - desaturate * fog_density)
    S = np.clip(S, 0, 255)
    hsv_mod = cv2.merge([H, S, V]).astype(np.uint8)
    img_desat = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    fog_density3 = fog_density[..., None]               # (H, W) -> (H, W, 1)
    img_flat = (img_desat - 0.5) * (1.0 - (contrast_fade * fog_density3)) + 0.5
    # img_flat = (img_desat - 0.5) * (1.0 - (contrast_fade * fog_density)) + 0.5
    img_flat = np.clip(img_flat, 0, 1)

    if light_pos is None:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, _, _, maxLoc = cv2.minMaxLoc(gray)
        light_pos = maxLoc

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_mask = np.clip((gray - 0.7) / 0.3, 0, 1)
    rays = light_rays(bright_mask, light_pos, ray_length=ray_length, ray_decay=ray_decay, blur_ksize=31)
    ray_map = np.clip(rays * fog_density, 0, 1)
    img_ray = np.clip(img_flat + (ray_map * ray_strength)[..., None], 0, 1)

    fog_col = np.array(fog_color, np.float32) / 255.0
    alpha = np.clip(fog_density * strength, 0, 1)[..., None]
    out = img_ray * (1 - alpha) + fog_col * alpha
    out = np.clip(out, 0, 1)

    return (out * 255).astype(np.uint8)

# ---------- 메인 루프 ----------
if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for fname in os.listdir(input_dir):
        if not any(fname.lower().endswith(ext) for ext in valid_exts):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"❌ Failed to load {in_path}")
            continue

        fogged = apply_volumetric_fog(img)
        cv2.imwrite(out_path, fogged)
        print(f"✅ Processed: {in_path} -> {out_path}")
