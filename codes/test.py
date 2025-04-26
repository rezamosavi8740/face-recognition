import torch

def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint.get('state_dict', checkpoint)

def compare_models(path1, path2):
    sd1 = load_state_dict(path1)
    sd2 = load_state_dict(path2)

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    if keys1 != keys2:
        missing1 = keys1 - keys2
        missing2 = keys2 - keys1
        raise RuntimeError(f"Key mismatch:\n only in first: {missing1}\n only in second: {missing2}")

    total = 0
    diff = 0
    for k in keys1:
        w1 = sd1[k].cpu().flatten()
        w2 = sd2[k].cpu().flatten()
        if w1.numel() != w2.numel():
            raise RuntimeError(f"Shape mismatch for {k}: {w1.size()} vs {w2.size()}")
        total += w1.numel()
        diff += (w1 != w2).sum().item()

    pct = diff / total * 100
    if diff == 0:
        print("Models are identical.")
    else:
        print(f"Total params: {total}")
        print(f"Differing elements: {diff}")
        print(f"Difference: {pct:.4f}%")

if __name__ == "__main__":
    path_a = "/home/user1/face-recognition/codes/CVLface/cvlface/pretrained_models/adaface_ir101_webface12m/model.pt"
    path_b = "/home/user1/face-recognition/codes/CVLface/cvlface/research/recognition/experiments/run_v1/default_04-20_14/checkpoints/best/model.pt"
    compare_models(path_a, path_b)
