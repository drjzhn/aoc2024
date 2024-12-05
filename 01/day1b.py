import numpy as np
import torch
import time

if torch.backends.mps.is_available():
    print("using metal (yay)")
    torch.mps.empty_cache()
    device = torch.device("mps")
else: 
    print("using cpu (boo)")
    device = torch.device("cpu")

start = time.time()

tensors = torch.from_numpy(np.loadtxt('input.txt', dtype=np.float32)).to(device) 

left = tensors[:, 0]
right = tensors[:, 1]

similarity_score = sum(
    left_val.int() * torch.sum(right == left_val).int()
    for left_val in left
)

print(f"similarity score: {similarity_score}")

end = time.time()
print(f"total time: {(end - start)*1000}ms")

## 147ms (M1 MBA)