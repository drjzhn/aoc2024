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

sorted, _ = torch.sort(tensors, dim=0) 

print(f"total distance: {torch.sum(torch.abs(sorted[:, 0] - sorted[:, 1]))}")

end = time.time()
print(f"total time: {(end - start)*1000}ms")

## 27ms (M1 MBA)