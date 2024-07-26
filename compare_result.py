import sys
import torch

ref_name=sys.argv[1]
cur_name=sys.argv[2]

ref_tensor = torch.load(ref_name)
cur_tensor = torch.load(cur_name)

print(ref_tensor.shape, ref_tensor.numel())
print(torch.allclose(ref_tensor, cur_tensor,atol=1e-2))
torch.set_printoptions(sci_mode=False)

equality = torch.eq(ref_tensor, cur_tensor)
# Find indices where tensors are not equal
not_equal_indices = torch.where(equality == False)
# Extract and print the differing elements
differing_elements_1 = ref_tensor[not_equal_indices]
differing_elements_2 = cur_tensor[not_equal_indices]
print(f"Differing elements in tensor1: {differing_elements_1}")
print(f"Differing elements in tensor2: {differing_elements_2}")
print(f"Differing elements in tensor2: {differing_elements_2 - differing_elements_1}")

print("diff size: ",differing_elements_1.shape)
