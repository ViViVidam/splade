from datasets import load_dataset
from safetensors.torch import load, save_file
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("/projects/bfcj/yzound/checkpoint/splade_v3_qat/")
'''
file_path = "/projects/bfcj/yzound/checkpoint/splade_v3_qat/model.safetensors"
with open(file_path, "rb") as f:
    data = f.read()
data = load(data)
print(data.keys())
new_tensors = {k[6:]:v for k,v in data.items()}
#print(new_tensors.keys())
#byte_data = save(tensors)
save_file(new_tensors, "/projects/bfcj/yzound/checkpoint/splade_v3_qat/model.safetensors")
'''