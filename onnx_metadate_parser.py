"""
pip install onnx onnxruntime

python3 -m venv .venv
source .venv/bin/activate
pip install onnx onnxruntime

"""


import onnx

model = onnx.load("yolo.onnx")

# 표준 필드
print("producer_name :", model.producer_name)
print("doc_string    :", model.doc_string)

# 커스텀 key-value (여기에 author/license/description/docs 들어있음)
for p in model.metadata_props:
    print(f"{p.key:12s}: {p.value}")
