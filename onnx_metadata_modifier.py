"""
pip install onnx onnxruntime

python3 -m venv .venv
source .venv/bin/activate
pip install onnx onnxruntime

"""


import onnx

# 가능하면 손상 위험 없는 백업본에서 작업하세요
model = onnx.load("yolo.onnx")

def set_meta(m, key, value):
    """metadata_props에서 key가 있으면 갱신, 없으면 추가."""
    for p in m.metadata_props:
        if p.key == key:
            p.value = value
            return
    entry = m.metadata_props.add()
    entry.key = key
    entry.value = value

set_meta(model, "description", "새 설명")
set_meta(model, "author",      "새 작성자")
set_meta(model, "license",     "새 라이선스")
set_meta(model, "docs",        "https://...")

# 저장 전 구조 검증 — 여기서 통과하면 바이너리 정합성 OK
onnx.checker.check_model(model)
onnx.save(model, "yolo_edited.onnx")
print("saved.")
