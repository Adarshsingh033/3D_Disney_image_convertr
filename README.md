#  Full AI Image Processing Pipeline

### (Validation → Face Detection → Disney 3D Avatar → Face Swapping)

This project integrates **computer vision**, **face analysis**, and **generative AI** into one complete workflow.
It validates uploaded images, detects faces, analyzes emotions, creates **Disney/Pixar-style 3D avatars**, and performs **realistic AI face swapping** using **InsightFace**.

Perfect for experimenting with **AI-based avatar creation**, **face re-styling**, or **digital character generation**.

---

##  Features

###  Step 1 — Image Validation

* Validates uploaded image (format, resolution, clarity)
* Checks for blurriness using **Laplacian variance**
* Ensures only high-quality images proceed to analysis

###  Step 2 — Face Detection & Analysis

* Detects faces using **DeepFace (RetinaFace)**
* Crops faces with adjustable padding
* Performs multi-backend analysis for:

  * **Age**
  * **Gender**
  * **Emotion**
* Saves cropped faces to `faces_output/`

###  Step 3 — Disney 3D Avatar Generation

* Generates **Pixar/Disney-style avatars** using **Stable Diffusion**
* Enhances color, contrast, and sharpness automatically
* Uses models such as:

  * `nitrosocke/mo-di-diffusion`
  * `prompthero/openjourney-v4`
  * `stablediffusionapi/disney-pixar-cartoon`
  * `runwayml/stable-diffusion-v1-5`
* Adjustable parameters for creative results:

  * `strength = 0.4`
  * `guidance_scale = 8.0`
  * `num_inference_steps = 45`

###  Step 4 — Face Swapping (Full Head Replacement)

* Swaps one or multiple faces using **InsightFace ONNX model**
* Supports batch face swaps (multiple source and target faces)
* Automatically aligns and pastes swapped faces back into the target image
* Outputs a high-quality composited image

---

##  Installation

Run the following commands in Google Colab or your environment:

```bash
!pip install deepface opencv-python pillow matplotlib torch torchvision diffusers transformers accelerate safetensors insightface onnxruntime
```

If running locally:

* Ensure you have **Python 3.10+**
* Install **CUDA-enabled PyTorch** if you have an NVIDIA GPU

---

##  Workflow Overview

### 1️⃣ Validate Image

```python
result = validate_input_image("your_image.png")
print(result)
```

### 2️⃣ Detect and Analyze Faces

```python
from deepface import DeepFace
# Automatically detects, crops, and analyzes all faces
```

### 3️⃣ Generate Disney 3D Avatar

```python
generator = Disney3DAvatarGenerator()
generator.generate_subtle_disney_avatar("faces_output/face_1.jpg")
```

### 4️⃣ Perform Face Swap

```python
result_image = face_swapper(
    source_image_path="faces_output/face_1.jpg",
    target_image_path="template_image.jpg",
    model_path="/content/inswapper_128 (1).onnx",
    output_path="result.png"
)
```

---

##  Example Output

| Step             | Description                            | Example                                   |
| ---------------- | -------------------------------------- | ----------------------------------------- |
|   Validation     | Image checked for clarity & size       | `"Image Successfully passed validation!"` |
|     Detection    | Cropped faces saved to `faces_output/` | ![face](faces_output/face_1.jpg)          |
|    Disney Avatar | Generated 3D Disney-style face         | ![disney](disney_subtle.png)              |
|    Face Swap     | Face replaced on target image          | ![result](result.png)                     |

---

## 🧠 Technical Notes

* **Model used for face swap:** `inswapper_128.onnx`
* **Face analysis model:** `buffalo_l`
* **Diffusers pipeline:** `StableDiffusionImg2ImgPipeline`
* **All steps compatible with Google Colab GPU runtime**

---

## Optional Parameters

### Disney Generator

| Parameter             | Description               | Default |
| --------------------- | ------------------------- | ------- |
| `strength`            | Degree of transformation  | `0.4`   |
| `guidance_scale`      | Prompt adherence strength | `8.0`   |
| `num_inference_steps` | Diffusion sampling steps  | `45`    |

### Face Swapper

| Parameter       | Description                         |
| --------------- | ----------------------------------- |
| `model_path`    | Path to ONNX model file             |
| `restore_faces` | (Optional) Enables restoration step |
| `output_path`   | Where final swapped image is saved  |

---

##  Tech Stack

* **Python 3.10+**
* **DeepFace**
* **InsightFace**
* **ONNX Runtime**
* **Diffusers (Stable Diffusion)**
* **OpenCV / Pillow / Matplotlib**
* **PyTorch with CUDA support**


---
