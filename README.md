# SAiD: Blendshape-based Audio-Driven Speech Animation with Diffusion

This is the code for `SAiD: Blendshape-based Audio-Driven Speech Animation with Diffusion`.

- [Project page](https://yunik1004.github.io/SAiD/)
- [Paper](https://arxiv.org/abs/2401.08655)

## Installation

Run the following command to install it as a pip module:

```bash
pip install .
```

If you are developing this repo or want to run the scripts, run instead:

```bash
pip install -e .[dev]
```

If there is an error related to pyrender, install additional packages as follows:

```bash
apt-get install libboost-dev libglfw3-dev libgles2-mesa-dev freeglut3-dev libosmesa6-dev libgl1-mesa-glx
```

## Directories

- `data`: It contains data used for preprocessing and training.
- `model`: It contains the weights of VAE, which is used for the evaluation.
- `blender-addon`: It contains the blender addon that can visualize the blendshape coefficients.
- `script`: It contains Python scripts for preprocessing, training, inference, and evaluation.
- `static`: It contains the resources for the project page.

## Inference

You can download the pretrained weights of SAiD from [Hugging Face Repo](https://huggingface.co/yunik1004/SAiD).

```bash
python script/inference.py \
        --weights_path "<SAiD_weights>.pth" \
        --audio_path "<input_audio>.wav" \
        --output_path "<output_coeffs>.csv" \
        [--init_sample_path "<input_init_sample>.csv"] \  # Required for editing
        [--mask_path "<input_mask>.csv"]  # Required for editing
```

## Enable OpenVINO pipeline

Install OpenVINO package

```bash
   python3 -m pip install openvino-dev[tensorflow2,onnx]
```

Convert torch file to be IR file

```bash
python3 script/inference.py \
    --weights_path "<SAiD>.pth" \
    --audio_path "<input_audio>.wav" \
    --output_path "<output_coeffs>.csv" \
    --device cpu \
    --dynamic_shape True \
    --convert_model True \
    --ov_model_path "ov_model_path" \
    --num_steps 1
```

Run inference with OpenVINO

```bash
python3 script/inference.py \
    --weights_path "<SAiD>.pth" \
    --audio_path "<input_audio>.wav" \
    --output_path "<output_coeffs>.csv" \
    --device gpu.1 \
    --use_ov True \
    --ov_model_path "ov_model_path" \
    --num_steps 200
```

## BlendVOCA

### Construct Blendshape Facial Model

Due to the license issue of VOCASET, we cannot distribute BlendVOCA directly.
Instead, you can preprocess `data/blendshape_residuals.pickle` after constructing `BlendVOCA` directory as follows for the simple execution of the script.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   └─ templates
      ├─ ...
      └─ FaceTalk_170915_00223_TA.ply
```

- `templates`: Download the template meshes from [VOCASET](https://voca.is.tue.mpg.de/download.php).

```bash
python script/preprocess_blendvoca.py \
        --blendshapes_out_dir "<output_blendshapes_dir>"
```

If you want to generate blendshapes by yourself, do the folowing instructions.

1. Unzip `data/ARKit_reference_blendshapes.zip`.
2. Download the template meshes from [VOCASET](https://voca.is.tue.mpg.de/download.php).
3. Crop template meshes using `data/FLAME_head_idx.txt`.
You can crop more indices and then restore them after finishing the construction process.
4. Use [Deformation-Transfer-for-Triangle-Meshes](https://github.com/guyafeng/Deformation-Transfer-for-Triangle-Meshes) to construct the blendshape meshes.
   - Use `data/ARKit_landmarks.txt` and `data/FLAME_head_landmarks.txt` as marker vertices.
   - Find the correspondance map between neutral meshes, and use it to transfer the deformation of arbitrary meshes.
5. Create `blendshape_residuals.pickle`, which contains the blendshape residuals in the following Python dictionary format.
Refer to `data/blendshape_residuals.pickle`.

    ```text
    {
        'FaceTalk_170731_00024_TA': {
            'jawForward': <np.ndarray object with shape (V, 3)>,
            ...
        },
        ...
    }
    ```

### Generate Blendshape Coefficients

You can simply unzip `data/blendshape_coeffcients.zip`.

If you want to generate coefficients by yourself, we recommend constructing the `BlendVOCA` directory as follows for the simple execution of the script.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   ├─ blendshapes_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ noseSneerRight.obj
   ├─ templates_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA.obj
   └─ unposedcleaneddata
      ├─ ...
      └─ FaceTalk_170915_00223_TA
         ├─ ...
         └─ sentence40
```

- `blendshapes_head`: Place the constructed blendshape meshes (head).
- `templates_head`: Place the template meshes (head).
- `unposedcleaneddata`: Download the mesh sequences (unposed cleaned data) from [VOCASET](https://voca.is.tue.mpg.de/download.php).

And then, run the following command:

```bash
python script/optimize_blendshape_coeffs.py \
        --blendshapes_coeffs_out_dir "<output_coeffs_dir>"
```

After generating blendshape coefficients, create `coeffs_std.csv`, which contains the standard deviation of each coefficients. Refer to `data/coeffs_std.csv`.

```text
jawForward,...
<std_jawForward>,...
```

## Training / Evaluation on BlendVOCA

### Dataset Directory Setting

We recommend constructing the `BlendVOCA` directory as follows for the simple execution of scripts.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   ├─ audio
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ sentence40.wav
   ├─ blendshape_coeffs
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ sentence40.csv
   ├─ blendshapes_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ noseSneerRight.obj
   └─ templates_head
      ├─ ...
      └─ FaceTalk_170915_00223_TA.obj
```

- `audio`: Download the audio from [VOCASET](https://voca.is.tue.mpg.de/download.php).
- `blendshape_coeffs`: Place the constructed blendshape coefficients.
- `blendshapes_head`: Place the constructed blendshape meshes (head).
- `templates_head`: Place the template meshes (head).

### Training VAE, SAiD

- Train VAE

    ```bash
    python script/train_vae.py \
            --output_dir "<output_logs_dir>" \
            [--coeffs_std_path "<coeffs_std>.txt"]
    ```

- Train SAiD

    ```bash
    python script/train.py \
            --output_dir "<output_logs_dir>"
    ```

### Evaluation

1. Generate SAiD outputs on the test speech data

    ```bash
    python script/test_inference.py \
            --weights_path "<SAiD_weights>.pth" \
            --output_dir "<output_coeffs_dir>"
    ```

2. Remove `FaceTalk_170809_00138_TA/sentence32-xx.csv` files from the output directory.
Ground-truth data does not contain the motion data of `FaceTalk_170809_00138_TA/sentence32`.

3. Evaluate SAiD outputs: FD, WInD, and Multimodality.

    ```bash
    python script/test_evaluate.py \
            --coeffs_dir "<input_coeffs_dir>" \
            [--vae_weights_path "<VAE_weights>.pth"] \
            [--blendshape_residuals_path "<blendshape_residuals>.pickle"]
    ```

4. We have to generate the videos to compute the AV offset/confidence.
To avoid the memory leak issue of the pyrender module, we use the shell script.
After updating `COEFFS_DIR` and `OUTPUT_DIR`, run the script:

    ```bash
    # Fix 1: COEFFS_DIR="<input_coeffs_dir>"
    # Fix 2: OUTPUT_DIR="<output_video_dir>"
    python script/test_render.sh
    ```

5. Use [SyncNet](https://github.com/joonson/syncnet_python) to compute the AV offset/confidence.

## Reference

If you use this code as part of any research, please cite the following paper.

```text
@misc{park2023said,
      title={SAiD: Speech-driven Blendshape Facial Animation with Diffusion},
      author={Inkyu Park and Jaewoong Cho},
      year={2023},
      eprint={2401.08655},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
