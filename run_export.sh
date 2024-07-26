export HF_ENDPOINT=https://hf-mirror.com

python3 script/inference.py \
    --weights_path "/home/openvino-ci-74/river/SAiD/SAiD/model/SAiD.pth" \
    --audio_path "/home/openvino-ci-74/river/SAiD/audio1.wav" \
    --output_path "/home/openvino-ci-74/river/SAiD/output.csv" \
    --device cpu \
    --convert_model True \
    --dynamic_shape True \
    --ov_model_path "/home/openvino-ci-74/river/SAiD/dynamic_models_1" \
    --num_steps 1

   # --device gpu.1 \
   # --use_ov True \
   # --convert_model True \
   # --dynamic_shape True \

