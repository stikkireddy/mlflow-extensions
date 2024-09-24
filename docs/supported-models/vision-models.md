# Vision Models

| Config                                                   | Huggingface Link                                             | context_length | min_azure_ep_type_gpu          | min_aws_ep_type_gpu           |
|:---------------------------------------------------------|:-------------------------------------------------------------|:---------------|:-------------------------------|:------------------------------|
| prebuilt.vision.sglang.LLAVA_NEXT_LLAMA3_8B              | https://huggingface.co/lmms-lab/llama3-llava-next-8b         | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.sglang.LLAVA_NEXT_QWEN_1_5_72B_CONFIG    | https://huggingface.co/lmms-lab/llama3-llava-next-8b         | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
| prebuilt.vision.sglang.LLAVA_ONEVISION_QWEN_2_7B_CONFIG  | https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov  | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.sglang.LLAVA_ONEVISION_QWEN_2_72B_CONFIG | https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_4K          | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | 4096           | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]         |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_8K          | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | 8192           | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]         |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_12K         | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | 12000          | GPU_LARGE [A100_80Gx1 80GB]    | GPU_MEDIUM [A10Gx1 24GB]      |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_32K         | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | 32000          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_64K         | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | 64000          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.vllm.PHI_3_5_VISION_INSTRUCT_128K        | https://huggingface.co/microsoft/Phi-3.5-vision-instruct     | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
| prebuilt.vision.vllm.QWEN2_VL_2B_INSTRUCT                | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct             | Default        | GPU_LARGE [A100_80Gx1 80GB]    | GPU_MEDIUM [A10Gx1 24GB]      |
| prebuilt.vision.vllm.QWEN2_VL_7B_INSTRUCT                | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct             | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.vllm.PIXTRAL_12B_32K_INSTRUCT            | https://huggingface.co/mistralai/Pixtral-12B-2409            | 32768          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.vllm.PIXTRAL_12B_64K_INSTRUCT            | https://huggingface.co/mistralai/Pixtral-12B-2409            | 65536          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.vision.vllm.PIXTRAL_12B_128K_INSTRUCT           | https://huggingface.co/mistralai/Pixtral-12B-2409            | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |