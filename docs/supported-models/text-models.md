# Text Models

| Config                                                 | Huggingface Link                                             | context_length | min_azure_ep_type_gpu          | min_aws_ep_type_gpu            |
|:-------------------------------------------------------|:-------------------------------------------------------------|:---------------|:-------------------------------|:-------------------------------|
| prebuilt.text.sglang.GEMMA_2_9B_IT                     | https://huggingface.co/google/gemma-2-9b-it                  | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.sglang.META_LLAMA_3_1_8B_INSTRUCT_CONFIG | https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.NUEXTRACT                           | https://huggingface.co/numind/NuExtract                      | Default        | GPU_LARGE [A100_80Gx1 80GB]    | GPU_MEDIUM [A10Gx1 24GB]       |
| prebuilt.text.vllm.NUEXTRACT_TINY                      | https://huggingface.co/numind/NuExtract-tiny                 | Default        | GPU_SMALL [T4x1 16GB]          | GPU_SMALL [T4x1 16GB]          |
| prebuilt.text.vllm.NOUS_HERMES_3_LLAMA_3_1_8B_64K      | https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B    | 64000          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.NOUS_HERMES_3_LLAMA_3_1_8B_128K     | https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B    | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]    |
| prebuilt.text.vllm.COHERE_FOR_AYA_23_35B               | https://huggingface.co/CohereForAI/aya-23-35B                | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.QWEN2_5_7B_INSTRUCT                 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct              | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.QWEN2_5_14B_INSTRUCT                | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct             | Default        | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.QWEN2_5_32B_INSTRUCT                | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct             | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | MULTIGPU_MEDIUM [A10Gx4 96GB]  |
| prebuilt.text.vllm.QWEN2_5_72B_8K_INSTRUCT             | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct             | 8192           | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]    |
| prebuilt.text.vllm.QWEN2_5_72B_INSTRUCT                | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct             | Default        | GPU_LARGE_8 [A100_80Gx8 640GB] | GPU_LARGE_8 [A100_80Gx8 640GB] |
