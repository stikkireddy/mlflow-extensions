# Audio Models

| Config                                              | Huggingface Link                              | context_length | min_azure_ep_type_gpu          | min_aws_ep_type_gpu           |
|:----------------------------------------------------|:----------------------------------------------|:---------------|:-------------------------------|:------------------------------|
| prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_64K_CONFIG  | https://huggingface.co/fixie-ai/ultravox-v0_4 | 64000          | GPU_LARGE [A100_80Gx1 80GB]    | MULTIGPU_MEDIUM [A10Gx4 96GB] |
| prebuilt.audio.vllm.FIXIE_ULTRA_VOX_0_4_128K_CONFIG | https://huggingface.co/fixie-ai/ultravox-v0_4 | Default        | GPU_LARGE_2 [A100_80Gx2 160GB] | GPU_MEDIUM_8 [A10Gx8 192GB]   |
