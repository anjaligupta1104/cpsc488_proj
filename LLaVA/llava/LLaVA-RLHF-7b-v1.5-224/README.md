---
license: apache-2.0
inference: false
---
# LLaVA-RLHF Model Card

## Model details

**Model type:**
LLaVA-RLHF represents a novel aligned end-to-end trained large multimodal model that combines a CLIP vision encoder and Vicuna for general-purpose visual and language understanding, achieving impressive visual reasoning and perception capabilities mimicking spirits of the multimodal GPT-4.
Via Factually Augmented RLHF, LLaVA-RLHF is presented to be more helpful and less hallucinated than LLaVA or other open-sourced LMMs.

**Usage:**
**NOTE: The RLHFed model is trained with LoRA and the bfloat16 data type.**
Users have to apply the PEFT-LoRA on the LLaVA-SFT+ model.

```python
dtype = torch.bfloat16
model_path = "LLaVA-RLHF-7b-v1.5-224/sft_model"
lora_path = "LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model"
model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    device_map={"": "cuda:0"},
    torch_dtype=dtype,
)
model = PeftModel.from_pretrained(
    model,
    lora_path,
)
```

**Model date:**
LLaVA-RLHF was trained in Sept 2023.

**Paper or resources for more information:**
https://llava-rlhf.github.io/

**License:**
Apache License 2.0

**Where to send questions or comments about the model:**
https://github.com/llava-rlhf/LLaVA-RLHF/issues

## Intended use
**Primary intended uses:**
The primary use of LLaVA-RLHF is research on large multimodal chatbots.

**Primary intended users:**
The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

## Training dataset
595K filtered image-text pairs from CC3M.

150K GPT-generated multimodal instruction-following chat data.

83K VQA v2 instruction-following VQA data.

16K A-OKVQA instruction-following CoT-VQA data.

23K FLICKR instruction-following spotting captioning data.

10K LLaVA-based human preference data