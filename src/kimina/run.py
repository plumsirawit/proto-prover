import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "AI-MO/Kimina-Prover-Distill-0.6B"
device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
model.eval()

problem = r"The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume?"
print(problem)
formal_statement = r"""import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat
/-- The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/
theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
"""
prompt = "Think about and solve the following problem step by step in Lean 4."
prompt += f"\n# Problem:{problem}"
prompt += f"\n# Formal statement:\n```lean4\n{formal_statement}\n```\n"
messages = [
    {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
    {"role": "user", "content": prompt},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=8096,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
    )

output_text = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
)

print(output_text)
