# Approach decision

## Requirement

Streaming input = **video chunks + protocol text**; output = **CONTINUE** or **HALT + reason**, then stop.

## Why a VLM is required

| Need | CLIP + heads only | VLM (LLaVA-NeXT-Video) |
|------|-------------------|-------------------------|
| Read protocol instructions | Weak / none | Yes |
| Ground chunk frames in protocol | Limited | Yes |
| Free-form error reason | Templates only | Native LM generation |
| Calibrated early stop | Easy | Aux `halt_head` on VLM features |

**CLIP-only was insufficient.** Correct design:

1. **VLM backbone** (`LLaVA-NeXT-Video-7B-DPO`) for multimodal understanding + reason text (`L_lm`)
2. **Aux halt / step heads** on VLM hidden states for streaming early-stop (`L_halt`, `L_step`)
3. **Chunk loop** at inference: feed successive prefixes; stop when HALT

Not “two-stage CLIP then LLM”, but **one VLM with aux heads** for streaming control.
