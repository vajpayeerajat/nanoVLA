I'd try these, roughly cheapest and most informative first. Assume DINOv3 B/16 at 224, which gives the same 14×14 grid as our SigLIP, and DINOv3 frozen as in openpi. Everything else stays fixed: same steps, seed and batch size.

### Reference runs
- **R0 – SigLIP only.** The current baseline.
- **R1 – DINOv3 only**, in SigLIP's place. Shows what DINOv3 is worth on its own. If R1 ≈ R0, any gain from fusion comes from combining the two, not from DINOv3 simply being better.

### Early fusion: before the language model, same token count
- **F1 – Add, like openpi.** `siglip_tok + proj(norm(dino_tok))`, with the projection starting at zero so training begins exactly at the baseline. Matches your openpi run, so the results are directly comparable.
- **F2 – Concatenate per patch, like Prismatic/OpenVLA.** `MLP([siglip ; dino])` for each patch. The usual robot-policy recipe. More capacity than adding, because the model learns how to mix the two instead of being forced to sum them.
- **F3 – Learned gate per patch.** `siglip + g(siglip, dino) · proj(dino)`. Lets the model decide where geometry helps, for example near the gripper or object versus the background. It's cheap, and the gate values can be plotted.

### Sequence fusion: more tokens
- **F4 – Append DINOv3 tokens to the sequence.** Separate projectors, with the DINOv3 tokens after the SigLIP ones, which doubles image tokens to about 294 for 3 cameras. The language model's attention does the mixing. The best case for quality, at the highest cost. It tells you whether keeping the two separate beats merging them early.

### Hybrid: send each encoder where it's most useful
- **F5 – DINOv3 straight to the action expert.** SigLIP goes through the language model as now. DINOv3 tokens are added to the context the expert cross-attends to, skipping the language model. This tests "the language model needs meaning, the action head needs geometry". It's cheap because the expert already cross-attends, and I think it's the most promising idea specific to this project.
- **F6 – F1 or F2 plus F5.** Fuse into the language model and also give the expert direct DINOv3 access. Run it only if both parents help.

### Cutting tokens (only if F4 or F5 win but are too slow)
- **F7 – C-Abstractor (convolution) over the combined tokens**, down to 49 tokens per camera. Keeps spatial layout. Prefer it over a Perceiver or Q-Former, which tend to lose position information that matters for grasping.

### Later, if language grounding is still weak
- **F8 – Q-Former that also reads the instruction.** Aimed directly at the "ignores the instruction when both octopi are visible" weakness. More code; only worth it once F1–F5 have settled the best encoder setup.

### One switch to cross with the winner
- **DINOv3 frozen vs. last few blocks unfrozen**, or two-stage training: first only the projectors and expert, then unfreeze. Run it on the best one or two strategies only.

### How to pick a winner
For each run, compare:
- test L1 error per joint group, especially right arm and right hand, from `eval.py`
- steps per second and image-token count
- the Grad-CAM report:
  - the loss change for "no cameras" and "other scene" (does it use vision?)
  - the instruction swap with both octopi visible (does it use language?)
  - how much of the ego-view heat map falls on the target object (is it looking at the right thing?)

**Suggested first batch:** R0, R1, F1, F2, F4, F5. That's six runs, and it answers the main questions: add vs. concatenate vs. separate tokens, and whether the language model or the action expert is the right place for DINOv3. F3 and F6–F8 depend on what wins.



