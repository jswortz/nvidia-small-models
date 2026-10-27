# Interview Story: Reliable Agents from a Small Model

A talk track for walking through this project, with the numbers to back each point. Every number here comes from the eval runs recorded in the README and CLAUDE.md.

## The one-sentence version

I built a retail support agent on Nemotron Nano 9B and showed that a small model is reliable when it handles understanding and wording while deterministic code makes the policy decisions and takes the actions. On the same 62 cases, letting the model decide and act gave 11 passes and 4 harmful actions; letting code decide gave 61 passes and none.

## The story in five beats

### 1. I started with the obvious design, and it failed in a specific way

One Nano agent with every tool and a careful prompt plateaued at 6 of 10 on my first test set. Each new prompt rule fixed one case and broke another. A harness that nudged the model to act turned missed refunds into wrong refunds. The model also ignored facts it was handed: the order lookup said "outside the 30 day window" and the model approved the refund anyway.

The lesson: the model was good at intent and wording and bad at policy, and more prompting did not change that.

### 2. I split the work by what each part is good at

The model routes the message into five fields and writes the reply. A policy engine of about 100 lines of plain rules makes every decision and names the clause it applies. Code runs the action. The writer has no tools, so it cannot promise a refund that did not happen.

Same model, same 62 cases (scored with the current eval):

| | Single agent | Routed agent |
| --- | --- | --- |
| Task success | 11/62 | 59/62 |
| Harmful actions | 4 | 0 |
| False promises | 7 | 0 |
| Time per case | 6.9 s | 3.1 s |

No case passes with the single agent and fails with the routed one.

### 3. I profiled before optimizing

With NeMo Agent Toolkit I found the policy engine takes 0.03 ms and model calls are 99.8% of each case. So the speedup was in serving, not the agent: removing one vLLM flag (eager mode) let CUDA graphs run and cut time per case 36%, from 4.91 s to 3.13 s, with identical outputs.

### 4. I fine-tuned the router, and the held-out set caught what the main set hid

The router is the one model call every case depends on. I generated 742 training messages with hosted Nemotron Super, with labels fixed before each message was written, and fine-tuned Nano with LoRA in NeMo AutoModel. Four minutes on a rented A100.

Before training, I ran a cheap control: a one-line prompt beat my carefully written prompt by 8 to 20 points. That reset the fair baseline from 78% to 86% on unseen message styles. Fine-tuning then took it to 100%.

Then the trained router passed all 62 cases and failed the held-out set. "Could you add a second toaster to order 1090?" was read as a replacement, and the engine approved swapping the toaster for the same toaster. It was the first harmful action the routed agent ever took. The fix was in two places: an engine rule (an exchange must be for a different product), which would have blocked it regardless of the router, and new training data for "add to my order" requests. After retraining: 11/11 held-out, 0 harmful.

### 5. I found a hole in my own eval and published the lower numbers

The same held-out run had the planner answer a gift-wrapping question by inventing "policy clause 2.1." The eval passed it, because it checked that the expected clause appeared and that the reply did not promise an action. It never checked that every cited clause was real and retrieved.

I added that check and rescored every earlier run from its saved traces. My own published 62/62 became 59/62, and the single agent had 47 invented citations in 62 cases. I kept the stricter numbers. The final setup (fine-tuned router, hosted Super planner) scores 61/62 and 11/11 with 0 harmful actions and 0 false promises.

## What I would say I learned

- **Put policy in code, not in the prompt.** Code was faster, cheaper, and never wrong on a date or a threshold. It also caught a mistake that only appeared after the model got better.
- **Measure harm separately from failure.** A wrongful refund costs money; a missed one costs goodwill. Those two numbers, not overall accuracy, separate the designs.
- **Run controls and held-out sets.** The control changed how much credit fine-tuning deserved. The held-out set found the only harmful action in the project.
- **Score what you claim, and accept the result.** A stricter check lowered my headline number, and it was the right call.
- **Check the plumbing.** A LoRA adapter loaded in vLLM with no error and changed nothing. Only comparing routes against a previous run caught it.

## Questions I expect, with short answers

**Why not just use a bigger model?**
The single agent's failures were policy decisions on dates, windows, and photos. A bigger model makes fewer of those mistakes, but it still makes them, and every one is a wrongful refund. Code makes none, costs 0.03 ms, and is auditable clause by clause. The big model is still useful where judgment is needed: the planner for the rare request nothing else can place.

**Did fine-tuning matter if the routed agent already passed?**
Yes, but less than it looks. Most of the gap closed with a better prompt. Fine-tuning closed the rest, made the router more robust to messy writing (86% to 100% on unseen styles), and sends about 350 fewer prompt tokens per call. The bigger win was what testing it exposed: an engine gap and an eval gap.

**Why did GRPO not run?**
NeMo RL keeps a training copy and a sampling copy of the 9B model in host memory. Loading alone needed about 82 GB. With 64 GB of swap added, the model loaded, but the first training step filled the instance's 94 GB of RAM, spilled into swap, and the machine stopped responding. It needs a node with roughly 200 GB of RAM. The SFT router was already at 100% on dev, so the expected gain was small, but its starting check did find something to learn: without enforced JSON, the router wraps most answers in a markdown code block.

**What would you do for a customer deploying this?**
Keep decisions in code with one rule per policy clause. Start with a small model for routing and wording, run a prompt control before paying for training, and fine-tune only where a held-out set shows a gap. Track harmful actions, false promises, and grounded citations as separate release gates. Serve on a pinned image, profile before optimizing, and treat any hosted endpoint as unreliable, with a timeout and a local fallback.

**What did it cost?**
The agent runs on one RTX 3090. All Phase 6 GPU time on Brev came to about $3.42: 2.1 hours of an A100 80 GB.

**What is still open?**
The writer sometimes cites a clause when it is only asking for an order number. That is the one remaining failure on the 62 cases. A templated reply for that outcome would remove a model call and the failure together.
