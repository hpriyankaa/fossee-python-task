# Fossee-python-task-3

Task Title: "Python Screening Task 3: Evaluating Open Source Models for Student Competence Analysis"
<br>
Name: H Priyanka
<br>
Email: priyankah2407@gmail.com
<br>

# Table of Contents
- [Research Plan](#research-plan)
- [Reasoning](#reasoning)
- [Benchmarks](#benchmarks)
- [Prototype](#prototype)
- [Future Plan](#future-research-plan)
- [References](#references)

# Research Plan
To assess LLMs and Open Source Models on their ability to support high-level student competence analysis in Python, I researched on various models such as CodeLlama[6], DeepSeek Coder v2[8], DeepSeek-V3[9], Qwen3-Coder[10], StarCoder[1], WizardCoder[2], and Codestral[3][4]. While each had it’s own advantages,like CodeLlama’s[6] specialization in Python, DeepSeek’s[8][9] scalability, Qwen3-Coder’s[10] long context. These models have constraints like high resource demands or less established ecosystems. Out of all these choices, I find Qwen2.5-Coder(instruction-tuned)[5] as the most balanced option because it is education-task-oriented, exhibits high-performance on the benchmarks such as HumanEval[11] and MBPP[12], and can be tested and scaled across parameters 0.5B to 32B on basis of available compute resources. A model is suitable for competence analysis if it can not only solve problems but analyzes the student's code, surfaces misconceptions, and proposes prompts that encourage further reasoning without directly revealing the final solution. Qwen2.5-Coder-Instruct[5] fulfills all these criterion better than other models.

To validate this, I developed a Planner–Executor–Critic (PEC)[15] prototype where the Planner detects concepts and gaps, the Executor generates Socratic-style questions, and the Critic checks their depth and clarity. This framework offers a systematic means to quantify whether the model produces meaningful prompts that surface reasoning errors and leads to further additional learning. The results showed that Qwen2.5-Coder-Instruct[5] gave interpretable, pedagogically useful feedback with reasonable computational cost.Despite newer models like Qwen3-Coder[10] pushing raw accuracy higher and DeepSeek-V3[9] excelling at general-purpose reasoning, their higher resource demands and less established ecosystems makes them less practical for this use case and Qwen3-Coder[10] is still in earlier stages of development. In contrast, Qwen2.5-Coder[5] offers proven reliability, better integration resources, and extensive real-world testing. Collectively, it can be said that these factors assures Qwen2.5-Coder-Instruct[5] is not only the most practical choice  but also the model best positioned to bring together accuracy, interpretability, and scalability to actual educational applications.

# Reasoning

## What makes a model suitable for high-level competence analysis?
A suitable model should be able to analyze student-written code, identify misconceptions, and generate probing prompts that encourage reasoning without revealing full solutions. Beyond accuracy, it should be trained on relevant datasets that capture reasoning patterns and common errors, so its feedback is aligned with real learning contexts. The model should also provide interpretable and scalable feedback that works in both classroom and individual settings, and be supported by a mature ecosystem with active development, frequent commits, and integration resources to ensure long-term reliability and practicality.

## How would you test whether a model generates meaningful prompts?
I would run it with a PEC-type configuration: the Planner characterizes ideas and gaps in student code, the Executor generates Socratic prompts and the Critic evaluates them in terms of relevance, depth, clarity and whether they avoid revealing the final solution.  A simple rubric and limited student trials can then confirm if prompts actually help surface misconceptions and guide deeper reasoning

## What trade-offs might exist between accuracy, interpretability, and cost?
Models like Qwen3 or DeepSeek-V3 achieve higher accuracy and can tackle more complex tasks, but they require greater compute resources, their reasoning is often less transparent, and their ecosystems are still evolving with fewer integration resources and community contributions. By contrast, the instruction-tuned Qwen2.5 is more efficient to run, produces clearer, more interpretable feedback for competence analysis, and benefits from a more established ecosystem with better integration support and active development, though it does not reach the very top end of accuracy in comparison to newer models. The trade-off is selecting a model that balances credibility in results with interpretability for learners, cost-effectiveness for deployment, and ecosystem maturity for long-term reliability.

## Why did you choose the model you evaluated, and what are its strengths or limitations?
I chose the instruction-tuned Qwen2.5-Coder over the base model because it is optimized for educational settings able to follow instructions, surface misconceptions, and generate reasoning-focused prompts, which the base model cannot do reliably. Compared to newer models like Qwen3 or DeepSeek-V3, it offers a stronger balance of benchmark accuracy (HumanEval, MBPP) and scalability across 0.5B–32B parameters, without the high compute costs or less mature ecosystems of those alternatives. Its strengths are instruction-tuning, flexible deployment, and interpretable outputs. Its limitation is that it does not yet match the very highest raw accuracy or longest context handling of the latest models, but these trade-offs are offset by its stability and practicality for real-world educational use.




# Benchmarks
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/benchmarks1.png" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 1: Qwen 2.5-Coder Benchmark vs other coding models [13]</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/benchmark2.png" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 2: Qwen2.5-Coder model size comparison [13]</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/roundbenchmark.png" alt="Sample Image" width="900" height="600">
<p align="center"><em>Figure 3: Qwen2.5-Coder vs peers across tasks [13]</em></p>
<br>
<img src="https://github.com/hpriyankaa/fossee-python-task/blob/main/assets/evalcomparison.png" alt="Sample Image" width="900" height="1200">
<p align="center"><em>Figure 4: EvalPlus Benchmark Leaderboad [14]</em></p>


# Prototype

As part of this work, we built a **prototype PEC pipeline** that analyzes student-written Python code, identifies key concepts and misconceptions, generates **Socratic-style questions**, and evaluates them using a structured rubric.

This prototype is lightweight, runs on open-source models (e.g., **Qwen2.5-Coder** via vLLM), and produces clean reports in both console and artifact formats (JSONL, CSV, Markdown).

Explore the full prototype, installation steps, and detailed usage here:  
[**Prototype Directory**](https://github.com/hpriyankaa/fossee-python-task/tree/main/Prototype).

<br>

# Future Research Plan

## Cross-Model Comparisons

Conduct side-by-side evaluations of Qwen2.5-Coder, Qwen3-Coder, and DeepSeek Coder v2 using the same PEC prototype. Track differences in accuracy, interpretability, efficiency, and pedagogical usefulness.

## Instruction-Tuning Customization

Fine-tune Qwen2.5-Coder-Instruct on education-specific datasets (e.g., student submissions with annotated misconceptions) to better align the model with pedagogical needs.

## Prototype Enhancement (PEC 2.0)

Expand the Planner–Executor–Critic pipeline into a fully adaptive feedback loop, where identified misconceptions automatically trigger new rounds of targeted Socratic questioning. Add multi-round evaluation to measure whether students improve after model-driven feedback.

<br>

# References
[1]R. Li et al., “StarCoder: may the source be with you!,” arXiv.org, May 09, 2023. https://arxiv.org/abs/2305.06161. 

[2] Z. Luo et al., “WizardCoder: Empowering Code Large Language Models with Evol-Instruct,” arXiv.org, Jun. 14, 2023. https://arxiv.org/abs/2306.08568.

[3] Mistral AI, “Codestral | Mistral AI,” Mistral.ai, May 29, 2024. [Online]. Available: https://mistral.ai/news/codestral

[4] Hugging Face, “mistralai/Codestral-22B-v0.1,” Hugging Face, May 29, 2024. [Online]. Available: https://huggingface.co/mistralai/Codestral-22B-v0.1

[5] B. Hui et al., “QWen2.5-Coder Technical Report,” arXiv.org, Sep. 18, 2024. https://arxiv.org/abs/2409.12186.

[6] B. Rozière et al., “Code llama: Open Foundation Models for code,” arXiv.org, Aug. 24, 2023. https://arxiv.org/abs/2308.12950

[7] Meta, “Meta Code Llama,” Llama.com, Aug. 24, 2023. [Online]. Available: https://www.llama.com/code-llama/

[8] DeepSeek-Ai et al., “DeepSeek-Coder-V2: Breaking the barrier of Closed-Source Models in Code Intelligence,” arXiv (Cornell University), Jun. 2024, doi: 10.48550/arxiv.2406.11931. Available: https://www.researchgate.net/publication/381517674_DeepSeek-Coder-V2_Breaking_the_Barrier_of_Closed-Source_Models_in_Code_Intelligence

[9] DeepSeek-Ai et al., “DeepSeek-V3 Technical Report,” arXiv.org, Dec. 27, 2024. https://arxiv.org/abs/2412.19437v1

[10] A. Yang et al., “QWEN3 Technical Report,” arXiv.org, May 14, 2025. Available: https://arxiv.org/abs/2505.09388

[11] M. Chen et al., “Evaluating large language models trained on code,” arXiv.org, Jul. 07, 2021. https://arxiv.org/abs/2107.03374

[12] J. Austin et al., “Program Synthesis with Large Language Models,” arXiv.org, Aug. 16, 2021. https://arxiv.org/abs/2108.07732

[13] Qwen Team, “QWeN2.5-Coder: Code more, learn more!,” Qwen, Sep. 18, 2024. https://qwenlm.github.io/blog/qwen2.5-coder/

[14] EvalPlus, “EvalPlus leaderboard,” EvalPlus. [Online]. Available: https://evalplus.github.io/leaderboard.html

[15] Anthropic, “Building effective AI agents,” Anthropic. [Online]. Available: https://www.anthropic.com/engineering/building-effective-agents
