# Fossee-python-task

## Research Plan
During my research, I explored about various open source models and LLMs such as StarCoder[1], WizardCoder[2], Codestral[3][4] recently released models like Qwen2.5-Coder[5], CodeLlama[6][7], DeepSeek Coder v2[8], DeepSeek-V3[9], and Qwen3-Coder[10].Each model has it’s advantages, CodeLlama[6][7] has python-focused variant and large user base as it’s strength,DeepSeek Coder v2[8] gives a strong balance of scalability and efficiency, DeepSeek V3[9] despite being a powerful general-purpose foundation model, it is not exclusive for coding tasks and Qwen3[10] has advanced features like longer context windows and larger scaling options, but it’s ecosystem is still young.Despite these strengths, I chose Qwen2.5-Coder[5], particularly its instruction-tuned variants (0.5B–32B), as the most suitable for Python competence analysis. On benchmarks such as HumanEval[11] and MBPP[12], it not only demonstrated strong accuracy in reasoning but also proved to be adaptable, scalable across hardware, and interoperable with existing educational workflows—making it the most balanced option for immediate deployment.


A model is considered appropriate for high-level competence analysis if it not only solves coding problems but analyzes the student's code, surfaces misconceptions, and proposes prompts that encourage further reasoning without giving away the final solution.I specifically used the instruction-tuned version of Qwen2.5-Coder[5] instead of using the base model since it's trained to comply with educational instructions, produce reasoning-related responses, and point out missing concepts which the base model lacks.To evaluate and validate this, I developed a prototype Planner-Executor-Critic (PEC). The Planner finds concepts, Bloom-level, and possible gaps; the Executor produces Socratic-like probational questions; and the Critic tests those questions for relevance, depth, and clarity.Running this pipeline confirmed that Qwen2.5’s[5] instruction tuning and maturity allow it to produce feedback that is accurate, interpretable, and pedagogically meaningful.For future research, this work can be extended by conducting cross-model comparisons with Qwen3-Coder[10] and DeepSeek Coder v2[8] to explore trade-offs, by fine-tuning Qwen2.5-Instruct[5] on student-specific datasets to enhance prompt generation quality, and by evolving the PEC framework into an adaptive multi-round system (PEC 2.0) that dynamically adjusts prompts as misconceptions are uncovered. Despite newer models like Qwen3-Coder[10] pushing raw accuracy higher and DeepSeek-V3[9] excelling at general-purpose reasoning, their higher resource demands and less established ecosystems make them less practical for this use case and Qwen3-Coder[10] is still in earlier stages of development, with its tools and community support continuing to grow. In contrast, Qwen2.5-Coder[5] offers proven reliability, better integration resources, and extensive real-world testing which reinforces that it’s a more reliable choice for this specific usecase. In conclusion, after an extensive technical analysis, benchmark comparisons on the model, and validating its practicality through my prototype, these factors reassured me to confidently state that Qwen2.5-Coder[5] is the most optimal and well-suited model for our use case.

## Benchmarks
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


## Prototype
## References
[1] R. Li et al., “StarCoder: may the source be with you!,” arXiv.org, May 09, 2023. https://arxiv.org/abs/2305.06161.


[2] Z. Luo et al., “WizardCoder: Empowering Code Large Language Models with Evol-Instruct,” arXiv.org, Jun. 14, 2023. https://arxiv.org/abs/2306.08568.


[3] “Codestral | Mistral AI.” https://mistral.ai/news/codestral “mistralai/Codestral-22B-v0.1.


[4] Hugging Face.” https://huggingface.co/mistralai/Codestral-22B-v0.1.


[5] B. Hui et al., “QWen2.5-Coder Technical Report,” arXiv.org, Sep. 18, 2024. https://arxiv.org/abs/2409.12186.


[6] “Code llama: Open Foundation Models for code.” Available: https://arxiv.org/html/2308.12950

[7] “Meta Code Llama,” Meta Llama. Available: https://www.llama.com/code-llama/

[8] DeepSeek-Ai et al., “DeepSeek-Coder-V2: Breaking the barrier of Closed-Source Models in Code Intelligence,” arXiv (Cornell University), Jun. 2024, doi: 10.48550/arxiv.2406.11931. Available: https://www.researchgate.net/publication/381517674_DeepSeek-Coder-V2_Breaking_the_Barrier_of_Closed-Source_Models_in_Code_Intelligence

[9] “DeepSeek-V3 Technical Report.” Available: https://arxiv.org/html/2412.19437v1

[10] A. Yang et al., “QWEN3 Technical Report,” arXiv.org, May 14, 2025. Available: https://arxiv.org/abs/2505.09388

[11] M. Chen et al., “Evaluating large language models trained on code,” arXiv.org, Jul. 07, 2021. https://arxiv.org/abs/2107.03374

[12] J. Austin et al., “Program Synthesis with Large Language Models,” arXiv.org, Aug. 16, 2021. https://arxiv.org/abs/2108.07732

[13] Qwen Team, “QWeN2.5-Coder: Code more, learn more!,” Qwen, Sep. 18, 2024. https://qwenlm.github.io/blog/qwen2.5-coder/

[14] “EvalPlus leaderboard.” https://evalplus.github.io/leaderboard.html





