# References

Generated: 2026-02-18T10:33:02.775Z

## 1. Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem
- URL: https://arxiv.org/abs/2512.24873
- Cite Key: `@weixun2025`
- Type: academic
- Authors: Wang, Weixun, Xu, XiaoXiao, An, Wanhe, Dai, Fangwen, Gao, Wei, He, Yancheng, Huang, Ju, Ji, Qiang, Jin, Hanqi, Li, Xiaoyang, Li, Yang, Li, Zhongwen, Lin, Shirong, Liu, Jiashun, Liu, Zenan, Luo, Tao, Muhtar, Dilxat, Qu, Yuanbin, Shi, Jiaqiang, Sun, Qinghui, Tan, Yingshui, Tang, Hao, Wang, Runze, Wang, Yi, Wang, Zhaoguo, Wu, Yanan, Xiong, Shaopan, Xu, Binchen, Xu, Xander, Xu, Yuchi, Zhang, Qipeng, Zhang, Xixia, Zhao, Haizhou, Zhao, Jie, Zhao, Shuaibing, Zheng, Baihui, Zheng, Jianhui, Zheng, Suhang, Zhu, Yanni, Cai, Mengze, Cao, Kerui, Chen, Xitong, Dai, Yue, Du, Lifan, Feng, Tao, He, Tao, Hu, Jin, Hu, Yijie, Jiang, Ziyu, Li, Cheng, Li, Xiang, Liang, Jing, Lin, Xin, Liu, Chonghuan, Liu, ZhenDong, Lv, Zhiqiang, Mi, Haodong, Mo, Yanhu, Ni, Junjia, Pei, Shixin, Shen, Jingyu, Song, XiaoShuai, Wang, Cecilia, Wang, Chaofan, Wang, Kangyu, Wang, Pei, Wang, Tao, Wang, Wei, Xiao, Ke, Xu, Mingyu, Xu, Tiange, Ya, Nan, Yang, Siran, Ye, Jianan, Zang, Yaxing, Zhang, Duo, Zhang, Junbo, Zheng, Boren, Deng, Wanxi, Pan, Ling, Qu, Lin, Su, Wenbo, Wang, Jiamang, Wang, Wei, Wei, Hu, Wu, Minggang, Yu, Cheng, Zhao, Bing, Zheng, Zhicheng, Zheng, Bo
- Date: 2025/12/31
- Description:
  Agentic crafting requires LLMs to operate in real-world environments over multiple turns by taking actions, observing outcomes, and iteratively refining artifacts. Despite its importance, the open-source community lacks a principled, end-to-end ecosystem to streamline agent development. We introduce the Agentic Learning Ecosystem (ALE), a foundational infrastructure that optimizes the production pipeline for agentic model. ALE consists of three components: ROLL, a post-training framework for weight optimization; ROCK, a sandbox environment manager for trajectory generation; and iFlow CLI, an agent framework for efficient context engineering. We release ROME, an open-source agent grounded by ALE and trained on over one million trajectories. Our approach includes data composition protocols for synthesizing complex behaviors and a novel policy optimization algorithm, Interaction-Perceptive Agentic Policy Optimization (IPA), which assigns credit over semantic interaction chunks rather than individual tokens to improve long-horizon training stability. Empirically, we evaluate ROME within a structured setting and introduce Terminal Bench Pro, a benchmark with improved scale and contamination control. ROME demonstrates strong performance across benchmarks like SWE-bench Verified and Terminal Bench, proving the effectiveness of ALE.

## 2. Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning
- URL: https://arxiv.org/abs/2602.10090
- Cite Key: `@zhaoyang2026`
- Type: academic
- Authors: Wang, Zhaoyang, Xu, Canwen, Liu, Boyi, Wang, Yite, Han, Siwei, Yao, Zhewei, Yao, Huaxiu, He, Yuxiong
- Date: 2026/02/10
- Tags: Agent, World Model
- Description:
  Recent advances in large language model (LLM) have empowered autonomous agents to perform complex tasks that require multi-turn interactions with tools and environments. However, scaling such agent training is limited by the lack of diverse and reliable environments. In this paper, we propose Agent World Model (AWM), a fully synthetic environment generation pipeline. Using this pipeline, we scale to 1,000 environments covering everyday scenarios, in which agents can interact with rich toolsets (35 tools per environment on average) and obtain high-quality observations. Notably, these environments are code-driven and backed by databases, providing more reliable and consistent state transitions than environments simulated by LLMs. Moreover, they enable more efficient agent interaction compared with collecting trajectories from realistic environments. To demonstrate the effectiveness of this resource, we perform large-scale reinforcement learning for multi-turn tool-use agents. Thanks to the fully executable environments and accessible database states, we can also design reliable reward functions. Experiments on three benchmarks show that training exclusively in synthetic environments, rather than benchmark-specific ones, yields strong out-of-distribution generalization. The code is available at https://github.com/Snowflake-Labs/agent-world-model.

## 3. SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents
- URL: https://arxiv.org/abs/2602.12984
- Cite Key: `@yujiong2026`
- Type: academic
- Authors: Shen, Yujiong, Yang, Yajie, Xi, Zhiheng, Hu, Binze, Sha, Huayu, Zhang, Jiazheng, Peng, Qiyuan, Shang, Junlin, Huang, Jixuan, Fan, Yutao, Tong, Jingqi, Dou, Shihan, Zhang, Ming, Bai, Lei, Yin, Zhenfei, Gui, Tao, Ma, Xingjun, Zhang, Qi, Huang, Xuanjing, Jiang, Yu-Gang
- Date: 2026/02/13
- Tags: Gym, Scientic Agent
- Description:
  Scientific reasoning inherently demands integrating sophisticated toolkits to navigate domain-specific knowledge. Yet, current benchmarks largely overlook agents' ability to orchestrate tools for such rigorous workflows. To bridge this gap, we introduce SciAgentGym, a scalable interactive environment featuring 1,780 domain-specific tools across four natural science disciplines, supported by a robust execution infrastructure. Complementing this, we present SciAgentBench, a tiered evaluation suite designed to stress-test agentic capabilities from elementary actions to long-horizon workflows. Our evaluation identifies a critical bottleneck: state-of-the-art models struggle with complex scientific tool-use. Even for a leading model like GPT-5, success rates drop sharply from 60.6% to 30.9% as interaction horizons extend, primarily due to failures in multi-step workflow execution. To address this, we propose SciForge, a data synthesis method that models the tool action space as a dependency graph to generate logic-aware training trajectories. By fine-tuning on these trajectories, our SciAgent-8B outperforms the significantly larger Qwen3-VL-235B-Instruct while exhibiting positive cross-domain transfer of scientific tool-use capabilities. These results underscore the promising potential of next-generation autonomous scientific agents.

## 4. Training-Free Group Relative Policy Optimization
- URL: https://arxiv.org/abs/2510.08191v1
- Cite Key: `@yuzheng2025`
- Type: academic
- Authors: Cai, Yuzheng, Cai, Siqi, Shi, Yuchen, Xu, Zihan, Chen, Lichao, Qin, Yulei, Tan, Xiaoyu, Li, Gang, Li, Zongyi, Lin, Haojia, Mao, Yong, Li, Ke, Sun, Xing
- Date: 2025/10/09
- Tags: GRPO
- Description:
  Recent advances in Large Language Model (LLM) agents have demonstrated their promising general capabilities. However, their performance in specialized real-world domains often degrades due to challenges in effectively integrating external tools and specific prompting strategies. While methods like agentic reinforcement learning have been proposed to address this, they typically rely on costly parameter updates, for example, through a process that uses Supervised Fine-Tuning (SFT) followed by a Reinforcement Learning (RL) phase with Group Relative Policy Optimization (GRPO) to alter the output distribution. However, we argue that LLMs can achieve a similar effect on the output distribution by learning experiential knowledge as a token prior, which is a far more lightweight approach that not only addresses practical data scarcity but also avoids the common issue of overfitting. To this end, we propose Training-Free Group Relative Policy Optimization (Training-Free GRPO), a cost-effective solution that enhances LLM agent performance without any parameter updates. Our method leverages the group relative semantic advantage instead of numerical ones within each group of rollouts, iteratively distilling high-quality experiential knowledge during multi-epoch learning on a minimal ground-truth data. Such knowledge serves as the learned token prior, which is seamlessly integrated during LLM API calls to guide model behavior. Experiments on mathematical reasoning and web searching tasks demonstrate that Training-Free GRPO, when applied to DeepSeek-V3.1-Terminus, significantly improves out-of-domain performance. With just a few dozen training samples, Training-Free GRPO outperforms fine-tuned small LLMs with marginal training data and cost.

## 5. GitHub - p-e-w/heretic: Fully automatic censorship removal for language models
- URL: https://github.com/p-e-w/heretic
- Cite Key: `@github2026`
- Type: blog
- Description:
  Fully automatic censorship removal for language models - p-e-w/heretic

## 6. GitHub - ComposioHQ/composio: Composio equips your AI agents & LLMs with 100+ high-quality integrations via function calling
- URL: https://github.com/ComposioHQ/composio
- Cite Key: `@github2026a`
- Type: blog
- Description:
  Composio equips your AI agents & LLMs with 100+ high-quality integrations via function calling - ComposioHQ/composio

## 7. "Compound Engineering" - Introducing it using Codex Skills - SnaqSh0t
- URL: https://limc.dev/log/article/dev/ai/ar-da-02-en-compound-engineering-introducing-it-using-codex-skills/
- Cite Key: `@compound2026`
- Type: blog
- Description:
  기록 남기는 곳

## 8. GitHub - maxritter/claude-pilot: Claude Code is powerful. Pilot makes it reliable. Start a task, grab a coffee, come back to production-grade code. Tests enforced. Context preserved. Quality automated.
- URL: https://github.com/maxritter/claude-pilot
- Cite Key: `@github2026b`
- Type: blog
- Description:
  Claude Code is powerful. Pilot makes it reliable. Start a task, grab a coffee, come back to production-grade code. Tests enforced. Context preserved. Quality automated. - maxritter/claude-pilot

## 9. GitHub - brendanhogan/hermitclaw
- URL: https://github.com/brendanhogan/hermitclaw/
- Cite Key: `@github2026c`
- Type: blog
- Description:
  Contribute to brendanhogan/hermitclaw development by creating an account on GitHub.

## 10. Synthetic Interaction Data for Scalable Personalization in Large Language Models
- URL: https://arxiv.org/abs/2602.12394
- Cite Key: `@yuchen2026`
- Type: academic
- Authors: Ma, Yuchen, Huang, Yue, Wang, Wenjie, Luo, Xiaonan, Zhang, Xiangliang, Feuerriegel, Stefan
- Date: 2026/02/12
- Description:
  Personalized prompting offers large opportunities for deploying large language models (LLMs) to diverse users, yet existing prompt optimization methods primarily focus on task-level optimization while largely overlooking user-specific preferences and latent constraints of individual users. This gap is primarily due to (i) the absence of high-quality, privacy-sensitive data that capture personalized user-LLM interactions at scale, and (ii) the lack of robust reward signals for individual preferences. To overcome existing data limitations, we introduce a high-fidelity synthetic data generation framework called PersonaGym. Unlike prior work that treats personalization as static persona-preference pairs, PersonaGym models a dynamic preference process via an agentic LLM system to simulate realistic preference behaviors and semantic-aware noise in order to generate personalized multi-turn interaction trajectories. Using PersonaGym, we release PersonaAtlas, a large-scale, high-quality, and diverse synthetic dataset of high-fidelity multi-turn personalized interaction trajectories that closely mirror real-world preference expression and noise patterns. We further propose Personalized Prompt Optimization (PPOpt), a scalable and model-agnostic framework that optimizes user prompts based on interaction histories without modifying the deployed LLM. PPOpt adopts a reason-then-optimize paradigm that infers an explicit user profile and conditions prompt rewriting on the user profile to avoid reward hacking. Our training procedure for PPOpt integrates a cold-start supervised prior with outcome-driven multi-objective reinforcement learning. We present extensive experiments to demonstrate consistent improvements over state-of-the-art baselines in terms of task performance, personalization quality, and robustness to noisy as well as to sparse preference signals.

## 11. Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents
- URL: https://arxiv.org/abs/2602.12662
- Cite Key: `@ruihan2026`
- Type: academic
- Authors: Yang, Ruihan, Ye, Fanghua, We, Xiang, Zhao, Ruoqing, Luo, Kang, Xu, Xinbo, Zhao, Bo, Ma, Ruotian, Wang, Shanyi, Tu, Zhaopeng, Li, Xiaolong, Yang, Deqing, Linus
- Date: 2026/02/13
- Tags: Agent, Level thinking
- Description:
  Large language models (LLMs) are increasingly deployed as autonomous agents for multi-turn decision-making tasks. However, current agents typically rely on fixed cognitive patterns: non-thinking models generate immediate responses, while thinking models engage in deep reasoning uniformly. This rigidity is inefficient for long-horizon tasks, where cognitive demands vary significantly from step to step, with some requiring strategic planning and others only routine execution. In this paper, we introduce CogRouter, a framework that trains agents to dynamically adapt cognitive depth at each step. Grounded in ACT-R theory, we design four hierarchical cognitive levels ranging from instinctive responses to strategic planning. Our two-stage training approach includes Cognition-aware Supervised Fine-tuning (CoSFT) to instill stable level-specific patterns, and Cognition-aware Policy Optimization (CoPO) for step-level credit assignment via confidence-aware advantage reweighting. The key insight is that appropriate cognitive depth should maximize the confidence of the resulting action. Experiments on ALFWorld and ScienceWorld demonstrate that CogRouter achieves state-of-the-art performance with superior efficiency. With Qwen2.5-7B, it reaches an 82.3% success rate, outperforming GPT-4o (+40.3%), OpenAI-o3 (+18.3%), and GRPO (+14.0%), while using 62% fewer tokens.

## 12. WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning
- URL: https://arxiv.org/abs/2602.12852
- Cite Key: `@junjie2026`
- Type: academic
- Authors: Wang, Junjie, Xie, Zequn, Yang, Dan, Feng, Jie, Shen, Yue, Sun, Duolin, Long, Meixiu, Jiao, Yihan, Tan, Zhehao, Wang, Jian, Wei, Peng, Gu, Jinjie
- Date: 2026/02/13
- Tags: Agent, Web, WebAgent
- Description:
  Deep Research systems based on web agents have shown strong potential in solving complex information-seeking tasks, yet their search efficiency remains underexplored. We observe that many state-of-the-art open-source web agents rely on long tool-call trajectories with cyclic reasoning loops and exploration of unproductive branches. To address this, we propose WebClipper, a framework that compresses web agent trajectories via graph-based pruning. Concretely, we model the agent's search process as a state graph and cast trajectory optimization as a minimum-necessary Directed Acyclic Graph (DAG) mining problem, yielding pruned trajectories that preserve essential reasoning while eliminating redundant steps. Continued training on these refined trajectories enables the agent to evolve toward more efficient search patterns and reduces tool-call rounds by about 20% while improving accuracy. Furthermore, we introduce a new metric called F-AE Score to measure the model's overall performance in balancing accuracy and efficiency. Experiments demonstrate that WebClipper compresses tool-call rounds under excellent performance, providing practical insight into balancing effectiveness and efficiency in web agent design.

## 13. Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning
- URL: https://arxiv.org/abs/2602.11748
- Cite Key: `@futing2026`
- Type: academic
- Authors: Wang, Futing, Yan, Jianhao, Luo, Yun, Cui, Ganqu, Wang, Zhi, Qu, Xiaoye, Zhang, Yue, Cheng, Yu, Lin, Tao
- Date: 2026/02/12
- Folders: Agent
- Description:
  Achieving effective test-time scaling requires models to engage in In-Context Exploration -- the intrinsic ability to generate, verify, and refine multiple reasoning hypotheses within a single continuous context.
    Grounded in State Coverage theory, our analysis identifies a critical bottleneck to enabling this capability: while broader state coverage requires longer reasoning trajectories, the probability of sampling such sequences decays exponentially during autoregressive generation, a phenomenon we term the ``Shallow Exploration Trap''.
    To bridge this gap, we propose Length-Incentivized Exploration(\method).
    This simple yet effective recipe explicitly encourages models to explore more via a length-based reward coupled with a redundancy penalty, thereby maximizing state coverage in two-step manner.
    Comprehensive experiments across different models (Qwen3, Llama) demonstrate that \method effectively incentivize in-context exploration.
    As a result, our method achieves an average improvement of 4.4\% on in-domain tasks and a 2.7\% gain on out-of-domain benchmarks.
