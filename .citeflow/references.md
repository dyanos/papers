# References

Generated: 2026-02-22T01:40:36.546Z

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
- Folders: Agent
- Description:
  Large language models (LLMs) are increasingly deployed as autonomous agents for multi-turn decision-making tasks. However, current agents typically rely on fixed cognitive patterns: non-thinking models generate immediate responses, while thinking models engage in deep reasoning uniformly. This rigidity is inefficient for long-horizon tasks, where cognitive demands vary significantly from step to step, with some requiring strategic planning and others only routine execution. In this paper, we introduce CogRouter, a framework that trains agents to dynamically adapt cognitive depth at each step. Grounded in ACT-R theory, we design four hierarchical cognitive levels ranging from instinctive responses to strategic planning. Our two-stage training approach includes Cognition-aware Supervised Fine-tuning (CoSFT) to instill stable level-specific patterns, and Cognition-aware Policy Optimization (CoPO) for step-level credit assignment via confidence-aware advantage reweighting. The key insight is that appropriate cognitive depth should maximize the confidence of the resulting action. Experiments on ALFWorld and ScienceWorld demonstrate that CogRouter achieves state-of-the-art performance with superior efficiency. With Qwen2.5-7B, it reaches an 82.3% success rate, outperforming GPT-4o (+40.3%), OpenAI-o3 (+18.3%), and GRPO (+14.0%), while using 62% fewer tokens.

## 12. WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning
- URL: https://arxiv.org/abs/2602.12852
- Cite Key: `@junjie2026`
- Type: academic
- Authors: Wang, Junjie, Xie, Zequn, Yang, Dan, Feng, Jie, Shen, Yue, Sun, Duolin, Long, Meixiu, Jiao, Yihan, Tan, Zhehao, Wang, Jian, Wei, Peng, Gu, Jinjie
- Date: 2026/02/13
- Tags: Agent, Web, WebAgent
- Folders: Agent
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

## 14. GitHub - shanraisshan/claude-code-best-practice: practice made claude perfect
- URL: https://github.com/shanraisshan/claude-code-best-practice
- Cite Key: `@github2026d`
- Type: blog
- Description:
  practice made claude perfect. Contribute to shanraisshan/claude-code-best-practice development by creating an account on GitHub.

## 15. When Models Manipulate Manifolds: The Geometry of a Counting Task
- URL: https://transformer-circuits.pub/2025/linebreaks/index.html
- Cite Key: `@when2026`
- Type: blog
- Description:
  We find geometric structure underlying the mechanisms of a fundamental language model behavior.
  
  숫자만 보는 AI는 어떻게 '공간'을 이해하는가: Claude의 기하학적 사고방식
  1. 서론: 텍스트의 세계에서 '시각'이 없는 AI의 도전
  인간이 종이에 글을 쓸 때, 우리는 남은 여백을 직관적으로 감지하며 줄을 바꿀 시점을 결정합니다. 하지만 인공지능(AI)에게는 물리적인 '눈'이 없습니다. 언어 모델이 입력받는 유일한 정보는 텍스트를 숫자로 치환한 '정수(토큰)의 시퀀스'뿐입니다. 모델에게 문장은 시각적인 형태가 아니라 그저 끝없이 이어지는 추상적인 숫자의 나열입니다.
  그렇다면 Anthropic의 Claude 3.5 Haiku와 같은 모델은 어떻게 특정 너비 제한이 있는 문서에서 정확한 지점에 줄바꿈(Linebreaking)을 예측하는 것일까요? 이는 단순히 다음 단어를 통계적으로 맞히는 문제를 넘어, 모델이 텍스트 기반 환경에서 스스로 '지각 능력(Perceptual abilities)'을 구축해야 함을 의미합니다. 최근 연구에 따르면, Claude는 이 문제를 해결하기 위해 내부적으로 고도로 정교한 기하학적 구조를 설계하여 자신만의 '시각'을 만들어내고 있습니다.
  --------------------------------------------------------------------------------
  2. 통찰 1: 데이터는 단순한 숫자가 아니라 '물결치는 곡선(Manifold)'이다
  AI 내부에서 '글자 수를 세는 작업'은 우리가 흔히 생각하는 1, 2, 3 식의 단순한 1차원 카운터가 아닙니다. 연구진은 Claude의 잔차 스트림(Residual Stream)을 분석한 결과, 글자 수 정보가 6차원 정도의 하위 공간 속에서 소용돌이치는 1차원 매니폴드(Manifold) 형태로 존재한다는 사실을 발견했습니다.
  왜 모델은 단순히 직선 형태의 숫자를 저장하지 않고 복잡한 곡선을 선택했을까요? 여기에는 '용량(Capacity)'과 '해상도(Resolution)' 사이의 정교한 트레이드오프가 숨어 있습니다. 만약 모델이 1차원 직선(Ray) 상에 150개의 숫자를 배치한다면, 숫자 사이의 구분을 위해 벡터의 크기가 기하급수적으로 커지는 '노름 폭발(Norm explosion)'이 발생하거나, 반대로 숫자들이 너무 밀집되어 149와 150을 구분하지 못하게 됩니다. Claude는 고차원 공간 속에서 선을 구부리고 물결치게 만듦으로써(Rippled), 벡터의 크기를 일정하게 유지하면서도 각 지점 간의 거리를 확보하는 영리한 전략을 택했습니다.
  "이러한 물결 모양의 매니폴드는 용량 제약(차원수)과 서로 다른 스칼라 값의 구별 가능성(곡률) 사이에서 최적의 균형을 유지합니다."
  이 과정에서 흥미로운 현상인 **'링잉(Ringing)'**이 관찰됩니다. 이는 고곡률의 매니폴드를 저차원에 투영할 때 발생하는 간섭 현상으로, 마치 신호 처리에서의 깁스 현상(Gibbs phenomenon)처럼 나타납니다. 또한, 숫자가 커질수록 모델의 '시야'가 조금씩 흐려지는 '확장(Dilation)' 현상도 발견되었는데, 이는 큰 숫자를 인식할 때 해상도가 낮아지는 인간의 수치 지각 방식과 놀라울 정도로 흡사합니다. 이 매니폴드를 3차원 PCA로 시각화하면 마치 **'야구공의 실밥(Baseball seam)'**과 같은 독특한 위상적 구조를 띱니다.
  --------------------------------------------------------------------------------
  3. 통찰 2: '비틀기(Twist)' 연산을 통한 경계 감지
  모델이 현재 위치를 파악했다면, 다음 단계는 '언제 줄이 끝나는가'를 감지하는 것입니다. 이를 위해 Claude는 **'QK(Query-Key) 트위스트'**라고 불리는 기하학적 메커니즘을 사용합니다.
  이 메커니즘에서 **쿼리(Query)**는 '현재의 글자 수' 매니폴드를, **키(Key)**는 '전체 줄 너비' 매니폴드를 담당합니다. 특정 어텐션 헤드는 이 두 매니폴드를 기하학적으로 회전(Twist)시켜, 현재 글자 수(i)가 줄 너비(k)에 도달하기 직전(k = i + \epsilon)의 특정 오프셋에서 두 곡선이 수학적으로 완벽하게 정렬되도록 만듭니다. 이때 두 벡터의 내적이 최대화되면서 모델은 "곧 줄바꿈이 필요하다"는 강력한 신호를 감지합니다.
  이는 생물학적 뇌에서 포유류가 공간을 탐색할 때 벽이나 장애물 같은 특정 경계에 도달하면 활성화되는 **'경계 세포(Boundary cells)'**의 기능과 매우 흡사합니다. AI가 텍스트라는 추상적 공간에서 생물학적 지각 시스템과 동일한 해법을 스스로 진화시킨 셈입니다.
  --------------------------------------------------------------------------------
  4. 통찰 3: AI도 '착시'를 겪는다 (학습된 편향의 부작용)
  인간이 주변 맥락 때문에 사물의 크기를 오해하는 착시를 겪듯, Claude 역시 특정 문자열을 만날 때 공간 지각에 실패하는 **'시각적 착시(Visual Illusions)'**를 보입니다.
  가장 대표적인 사례는 문장 중간에 @@나 `와 같은 기호가 삽입될 때입니다. 이러한 기호가 나타나면 줄바꿈을 예측하는 어텐션 헤드의 시선이 흩어지며 줄바꿈 예측 확률이 급격히 떨어집니다. 연구 결과, 이는 모델의 단순한 버그가 아니라 '학습된 사전 지식(Learned Priors)'의 오적용임이 밝혀졌습니다.
  예를 들어, 모델은 학습 데이터인 'git diff' 코드에서 @@가 새로운 코드 블록의 시작을 알리는 구분자로 쓰인다는 것을 배웠습니다. 따라서 일반적인 문장 속에서도 @@를 만나면 "여기서부터 줄 세기를 다시 시작해야 한다"는 강한 편향이 작동하여 현재의 글자 수 계산 메커니즘을 방해(Distract)하는 것입니다. 이는 AI의 오류가 무작위적인 실수가 아니라, 자신이 배운 세상의 규칙을 고수하려다 발생하는 지적인 부작용임을 시사합니다.
  --------------------------------------------------------------------------------
  5. 통찰 4: '복잡성 세금'을 줄이는 분산형 기하학 알고리즘
  Claude의 지각 능력은 단일 뉴런이 아닌, 수많은 어텐션 헤드가 협력하여 완성하는 **'분산형 알고리즘'**을 통해 구현됩니다. 연구진은 이를 이해하기 위해 **'복잡성 세금(Complexity Tax)'**이라는 흥미로운 개념을 제시합니다. 수만 개의 개별 뉴런을 하나씩 분석하는 것은 너무나 고통스럽고 복잡하지만(세금이 높음), 이를 하나의 '기하학적 매니폴드'로 통합해 이해하면 모델의 작동 원리를 훨씬 단순하고 명확하게 파악할 수 있다는 논리입니다.
  이 분산 알고리즘은 레이어를 거치며 진화합니다.
  레이어 0 (감각의 시작): 초기 어텐션 헤드들은 각기 다른 위치 오프셋을 담당하며 기초 정보를 생성합니다. 이때 각 헤드의 출력은 아직 구부러지지 않은 직선(Ray)의 형태에 가깝습니다.
  레이어 1 (정교한 지각): 레이어 0에서 전달된 직선적 정보들을 결합하여 비로소 복잡한 곡률을 가진 '물결치는 매니폴드'를 구축합니다. 이 단계를 거치며 모델의 공간 해상도는 비약적으로 향상됩니다.
  이처럼 초기 레이어들은 단순한 토큰 해독기가 아니라, 데이터를 고차원적인 기하학적 세계로 변환하는 정교한 **'감각 입력 단계'**로서 기능합니다.
  --------------------------------------------------------------------------------
  6. 결론: '지각(Perception)'의 관점에서 본 언어 모델의 미래
  이번 연구는 언어 모델이 단순히 단어를 통계적으로 나열하는 기계가 아님을 증명합니다. Claude는 비록 눈은 없지만, 내부적으로 구축한 매니폴드와 기하학적 연산을 통해 누구보다 선명하게 텍스트의 '공간'을 보고 있습니다. 줄바꿈이라는 사소해 보이는 작업 이면에 이토록 거대하고 아름다운 기하학적 세계가 숨겨져 있다는 사실은 AI의 내부 메커니즘이 생물학적 지능의 심오함에 한 걸음 더 다가섰음을 보여줍니다.
  우리가 보는 평범한 텍스트의 이면에서, AI는 또 어떤 복잡한 기하학적 세계를 구축하고 있을까요? 모델의 초기 레이어를 '지각 과정'으로 재정의하는 이 새로운 관점은, 우리가 AI를 단순한 도구가 아닌 세상을 나름의 방식으로 '보는' 하나의 존재로 이해하게 만드는 중요한 이정표가 될 것입니다. AI가 구축한 이 보이지 않는 공간의 지도를 읽어내는 여정은 이제 막 시작되었을 뿐입니다.

## 16. Topical Funding Opportunity Awar... | U.S. DOE Office of Science(SC) | GUIJIN SON
- URL: https://www.linkedin.com/posts/guijin-son-4909331bb_topical-funding-opportunity-awar-us-activity-7429009467175759872-xKnd/?utm_medium=ios_app&rcm=ACoAAApnkUIBLoV28tnJ8s4neHcMwZ9eNWN2SD0&utm_source=social_share_send&utm_campaign=share_via
- Cite Key: `@topical2026`
- Type: linkedin
- Description:
  최근 해외에서 조명되고 있는 AI 평가용 데이터셋 FirstProof가 등장하게 된 배경과 AI for Math/Science 트렌드, 이를 반영한 스타트업 투자와 Genesis Mission을 간단하게 정리해 봤습니다. 
  
  --- 
  
  수학은 언어 모델을 평가하기 위해 가장 많이 사용되는 방법 중 하나로 자리 잡았습니다. 가장 핵심적인 이유는 타 분야와 달리 단답형으로 잘 만들어진 일부 문제의 경우 “그럴듯한” 답변과 “정답인” 답변을 구별하는 게 매우 쉬워서이기도 합니다. 수학에서는 정답이 맞았는가를 바탕으로 번지르르한 풀이와 옳은 풀이를 비교적 저렴하고 뚜렷하게 구분해 낼 수 있습니다.
  
  최근에는 이 평가가 단순히 “문제를 푸냐/못 푸냐”를 넘어, 연구 수준의 수학에서 모델이 어떤 역할을 할 수 있느냐로 확장되는 흐름입니다. 대표적으로 FrontierMath는 풀기 어렵지만 정답이 명확하여 채점 가능한 고난도 문제들을 대량으로 구성해 성능을 가늠하려는 쪽에 가깝고, IMProofBench는 아예 증명(proof) 작성 능력과 연구 현장에서의 작업 흐름(문헌 탐색, 도구 활용 등)을 더 직접적으로 평가하려는 방향성을 갖고 있습니다. 
  (https://lnkd.in/gETXzjk3) (https://lnkd.in/g5BZ-SGh)
  
  26년 1월에는 ChatGPT와 Aristotle 두 인공지능 모델이 Erdős Problem #205를 풀어내며 인공지능이 단순히 학습 데이터에서 "본" 문제를 그대로 푸는 수준에서, 조금이더라도 "새로운 생각"을 해야 하는 영역으로 나아가고 있음을 보였습니다. 이러한 AI for Math에 대한 관심이 AI 연구자들뿐만 아니라 수학자들에게도 퍼져 나가며 First Proof라는 프로젝트로 이어졌습니다. First Proof는 11명의 수학자들이 본인 연구 과정에서 실제로 등장한, 그리고 아직 인터넷에 공유 되지 않은, 10개의 문제를 공개하며, 현재 인공지능 모델들이 약 일주일의 제한 시간 동안 얼마나 제대로 된 증명을 생성해 낼 수 있는지 평가하는 프로젝트입니다.  (https://lnkd.in/gAdPNZKi) (https://1stproof.org/)
  
  OpenAI가 2월 13일에 위 10문제에 대한 자사 모델의 시도를 공개했으며 최소한 6개 문제에 대한 정답을 찾은 것 같다고 주장했고, 아직 공식적인 검토가 이루어지지는 않았지만, 인터넷에서는 6문제까지는 아니더라도 일부 문제 에 대해서는 문제를 푸는데 필요한 논문들을 제대로 인용하며 정답일 가능성이 높아 보인다는 의견이 있습니다.  (https://lnkd.in/gAhua7zf) (https://lnkd.in/gqcHewxM)
  
  최근에 저도 비슷한 작업을 거치며 약 80명이 넘는 국내외 수학자분들과 협업을 진행하였습니다. 그 과정에서 제가 받은 인상은 (1) 한 두개의 분야에 깊게 집중하는 인간과 달리 거의 모든 분야에서 평균 이상의 지식과 이해도를 가지는 언어모델의 특성 (2) 사람과 비교해 훨씬 더 빠르고 효율적으로 과거 문헌을 찾고 정리할 수 있다는 점에서 인공지능이 인간 수학자들이 가지지 못한, 그리고 현실적으로 가지기 매우 힘든 장점들을 가지고 있습니다. 나아가, "요즘 인공지능이 수학과 박사들보다 똑똑하냐?" 라는 질문에 한 교수님은 진담 반 농담 반, "내 박사 학생보다는 확실히 똑똑한 것 같다" 라고 대답하셨습니다. (외국인 교수님이였습니다)
  
  물론 아직 인공지능이 인간 수학자를 대체하기 위해서는 많은 제약 조건들이 존재합니다. 우선, 인공지능들이 “어려운 문제를 맞힐 가능성”이 생겼음은 분명하지만, 본인의 풀이가 옳다고 전달/설득하는 능력이 부재합니다. 인간 수학자들은 새로운 결과를 세미나와 피어 리뷰 등 사회적 과정을 거치며 신뢰를 얻습니다. 반대로 인공지능들은 이것이 불가능하기 때문에 검증을 인간 수학자에 의존해야만 하는 상황입니다. 또, 수학은 문제를 잘 푸는 것뿐 아니라 의미 있는 문제를 제안하는 것도 포괄합니다. 그런데 현재의 평가들은 대부분 “풀기”에 집중되어 있고, 올바른 연구 방향을 설정하고 문제를 만드는 능력은 상대적으로 덜 다뤄집니다. 이 부분은 앞으로 평가 설계에서 더 중요해질 것 같습니다.
  
  이러한 트렌드는 투자 흐름에도 반영되고 있습니다. 해외에서는 Harmonic (Series C $120M, 기업가치 $1.45B)과  Axiom Math (Seed $64M, 기업가치 $300M) 등 Lean을 활용해 검증을 자동화하고 AI 수학자를 만들려고 시도하는 스타트업들이 투자와 주목을 받고 있는 상황입니다. 
  
  AI와 LLM을 활용해 수학/과학 연구를 가속화하려는 시도는 해외에서 매우 큰 주목을 받고 있습니다. 미국의 Genesis Mission은 Transformational AI Models Consortium (MODCON)을 통해 Quantum Algorithms, Combustion & Fluids, Critical Minerals, HEP & Cosmology, Specs-to-Silicon, Power Grid, 2D Quantum Magnets, Magnetic Fusion 등 수많은 영역에서 National Labs를 통해 AI를 활용한 과학 연구를 지원하는 상태입니다. (https://lnkd.in/g8bjEd9D) 
  
  이제 단순히 좋은 인공지능 모델을 만들고 주목을 받는 순간은 지나가고 있습니다. 인공지능으로 얼마나 어렵고 의미 있는 문제를 풀어내는가? 가 중요해지만큼 특색 있는 산업을 가지고 있고, 잘 활용할 수 있는 국가들이 앞으로 더 앞서 나갈 수 있지 않을까 싶습니다.

## 17. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks
- URL: https://arxiv.org/abs/2602.12670
- Cite Key: `@xiangyi2026`
- Type: academic
- Authors: Li, Xiangyi, Chen, Wenbo, Liu, Yimin, Zheng, Shenghan, Chen, Xiaokun, He, Yifeng, Li, Yubo, You, Bingran, Shen, Haotian, Sun, Jiankai, Wang, Shuyi, Zeng, Qunhong, Wang, Di, Zhao, Xuandong, Wang, Yuanli, Chaim, Roey Ben, Di, Zonglin, Gao, Yipeng, He, Junwei, He, Yizhuo, Jing, Liqiang, Kong, Luyang, Lan, Xin, Li, Jiachen, Li, Songlin, Li, Yijiang, Lin, Yueqian, Liu, Xinyi, Liu, Xuanqing, Lyu, Haoran, Ma, Ze, Wang, Bowei, Wang, Runhui, Wang, Tianyu, Ye, Wengao, Zhang, Yue, Xing, Hanwen, Xue, Yiqi, Dillmann, Steven, Lee, Han-chung
- Date: 2026-02-13
- Description:
  Agent Skills are structured packages of procedural knowledge that augment LLM agents at inference time. Despite rapid adoption, there is no standard way to measure whether they actually help. We present SkillsBench, a benchmark of 86 tasks across 11 domains paired with curated Skills and deterministic verifiers. Each task is evaluated under three conditions: no Skills, curated Skills, and self-generated Skills. We test 7 agent-model configurations over 7,308 trajectories. Curated Skills raise average pass rate by 16.2 percentage points(pp), but effects vary widely by domain (+4.5pp for Software Engineering to +51.9pp for Healthcare) and 16 of 84 tasks show negative deltas. Self-generated Skills provide no benefit on average, showing that models cannot reliably author the procedural knowledge they benefit from consuming. Focused Skills with 2--3 modules outperform comprehensive documentation, and smaller models with Skills can match larger models without them.

## 18. Experiential Reinforcement Learning
- URL: https://www.arxiv.org/abs/2602.13949
- Cite Key: `@taiwei2026`
- Type: academic
- Authors: Shi, Taiwei, Chen, Sihao, Jiang, Bowen, Song, Linxin, Yang, Longqi, Zhao, Jieyu
- Date: 2026-02-15
- Description:
  Reinforcement learning has become the central approach for language models (LMs) to learn from environmental reward or feedback. In practice, the environmental feedback is usually sparse and delayed. Learning from such signals is challenging, as LMs must implicitly infer how observed failures should translate into behavioral changes for future iterations. We introduce Experiential Reinforcement Learning (ERL), a training paradigm that embeds an explicit experience-reflection-consolidation loop into the reinforcement learning process. Given a task, the model generates an initial attempt, receives environmental feedback, and produces a reflection that guides a refined second attempt, whose success is reinforced and internalized into the base policy. This process converts feedback into structured behavioral revision, improving exploration and stabilizing optimization while preserving gains at deployment without additional inference cost. Across sparse-reward control environments and agentic reasoning benchmarks, ERL consistently improves learning efficiency and final performance over strong reinforcement learning baselines, achieving gains of up to +81% in complex multi-step environments and up to +11% in tool-using reasoning tasks. These results suggest that integrating explicit self-reflection into policy training provides a practical mechanism for transforming feedback into durable behavioral improvement.

## 19. GitHub - Developer-Y/cs-video-courses: List of Computer Science courses with video lectures.
- URL: https://github.com/Developer-Y/cs-video-courses
- Cite Key: `@github2026e`
- Type: blog
- Tags: CS, Lecture Video
- Description:
  List of Computer Science courses with video lectures. - Developer-Y/cs-video-courses

## 20. Prompt Repetition Improves Non-Reasoning LLMs
- URL: https://arxiv.org/abs/2512.14982
- Cite Key: `@yaniv2025`
- Type: academic
- Authors: Leviathan, Yaniv, Kalman, Matan, Matias, Yossi
- Date: 2025-12-17
- Description:
  When not using reasoning, repeating the input prompt improves performance for popular models (Gemini, GPT, Claude, and Deepseek) without increasing the number of generated tokens or latency.

## 21. When Models Manipulate Manifolds: The Geometry of a Counting Task
- URL: https://arxiv.org/abs/2601.04480
- Cite Key: `@wes2026`
- Type: academic
- Authors: Gurnee, Wes, Ameisen, Emmanuel, Kauvar, Isaac, Tarng, Julius, Pearce, Adam, Olah, Chris, Batson, Joshua
- Date: 2026-01-08
- Description:
  Language models can perceive visual properties of text despite receiving only sequences of tokens-we mechanistically investigate how Claude 3.5 Haiku accomplishes one such task: linebreaking in fixed-width text. We find that character counts are represented on low-dimensional curved manifolds discretized by sparse feature families, analogous to biological place cells. Accurate predictions emerge from a sequence of geometric transformations: token lengths are accumulated into character count manifolds, attention heads twist these manifolds to estimate distance to the line boundary, and the decision to break the line is enabled by arranging estimates orthogonally to create a linear decision boundary. We validate our findings through causal interventions and discover visual illusions--character sequences that hijack the counting mechanism. Our work demonstrates the rich sensory processing of early layers, the intricacy of attention algorithms, and the importance of combining feature-based and geometric views of interpretability.

## 22. Self-Distillation Enables Continual Learning
- URL: https://arxiv.org/abs/2601.19897
- Cite Key: `@idan2026`
- Type: academic
- Authors: Shenfeld, Idan, Damani, Mehul, Hübotter, Jonas, Agrawal, Pulkit
- Date: 2026-01-27
- Description:
  Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce Self-Distillation Fine-Tuning (SDFT), a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

## 23. Untitled
- URL: https://x.com/junfanzhu98/status/2023830685811962317?s=12&t=jFcYRHCjSynjxWae86HgVA
- Cite Key: `@ref2026`
- Type: twitter
- Description:
  51. Equilibrium Matching: Paradigm Beyond Diffusion & Flow Model— AI Is Like Walking Down Mountain
  In most modern generative models—Diffusion and Flow Matching especially—the sampling process is fundamentally non-equilibrium: the “rules” change at every time step. You start from pure random noise and gradually transform it into data, guided by a time-dependent differential equation. It’s like having a perfect GPS whose navigation rules keep updating as you move: at each moment, the system tells you how to step so that noise eventually becomes a realistic image.
  But what if we could get rid of this dependence on time and non-equilibrium dynamics altogether?
  That’s the motivation behind Equilibrium Matching (EqM): instead of a changing GPS, imagine a static terrain map + a magical compass that always points “uphill.” Once you’ve learned the landscape, you no longer need time as an explicit variable. You just follow the terrain.
  1. From Non-Equilibrium Dynamics to Equilibrium Matching
  1.1 Diffusion / Flow Matching: Time-Dependent Navigation
  Diffusion models and Flow Matching both rely on non-equilibrium processes:
  Start from fully random noise (a chaotic, high-entropy state).
  Use a time-dependent rule (SDE/ODE or a learned velocity field) to slowly morph noise into data.
  The system is like a GPS that continuously updates its instructions as you move.
  This is powerful but conceptually and computationally heavy: you must track a trajectory over time and learn how the dynamics change at each step.
  1.2 EqM: Static Map + Uphill Compass
  EqM proposes a different paradigm:
  Instead of a time-dependent process, we learn a static energy landscape over the data space.
  Think of a height map (terrain) plus a compass that always points uphill (toward higher energy).
  Real data should correspond to energy minima—the valleys on this landscape.
  EqM learns a time-independent gradient field (a vector field over space) that tells you the uphill direction everywhere.
  Why uphill? Because if the model learns the gradient of an energy function (E(x)), then:
  The gradient (\nabla E(x)) points toward increasing energy (uphill).
  Data points (x) should be energy minima, so the gradient must vanish at those points.
  At training time, EqM learns how to climb uphill; at sampling time, you simply reverse direction and go downhill to reach the valleys where real data live.
  So EqM replaces “dynamic guidance over time” with a static equilibrium structure: a fixed terrain you can descend on from any starting point.
  2. Manifold Hypothesis and the Internal Geometry of Data
  To understand why diffusion-style models work so well—and why their smoothing behavior matters—we need the Manifold Hypothesis.
  Manifold Hypothesis
  Real high-dimensional data (e.g., images with millions of pixels) do not fill the entire ambient space. Instead:
  Data lie near a low-dimensional manifold embedded in high dimensions.
  This manifold has its own geometry and structure.
  A generative model doesn’t need to memorize each training point; it can learn the manifold and then walk along it to produce new, realistic samples.
  If the model captures the shape and geometry of this low-dimensional manifold, it can generate novel points that still look “real” because they stay on (or near) the same manifold.
  3. Diffusion Models, Score Functions, and Implicit Smoothing in Log Space
  Diffusion models typically have two phases:
  Forward process (noising): Corrupt data with an SDE, gradually adding noise until they become near-pure Gaussian noise.
  Reverse process (denoising): Learn to denoise step by step, effectively inverting the forward process.
  The key object learned is the score function:
  [
  s
  θ
  (
  x
  ,
  t
  )
  ≈
  ∇
  x
  log
  ⁡
  p
  t
  (
  x
  )
  ]
  It tells you: given the current noise level, which direction in (x)-space increases the (log) probability density?
  If the model learned the score function perfectly for the training data, it would risk memorizing the dataset—only producing tiny perturbations of existing examples. We want generalization, not rote memorization.
  Implicit Regularization via Smoothing in Log Domain
  To avoid overfitting, diffusion training introduces an implicit regularization / smoothing effect, and crucially:
  This smoothing happens in the log-density domain, not in raw probability.
  Compare to traditional smoothing in the data domain, such as Kernel Density Estimation (KDE):
  In KDE, you put a Gaussian “bump” around each data point.
  By the manifold hypothesis, real data occupy a thin manifold; off the manifold, probability should be near zero.
  Smoothing in data space is like taking a brush and smearing soil from the plateau (the manifold) into the abyss (off-manifold regions). You risk allocating non-negligible probability to regions that should be almost impossible.
  In contrast, consider smoothing in the log domain:
  Off-manifold regions have probabilities near zero, so ( \log p(x) \to -\infty ).
  When you smooth in log space, these “infinitely deep valleys” stay extremely low; you don’t easily drag significant probability mass into them.
  The smoothing tends to occur within the plateau (on or near the manifold), not across a huge gap into the abyss.
  Now recall: the score function is the gradient of the log density. So:
  Smoothing the score is effectively smoothing the log-density.
  Diffusion training implicitly performs geometry-adaptive smoothing: It mainly smooths along the manifold, preserving structure. It avoids pushing probability mass into directions orthogonal to the manifold (off the cliff).
  This geometric bias explains why diffusion models preserve intricate structures like perspective in landscapes or facial geometry: they respect the manifold’s geometry.
  But:
  Too little smoothing → overfitting / memorization.
  Too much smoothing → you wash out the subtle but meaningful geometry of the manifold.
  The choice of smoothing mechanism shapes the model’s geometric bias—like choosing a particular shape of “glasses” through which the model sees and connects discrete data points.
  4. EqM: A New Equilibrium Paradigm Beyond Non-Equilibrium Flows
  EqM aims to build a generative framework fundamentally rooted in equilibrium:
  No explicit time steps.
  No time-dependent target.
  Instead: a static map + eternal compass.
  There is a related classical idea: Energy-Based Models (EBMs), which also try to learn an energy landscape. But EBM training is notoriously unstable and difficult to scale.
  EqM’s key contribution is a carefully designed training objective that:
  Connects to energy landscapes like EBMs, and
  Retains the stability and scalability of flow-based/diffusion approaches.
  5. Flow Matching vs EqM: Opposite Gradient Directions
  In Flow Matching, the model learns a velocity field that transports noise to data:
  You usually mix a data sample (x) with noise (\epsilon), and define a target direction like: . i.e., a direction from noise to data.
  [
  v
  FM
  ∝
  x
  −
  ϵ
  ]
  In EqM, the target is ingeniously reversed:
  EqM defines its training target gradient as: [ g_{\text{EqM}} \propto \epsilon - x ]. i.e., a direction from data to noise, the exact opposite of Flow Matching.
  Why? Because EqM is designed from an energy perspective:
  Real data (x) should correspond to energy minima.
  The gradient of the energy function always points toward increasing energy (uphill).
  Near a real data point (x), the uphill direction must point away from (x) (toward higher-energy regions, e.g., noisy states like (\epsilon)).
  Therefore, EqM’s target gradient around (x) must be oriented from data toward noise.
  This flips the usual intuition:
  Flow Matching: “How do I flow from noise to data?”
  EqM: “What gradient field makes data points become valleys in an energy landscape, where the uphill direction is toward noise?”
  Learning this uphill field means that at sampling time we can just go downhill.
  6. Two Key Design Choices in EqM
  6.1 Mixing Data and Noise via a Random Interpolation Factor γ
  EqM constructs mixed states between data (x) and noise (\epsilon) using a random interpolation factor (\gamma):
  [
  z
  γ
  =
  γ
  x
  +
  (
  1
  −
  γ
  )
  ϵ
  ]
  Crucially:
  (\gamma) is not given as input to the model.
  The model (F) learns a single, unified, static gradient field that works for all mixing levels.
  In other words, (F) doesn’t learn a time-dependent field; it learns the geometry of the terrain itself, independent of any “time” or “schedule”.
  This is how EqM “escapes” dependence on dynamic variables and moves to a pure equilibrium view.
  6.2 Amplitude Controller (c(\gamma)): Enforcing Valleys at Data Points
  EqM uses a scalar amplitude controller (c(\gamma)), which modulates the magnitude of the target gradient:
  [
  target gradient
  ∝
  c
  (
  γ
  )
  ⋅
  (
  ϵ
  −
  x
  )
  ]
  Interpretation:
  (c(\gamma)) is like a light dimmer that controls how strong the uphill signal should be at different mixing levels.
  The critical constraint is: [ c(1) = 0 ]. When (\gamma = 1), the mixed point is exactly the data sample (x).
  This enforces that in the final learned energy landscape, all real data points (x) must satisfy:
  Gradient = 0 at (x).
  So data points become stable equilibrium points—true valleys of the energy function.
  A particularly effective choice is a truncated decay schedule:
  Far from the valley (heavily mixed with noise), the gradient magnitude is a relatively large constant, giving strong “downhill force” during sampling.
  Near the valley, the gradient magnitude rapidly decays to 0, allowing the optimization to settle smoothly instead of overshooting.
  Overall, the EqM training objective combines:
  Opposite target gradient direction ((\epsilon - x) instead of (x - \epsilon)).
  A model that does not depend on dynamic variables (same field for all (\gamma)).
  A precise zero-gradient constraint at data points via (c(1) = 0).
  The result: the model learns a static, well-behaved energy landscape over data space.
  7. Sampling with EqM: Optimization-Based Generation
  Once EqM has learned the energy landscape, how do we generate samples?
  Instead of solving an SDE/ODE, EqM uses optimization-based sampling:
  Start from an arbitrary point (e.g., random noise).
  Perform gradient descent on the learned energy function: Move downhill repeatedly until you reach a valley.
  This is classic optimization rather than numerical integration of a time-dependent flow. It brings several advantages:
  Robustness to Step Size: Gradient descent is more forgiving: you don’t need a very finely tuned schedule of step sizes to maintain sample quality.
  Use of Advanced Optimizers. You can plug in powerful optimizers like Nesterov Accelerated Gradient (NAG): Uses momentum to smooth updates. Takes a look-ahead step before computing the gradient, which can help (Converge faster / Jump over shallow local minima and find better valleys.)
  Adaptive Compute (Adaptive Number of Steps). You don’t need a fixed number of iterations. You can set a stopping condition, e.g., when gradient norm falls below a threshold. This means: Simple samples (easy terrain) converge quickly with fewer steps or Harder samples (rugged terrain) naturally consume more compute. Overall, the system automatically allocates compute per sample based on landscape difficulty, improving efficiency.
  So EqM reframes generation as energy optimization rather than trajectory simulation.
  8. Unique Properties of EqM
  EqM’s equilibrium structure yields several distinctive advantages.
  8.1 Partially Noised Image Denoising
  Traditional diffusion models are typically trained to follow a specific noise schedule. If you drop them into a partially noised intermediate state that doesn’t lie on that schedule (e.g., arbitrary noise injection), they can struggle or behave unpredictably.
  EqM, however, learns a global, static energy landscape:
  No matter where you start—pure noise, partially corrupted image, or in-between—
  The same gradient field tells you how to descend toward the nearest valley.
  This makes EqM naturally good at partial denoising and handling arbitrary noise levels without retraining or schedule engineering.
  8.2 Built-in Out-of-Distribution (OOD) Detection
  EqM has a natural out-of-distribution detection mechanism:
  In-distribution (ID) data lie in low-energy valleys.
  OOD data lie on the hillsides or high-energy regions.
  By simply inspecting the energy value estimated by the model, you can tell whether a sample is “normal” or “weird.”
  No extra classifier is required. This built-in OOD ability is especially valuable for safety and reliability: EqM can flag suspicious inputs by energy level alone.
  8.3 Simple Composition of Concepts
  EqM makes compositional generation conceptually straightforward.
  Suppose you have:
  An EqM model for pandas (with its own energy landscape and gradient field).
  An EqM model for bamboo.
  To generate “panda eating bamboo,” you can:
  At each step, add the gradient vectors from the two models: Compute gradients from the panda model and from the bamboo model. Sum them to get a combined “force”.
  Then perform gradient descent along the negative of that combined gradient.
  The system effectively descends a combined energy landscape representing the joint concept—without complex hacks or auxiliary tricks typical in diffusion pipelines.
  This vectorial composition of gradients is intuitive and elegant in the energy-based view.
  9. EqM as a Bridge Between Diffusion and Energy-Based Models
  In summary, EqM occupies a sweet spot between:
  The power and stability of diffusion / flow models, and
  The elegant equilibrium semantics of Energy-Based Models.
  It:
  Retains the soul of EBMs (learning a static energy landscape, equilibrium reasoning).
  Inherits the body of flow models (scalable, stable training and high-quality generation).
  10. Looking Ahead
  EqM also opens several exciting research directions:
  Incorporating second-order information (curvature of the energy landscape) to enable: More advanced optimizers. Faster convergence. Higher sample quality and better controllability.
  Viewing generation as direct manipulation of the energy landscape itself: Instead of only searching for minima, we could edit the learned energy function to: Shape where valleys lie. Sculpt constraints or preferences. Exert fine-grained control over the generative process.
  By reframing generative modeling as equilibrium energy optimization, EqM doesn’t just propose a new algorithm—it suggests a new way to understand and control how generative models create.
  11. Refernces
  [1] Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models. Runqian Wang, Yilun Du. https://arxiv.org/abs/2510.02300
  [2] ⛰️ 51. Equilibrium Matching: Paradigm Beyond Diffusion & Flow Models — GenAI Is Just Like Walking Down a Mountain https://www.linkedin.com/pulse/51-equilibrium-matching-diffusion-models-why-generative-ai-like-rqdgc

## 24. BEACONS: Bounded-Error, Algebraically-Composable Neural Solvers for Partial Differential Equations
- URL: https://arxiv.org/abs/2602.14853
- Cite Key: `@jonathan2026`
- Type: academic
- Authors: Gorard, Jonathan, Hakim, Ammar, Juno, James
- Date: 2026-02-16
- Description:
  The traditional limitations of neural networks in reliably generalizing beyond the convex hulls of their training data present a significant problem for computational physics, in which one often wishes to solve PDEs in regimes far beyond anything which can be experimentally or analytically validated. In this paper, we show how it is possible to circumvent these limitations by constructing formally-verified neural network solvers for PDEs, with rigorous convergence, stability, and conservation properties, whose correctness can therefore be guaranteed even in extrapolatory regimes. By using the method of characteristics to predict the analytical properties of PDE solutions a priori (even in regions arbitrarily far from the training domain), we show how it is possible to construct rigorous extrapolatory bounds on the worst-case L^inf errors of shallow neural network approximations. Then, by decomposing PDE solutions into compositions of simpler functions, we show how it is possible to compose these shallow neural networks together to form deep architectures, based on ideas from compositional deep learning, in which the large L^inf errors in the approximations have been suppressed. The resulting framework, called BEACONS (Bounded-Error, Algebraically-COmposable Neural Solvers), comprises both an automatic code-generator for the neural solvers themselves, as well as a bespoke automated theorem-proving system for producing machine-checkable certificates of correctness. We apply the framework to a variety of linear and non-linear PDEs, including the linear advection and inviscid Burgers' equations, as well as the full compressible Euler equations, in both 1D and 2D, and illustrate how BEACONS architectures are able to extrapolate solutions far beyond the training data in a reliable and bounded way. Various advantages of the approach over the classical PINN approach are discussed.

## 25. Persona Generators: Generating Diverse Synthetic Personas at Scale
- URL: https://arxiv.org/abs/2602.03545
- Cite Key: `@davide2026`
- Type: academic
- Authors: Paglieri, Davide, Cross, Logan, Cunningham, William A., Leibo, Joel Z., Vezhnevets, Alexander Sasha
- Date: 2026-02-03
- Description:
  Evaluating AI systems that interact with humans requires understanding their behavior across diverse user populations, but collecting representative human data is often expensive or infeasible, particularly for novel technologies or hypothetical future scenarios. Recent work in Generative Agent-Based Modeling has shown that large language models can simulate human-like synthetic personas with high fidelity, accurately reproducing the beliefs and behaviors of specific individuals. However, most approaches require detailed data about target populations and often prioritize density matching (replicating what is most probable) rather than support coverage (spanning what is possible), leaving long-tail behaviors underexplored. We introduce Persona Generators, functions that can produce diverse synthetic populations tailored to arbitrary contexts. We apply an iterative improvement loop based on AlphaEvolve, using large language models as mutation operators to refine our Persona Generator code over hundreds of iterations. The optimization process produces lightweight Persona Generators that can automatically expand small descriptions into populations of diverse synthetic personas that maximize coverage of opinions and preferences along relevant diversity axes. We demonstrate that evolved generators substantially outperform existing baselines across six diversity metrics on held-out contexts, producing populations that span rare trait combinations difficult to achieve in standard LLM outputs.

## 26. Performance Engineering of Software Systems | Electrical Engineering and Computer Science | MIT OpenCourseWare
- URL: https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/
- Cite Key: `@performance2026`
- Type: blog
- Description:
  6.172 is an 18-unit class that provides a hands-on, project-based introduction to building scalable and high-performance software systems. Topics include performance analysis, algorithmic techniques for high performance, instruction-level optimizations, caching optimizations, parallel programming, and building scalable systems. The course programming language is C.

## 27. Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens
- URL: https://arxiv.org/abs/2602.13517
- Cite Key: `@weilin2026`
- Type: academic
- Authors: Chen, Wei-Lin, Peng, Liqian, Tan, Tian, Zhao, Chao, Chen, Blake JianHang, Lin, Ziqian, Go, Alec, Meng, Yu
- Date: 2026/02/13
- Description:
  Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal &#34;overthinking,&#34; leading to performance degradation. In this work, we quantify inference-time effort by identifying deep-thinking tokens -- tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that deep-thinking ratio (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce Think@n, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes.
  
  
  1. 문제의식: reasoning 성능을 높이기 위해 CoT 길이를 늘리는 방식이 일반적이지만, 단순 토큰 길이는 실제 “생각 effort”를 제대로 반영하지 못한다는 점을 지적함. 길어도 얕은 계산일 수 있고, 짧아도 깊은 계산일 수 있음. 
  2. 핵심 가설: reasoning effort는 “얼마나 오래 출력했는가”가 아니라 “토큰이 내부 네트워크 깊이에서 얼마나 많이 수정·재계산되었는가”로 봐야 한다고 정의함. 즉 depth-wise computation을 측정해야 함. 
  3. 이를 위해 deep-thinking tokens 개념을 제안함. 토큰 생성 과정에서 예측이 초기 레이어에서 안정되는 경우는 얕은 사고, 깊은 레이어까지 계속 수정되는 경우는 깊은 사고로 간주함. 
  4. 구체적으로 deep-thinking ratio(DTR)를 정의함. 각 토큰이 네트워크 깊이 방향으로 얼마나 많은 업데이트를 거쳤는지를 측정해 inference-time thinking effort의 직접 지표로 사용함. 
  5. 기존 지표(출력 길이, token count, latency)는 surface-level proxy라 reasoning effort와 상관이 약하거나 왜곡될 수 있다고 분석함. 특히 긴 CoT가 항상 깊은 reasoning을 의미하지 않는다는 점을 실험적으로 보여주려 함. 
  6. 프레임워크는 토큰 생성 시 각 레이어에서의 logits 변화를 추적해 “깊이 방향 수정량”을 계산하고, 이를 토큰별 deep-thinking score로 집계하는 방식임. 
  7. 직관적으로는
  - early layer에서 prediction이 바로 고정되면 shallow thinking
  - 깊은 레이어까지 계속 재조정되면 deep thinking으로 분류됨. 
  8. 이 분석은 CoT reasoning 중 실제로 “생각이 필요한 토큰”과 단순 연결/형식 토큰을 분리할 수 있게 해줌. reasoning trace 전체를 동일하게 보지 않고 effort가 집중되는 구간을 찾는 목적임. 
  9. 논문은 reasoning 모델의 compute scaling이 단순 length scaling이 아니라 depth-aware scaling으로 이해되어야 한다고 주장함. 즉 test-time compute를 “얼마나 길게”가 아니라 “얼마나 깊게” 쓰느냐의 문제로 재정의함. 
  10. 이 지표를 사용하면 어떤 reasoning 단계가 실제 계산 부담이 큰지, 모델이 언제 진짜로 고민하는지, 길이만 늘린 CoT와 깊은 reasoning CoT의 차이를 분리해서 분석할 수 있다고 설명함. 
  11. 더 나아가 deep-thinking token 분석은 inference 최적화에도 연결됨. 얕은 토큰은 계산을 줄이고, 깊은 토큰에는 compute를 더 할당하는 adaptive inference 설계로 이어질 수 있다고 제안함. 
  12. 전체 메시지는 reasoning scaling을 length 중심에서 depth 중심으로 재정의해야 한다는 것. CoT 길이 증가만으로는 reasoning effort를 설명할 수 없으며, 토큰 내부 계산 깊이를 측정하는 새로운 분석 축이 필요하다는 주장임.

## 28. Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens
- URL: https://arxiv.org/abs/2602.13517
- Cite Key: `@weilin2026a`
- Type: academic
- Authors: Chen, Wei-Lin, Peng, Liqian, Tan, Tian, Zhao, Chao, Chen, Blake JianHang, Lin, Ziqian, Go, Alec, Meng, Yu
- Date: 2026/02/13
- Description:
  Large language models (LLMs) have demonstrated impressive reasoning capabilities by scaling test-time compute via long Chain-of-Thought (CoT). However, recent findings suggest that raw token counts are unreliable proxies for reasoning quality: increased generation length does not consistently correlate with accuracy and may instead signal &#34;overthinking,&#34; leading to performance degradation. In this work, we quantify inference-time effort by identifying deep-thinking tokens -- tokens where internal predictions undergo significant revisions in deeper model layers prior to convergence. Across four challenging mathematical and scientific benchmarks (AIME 24/25, HMMT 25, and GPQA-diamond) and a diverse set of reasoning-focused models (GPT-OSS, DeepSeek-R1, and Qwen3), we show that deep-thinking ratio (the proportion of deep-thinking tokens in a generated sequence) exhibits a robust and consistently positive correlation with accuracy, substantially outperforming both length-based and confidence-based baselines. Leveraging this insight, we introduce Think@n, a test-time scaling strategy that prioritizes samples with high deep-thinking ratios. We demonstrate that Think@n matches or exceeds standard self-consistency performance while significantly reducing inference costs by enabling the early rejection of unpromising generations based on short prefixes.
