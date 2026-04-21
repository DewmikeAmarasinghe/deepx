Liquid Transformer models: concise research brief (through 2025)

Executive summary
- Liquid Transformer refers to transformer-style architectures that incorporate liquid-time-constant dynamics or continuous-time dynamical systems within the model components (often via LTCs or state-space formulations). This yields adaptive time scales and improved expressivity, particularly for time-series and irregular data.
- Core lines of work include Liquid Time-Constant Networks (LTCs) as continuous-time RNNs with input-dependent time constants [Hasani et al., 2020; 2021; AAAI 2021 DOI:10.1609/aaai.v35i9.16936], Liquid Structural State-Space Models (Liquid S4) by Gu et al., 2022 [arXiv:2209.12951], and transformer hybrids that fuse liquid/LTC dynamics with attention-based architectures (e.g., Liquid-Neural hybrids; Liquid S4; 2025 energy-forecast work). A practical implementation: the Liquid library converts vanilla transformers into Liquid transformers via dynamic time-constants [Kyegomez, GitHub].
- Applications center on time-series forecasting (including energy/building forecasting), irregular time-series modeling, control, and sequence modeling with long-range dependencies.

1) What is a Liquid Transformer?
- Concept: models whose internal computation uses time-varying, input- and state-dependent time constants, implemented as continuous-time recurrence or as a linearized form with LTCs embedded inside transformer kernels (Liquid S4).
- Rationale: standard transformers use fixed, discrete-time updates; Liquid variants aim to capture dynamic causal structure and flexible memory horizons, especially for non-uniform data.
- Representative families:
  - Liquid Time-Constant Networks (LTCs): continuous-time RNNs with dynamic time constants modulated by inputs; proven to be universal approximators and expressive in time-series tasks. Key papers: Hasani et al. 2018 NeurIPS; 2020/2021 arXiv; AAAI 2021 (DOI 10.1609/aaai.v35i9.16936) [Hasani et al.].
  - Liquid Structural State-Space Models (Liquid S4): linearized LTCs inside SSMs to build a liquid-state kernel with S4 structure; new state-space transformer variants capable of long-range modeling; arXiv:2209.12951.
  - Hybrid Transformer-LNN (Liquid Neural Network) architectures: transformers augmented with Liquid NNs for reservoirs or nonlinearity; 2025 work on energy forecasting using a liquid reservoir with encodings (DOI 10.1016/j.egyai.2025.100489).
  - Continuous-time transformers for irregular time series (ContiFormer): CT-MHA and neural ODE-like dynamics embedded in transformer attention for irregular sampling; related literature on continuous-time transformers.
- Notable resources: Liquid library converts vanilla transformers to Liquid transformers; Liquid S4 repo demonstrates a practical Liquid-SSM kernel; ContiFormer demonstrates continuous-time transformer for irregular series.

2) Core ideas and architectural motifs
- Time-constant modulation: Each unit’s effective time constant varies with the hidden state and input, enabling adaptive memory windows and dynamic causality (LTCs) [Hasani et al., 2020; 2021; AAAI 2021 DOI:10.1609/aaai.v35i9.16936].
- ODE-based dynamics vs discrete layers: LTCs view hidden states as solutions to stiff ODEs; numerical solvers (or closed-form CfC variants) yield solver-free variants and different compute semantics. See LTC papers.
- Structural/state-space augmentation: Liquid S4 uses a linearized LTC kernel within a structured state-space model, enabling efficient long-range modeling with S4 primitives [Gu et al., 2022; arXiv:2209.12951].
- Hybridization with transformers: Merging Liquid/NCT-like reservoirs or dynamic kernels with attention to combine responsive temporal dynamics with sequence modeling strengths of Transformer attention.

3) Representative papers, key ideas, and results
- Liquid Time-Constant Networks (LTCs) – R. Hasani, M. Lechner, A. Amini, D. Rus, R. Grosu. NeurIPS 2018 (LTC concept) and AAAI 2021 (LTCs, DOI:10.1609/aaai.v35i9.16936). LTCs define a time-continuous RNN whose time constants are modulated by the input/state; show expressivity gains and improved time-series prediction vs classic RNNs [Hasani et al., 2020; 2021]. See arXiv 2006.04439 for foundations.
- Liquid Structural State-Space Models (Liquid S4) – Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré et al. arXiv:2209.12951 (2022). Build a convolutional kernel by linearizing LTCs within a structural SSM (S4) framework; demonstrate improved performance across modalities relative to standard RNN/CNN baselines and other Transformers; provides a repository at liquid-s4. See OpenReview and arXiv sources.
- Liquid Neural Networks and Hybrid Transformers – papers and implementations exploring Transformer-LNN hybrids for time-series and energy forecasting; 2025 work on building energy forecasting reporting improvements over base Transformer, LSTM, and ANN baselines; DOI 10.1016/j.egyai.2025.100489. See ADS listing and Semantic Scholar.
- ContiFormer (Continuous-Time Transformer for Irregular Time Series) – Chen et al.; treats attention in continuous time with Neural ODE-like dynamics; demonstrates strong performance on irregular time series; illustrates how continuous-time dynamics can be integrated with Transformer architectures as an alternative to discrete-time recurrence. OpenReview/NeurIPS pages provide details.
- Liquid Transformer library – Kyegomez: Liquid library converts vanilla Transformers into Liquid Transformers; GitHub resource for practical adoption [Liquid, 2024].

4) Data, training regimes, and practicalities
- LTCs require solving continuous-time dynamics; training can use standard backpropagation through time with ODE solvers or closed-form variants (CfC). These methods can yield solver-free speedups but may introduce numerical considerations and stiffness.
- Liquid S4 leverages structured SSMs and HiPPO/diagonal operators to memorize histories; training benefits from efficient linear-time kernels.
- Hybrid Transformer-LNN approaches often rely on reservoir-like components to capture nonlinear temporal dynamics and then feed into or alongside Transformer attention. Training typically follows standard supervised learning with cross-entropy or regression losses depending on the task.
- Irregular time series handling (ContiFormer) is a core motivation for continuous-time Transformers; the modeling choice is beneficial when data are irregularly sampled or there are missing timestamps.

5) Strengths, limitations, and practical considerations
- Strengths:
  - Enhanced temporal flexibility: adaptive memory horizons via liquid time constants; better modeling of nonstationary time-series.
  - Improved out-of-distribution generalization in some LTC-like models; linear Liquid SSMs (Liquid-S4) claim robust modeling across modalities.
  - Ability to handle irregular sampling and continuous-time dynamics in transformer-like architectures (ContiFormer) or via LTC-S4 hybrids.
- Limitations:
  - Increased computational overhead due to ODE solvers or reservoir dynamics and more complex training. Reservoir-based variants can be more data-hungry to train robustly.
  - Fewer standardized benchmarks; results are often domain-specific (energy forecasting, control, etc.).
  - Interpretability of dynamic time constants and their relation to attention remains an area of investigation.
- Data requirements and regimes:
  - LTC-based models can be data-efficient for certain dynamical systems but may require careful hyperparameterization of time-constants and solver settings.
  - In irregular time-series tasks, continuous-time transformers (ContiFormer) can exploit timestamp information directly.

6) Open questions and research directions
- How do liquid-time-constant mechanisms interact with self-attention to yield robust representations across domains? Are there principled ways to integrate dynamic time constants with attention more deeply? (Watch for work on Liquid-S4 and related: 2022-2025.)
- What are the most scalable training strategies (solver-free variants, CfC, truncated backprop) for large-scale Transformer-Liquid hybrids? 
- How do we establish standardized benchmarks for Liquid Transformers to enable fair comparisons against standard Transformers and other dynamic architectures?
- Can we unify LTC-based continuous-time dynamics with discrete-time Transformer blocks in a single modular architecture for broad applicability? 

7) Quick comparison (compact table)

| Model family | Core idea | Typical architecture motif | Strengths | Limitations | Typical tasks / data | Key papers / sources |
|---|---|---|---|---|---|---|
| Liquid Time-Constant Networks (LTCs) | Dynamic, input/state-dependent time constants in continuous-time RNNs | ODE-based hidden state with time-varying tau; solver-based updates | High expressivity; suited for time-series; continuous-time causality | Computationally heavier; solver choices matter | Time-series forecasting; dynamical systems | Hasani et al., 2018 NeurIPS; 2020 arXiv; 2021 AAAI DOI:10.1609/aaai.v35i9.16936; arXiv:2006.04439 |
| Liquid Structural State-Space Models (Liquid S4) | Linearized LTC kernel inside a Structural State-Space Model (S4) for scalable long-range modeling | Structured SSM with HiPPO diagonal operators; convolutional kernels | Robust long-range modeling; good cross-modal performance | Complexity of SSM kernels; still integration with Transformer blocks needed | Text, audio, time-series, cross-modal | Gu et al., 2022 arXiv:2209.12951; OpenReview; Liquid-S4 repo |
| Liquid-Neural Hybrid Transformers | Hybrid Transformer with Liquid Neural Networks for reservoirs/encodings | Transformer + reservoir-like LTC components | Improved encoding of temporal structure; good on forecasting | Training complexity; interpretability | Building energy forecasting; time-series | 2025 EneAI; DOI 10.1016/j.egyai.2025.100489; ADS/semantic scholar |
| Continuous-Time Transformers (ContiFormer) | Continuous-time attention for irregular time series | CT-MHA with Neural ODE-like dynamics | Handles irregularity; strong performance on irregular series | Still early in evaluation; complexity | Irregular time series | Chen et al., openreview/seqml contiformer (2024-2025) |
| Liquid Transformer library | Practical catalyst – turning vanilla Transformers into Liquid transformers | Software/library | Enables quick experimentation; fosters adoption | Abstract concept of liquidity remains; standard benchmarks vary | General transformer tasks; time-series? | Kyegomez, GitHub Liquid |

8) Short bibliography and sources (DOIs/URLs)
- Hasani, R., Lechner, M., Amini, A., Rus, D., Grosu, R. Liquid Time-Constant Networks. NeurIPS 2018; AAAI 2021 (DOI 10.1609/aaai.v35i9.16936).
- Hasani, R. et al. Liquid Time-Constant Networks. arXiv:2006.04439 (foundation).
- Gu, Albert; Gupta, Ankit; Goel, Karan; Ré, Christopher; et al. Liquid Structural State-Space Models. arXiv:2209.12951 (DOI:10.48550/arXiv.2209.12951).
- Liquid S4 repository: https://github.com/raminmh/liquid-s4; arXiv OpenReview: Liquid Structural State-Space Models.
- Kyegomez, Liquid library: https://github.com/kyegomez/Liquid.
- Hybrid transformer model with liquid neural networks and learnable encodings for buildings’ energy forecasting. 2025 EneAI; DOI:10.1016/j.egyai.2025.100489.
- ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling. OpenReview: https://openreview.net/forum?id=YJDz4F2AZu; CT-MHA; NeurIPS 2024/2025 outputs.
- Liquid Neural Networks: A Novel Hybrid Transformer (Kaggle notebook) – not peer-reviewed; illustrative.

Sources
- Hasani, R. et al. Liquid Time-Constant Networks. arXiv:2006.04439; NeurIPS 2018; AAAI 2021 DOI 10.1609/aaai.v35i9.16936.
- Hasani, R. et al. Liquid Time-Constant Networks – Simons Institute Slides and more. 
- Gu, A.; Gupta, A.; Goel, K.; Ré, C. Liquid Structural State-Space Models. arXiv:2209.12951; DOI:10.48550/arXiv.2209.12951.
- Liquid S4 – GitHub repository: https://github.com/raminmh/liquid-s4; arXiv:2209.12951; OpenReview page.
- Kyegomez, L Liquid – GitHub: https://github.com/kyegomez/Liquid.
- Hybrid Transformer with Liquid Neural Networks for Building Energy Forecasting. 2025 EneAI DOI:10.1016/j.egyai.2025.100489.
- ContiFormer – Continuous-Time Transformer for Irregular Time Series Modeling. OpenReview: https://openreview.net/forum?id=YJDz4F2AZu; related work on CT-MHA.
