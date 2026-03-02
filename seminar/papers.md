
#  Course description Topics, papers, and supervisors

## P1: Matching and Multi-Armed Bandits  

- **Supervisor:** Andreas Athanasopoulos
- **mail**: andreas.athanasopoulos@unine.ch 

- **General description:**  
  This topic aims to explore learning algorithms in the context of matching markets. Students will specifically study matching problems, where a central platform matches agents from two distinct sets based on their preferences. For example, consider the assignment of students to universities, where a central mechanism decides the allocation.

  The papers specifically focus on learning fair (or trustworthy) outcomes of agents, where the notion of fairness is defined with respect to agents’ preferences. One important solution concept is **stable matching**, a matching where we cannot find a pair of agents who both prefer each other over their current partners. Stable matchings respect agents’ preferences and connect to the concept of equilibrium in game theory, since no pair of agents has an incentive to deviate from the market.

  This topic represents a paradigm of multi-agent learning, where multiple agents participate in a market and compete for matches. Students will study matching problems, which are central to economics (with the Nobel Prize awarded to Gale and Shapley), and their intersection with learning algorithms from **Multi-Armed Bandits (MAB)**, a core framework in reinforcement learning.

- 👨‍🏫: ⭐⭐  
  Some theoretical knowledge, particularly in multi-armed bandits, is required to fully understand the papers. We provide all the necessary material to ensure that students can follow along.
- 🖥️: ⭐  
  The experiments are simulation-based and can be conducted on a standard laptop.
  
- **Paper 1:**  **Competing Bandits in Matching Markets**  
  The paper studies learning algorithms in settings where agents are initially unaware of their preferences. More specifically, it considers a scenario in which agents learn preferences through repeated interactions with each other, formulating the problem within the Multi-Armed Bandit framework.
  - **Link:** [https://arxiv.org/abs/1906.05363](https://arxiv.org/abs/1906.05363)  
  - **How to reproduce:**  
    To reproduce the paper, students should first implement classic Multi-Armed Bandit algorithms, specifically Explore-Then-Commit (ETC) and Upper Confidence Bound (UCB). Then, they should adapt these base algorithms to matching markets and replicate the experiments presented in the paper.  
  - **How to extend:**  
    1. Provide experiments on randomly generated examples and plot the regret of the proposed algorithm.
    2. Provide experiments where both sides of the market are initially unaware of their preferences.  
    3. An interesting extension is to perform experiments that empirically quantify the algorithm’s performance in the case of non-truthful agents (Section 3.3 of the paper). While the paper studies this theoretically, many questions remain regarding the extent to which an agent can exploit rewards. Initial experiments would be valuable for students and could potentially lead to a thesis. Additionally, students can research related literature to complement their work.  

- **Paper 2:** **Learning Equilibria in Matching Markets from Bandit Feedback**     
The paper extends the previous work by considering an alternative regret measures, and models that include transfers.

  - **Link:** [[https://arxiv.org/abs/2506.03802](https://arxiv.org/abs/2506.03802) ](https://arxiv.org/abs/2108.08843) 
  - **How to reproduce:**  
    To reproduce the paper, students should first implement a Upper Confidence Bound (UCB) algorithm. Then, they can extend the UCB algorithm according to the paper and reproduce the experiments.  
  - **How to extend:**  
    1. Empirically study alternative regret measures, such as player-optimal and player-pessimal regret, independent of the matching instability studied in the paper.


## P2 : Zero-Sum Matrix Games with Bandit Feedback

- **Supervisor:** Elif
- **Email**: elif.yilmaz@unine.ch

- **General description:**  
  This topic covers multi-armed bandit (MAB) algorithms and zero-sum games. Students will investigate standard MAB algorithms and their performances in a zero-sum game setting, understanding how the setting dynamics influence decision-making and regret performance. Through simulations and experiments, students will gain experience in evaluating algorithm performance under different environments and constraints. Overall, the topic emphasizes both the theoretical foundations and practical implementation challenges of MAB algorithms.

- 👨‍🏫: ⭐⭐
- 🖥️: ⭐

- **Paper**
  - **Link:** [On the Limitations and Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games Under Noisy Feedback](https://arxiv.org/pdf/2306.13233)
  - **How to reproduce:**
    Students should begin by implementing the standard MAB algorithms and then apply them in a zero-sum game setting. They should reproduce the experimental setup by simulating environments (Section 3 and 4). Then, they should compare the performance of the algorithms by generating regret curves and analyzing whether the empirical results align with the theoretical regret bounds.
  - **How to extend:**
    1. Implement different MAB algorithms (includes elimination based approaches) and apply them to a zero-sum game setting.
    2. In the paper, they are interested in only controlling the row player. Assume controlling over both players, apply the proposed algorithms and compare the results.
    3. Assume there exists a unique pure strategy Nash equilibria in the game and compare the PSNE identification capability of different MAB algorithms.


## P3: Matching and Combinatorial Multi-Armed Bandits
- **Supervisor** : Hortence Nana

- **Email**: hortence.yiepnou@unine.ch
  
- **PAPER** : Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits

  - 👨‍🏫: ⭐
  - 🖥️: ⭐⭐⭐

  - **Brief description**: The paper studies an online learning setting called stochastic combinatorial semi-bandits. In this setting, a learning agent repeatedly selects subsets of items (subject to combinatorial constraints), observes their individual stochastic rewards, and aims to maximize cumulative reward over time. The key contribution of the paper is an analysis of the CombUCB1 algorithm, a UCB-like algorithm for this setting, and the derivation of tight regret bounds that are both gap-dependent and gap-free.
 
  - **Link** : [https://proceedings.mlr.press/v38/kveton15.pdf]
  
  - **How to reproduce**: Student should implement the CombUCB1 . concretely, carefully read and understand the problem definition (Section 2), the CombUCB1 algorithm and its initialization, and the implementation part. Implement CombUCB1 as described in Algorithm 1 and make some experiments.
    
  - **How to extend**: As extention, students will apply that setting to many-to-one matching and derive non linear rewards.


## P6: Model-Based Exploration in MDPs

- **Supervisor:** Victor Villin
- **mail**: victor.villin@unine.ch

- **General description:** 
This project focuses on exploration in Reinforcement Learning (RL). When the MDP is not known to the learner, the latter faces an exploration–exploitation dilemma:
    - **Exploitation**: Use current knowledge to take actions maximizing reward.
    - **Exploration**: Gather information about uncertain states and actions.

  For example, the famous Q-learning algorithm typically employs $\epsilon$-greedy exploration. While simple, its exploration takes up to exponential time to converge to the optimal solution.
  This project studies model-based exploration algorithms that provide faster theoretical guarantees on the optimality of the learned policy. 
  In particular, we investigate model-based RL algorithms that use optimism in the face of uncertainty to ensure theoretically grounded, efficient exploration.

- 👨‍🏫: ⭐⭐
- 🖥️: ⭐⭐


- **Project content**
    - I. **Reading and Implementation**: 
Students will have to reproduce the results from the [Model-based Interval Estimation for RL paper](https://pdf.sciencedirectassets.com/272574/1-s2.0-S0022000008X00078/1-s2.0-S0022000008000767/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCJSSpQ1xsG5aEikHyy%2F1fPuzGlbYpqeUu59PraxvU%2FzgIgRHq4UZbx1Ey%2FhcmxjRabSZV%2F4b5HJ4MXI3DWeyG7%2FC4qvAUIg%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDDcgevWhUfPGzli6jSqQBcrXU6wHdBQzi31yj%2F8uyy45TZwKklnq6oQyfBUoOBGM%2F7dwUagXo%2F1OPaPsL0B3deT4LcTntKs%2FtdPESM0QirZBFHKuKu84korj8suosfxlas4JMnCEjjLPt3Ve2vbMWo%2BHXMO4vdtXiGByIpgnxgNCTWJw0E8FYhRY%2BwJxC%2FfSfRgH0Ayy9BrU1c5dwPvbZCIV3tU44zTWUVxLPu4iaYvSffcqY%2B8nENrZYTTdmibEasU3TqsF6KRseud4yI%2FQlYyMJqrGHl%2B6d9a3BaG%2B0oQMi1aqeiu2iGOtSFldCdnYuUv44PSF9o%2BHdMliJ3wwZKnQVqLB9VcbMwXUZI0SnEz1m9k%2FLLH45GbVpjdiwaw22jC7xQ9VI4vSQGx2fWLVmPKy%2BZjJJTNiC%2FkWHaeUaEcCoCF0HcHQvC%2FJ24mgZ0zygzg%2Fu5SFCDSLwrVlaEoByGYdrX5gnXDmIqMPxk3sjghwwHreaK5oEiUI7QCuETot5DqFacdf8UNBYD0aAjzax4JGqulV8gt0lw9yggZJzPS0p5hbdafc7eQ17tEB8Y4bH3F82eVnJhHcgfUTAsbQiDmv4KyHZh75rY4WH2tBDtjssLXwfJN07AByB9iSKdkHg4Fk%2BNqo9pkUTiEUT3jVhP4dqCloIWFiS%2B9k2QBFklf%2FNWH6vYgF1YD3LSSd8ESxcZBXqH6i2dW%2FPhgQODBlbTk3VWmk%2Fs5uHW0acoAV4%2BXUWzOKGWNrIzKvaz4CxcBLXU5cVwItXQhAnGfVtXT3TUI3tpN63t5jaZXihC9e3LiC5tzBTv84gR%2FTMO6h4CumumAKPjAeksB485jD9EeM5%2FzmvKEsrG6KbLMAt1fOnXGUV1%2BEelVcTW9VkQRccesxMIS0lc0GOrEBuh5%2FcdnrPDsy1CzP1p9hFO%2B6CMvqAxDpIfE6i69gJCBvOmf8ppyf1hk8b84cn48TfEQqKVKJwqgmNj4stAFDY0giYFKaVelF%2BFJgFZCPi7uEsrNg9cl29JYbLSxxIlTDSM9b07O6Qtbt6%2FcURYlv8HWpsr2PNMLKftJuD4iT6NmUZMB9vsXnlLIRDrRk2m8jIj6QSE1nxmGjWTE%2BTo2F%2FLgmFgYcWXXybQQ5CKUdM8S3&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260302T110556Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTY2TYMBL4S%2F20260302%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=21d205739b127adcf672a7f5aebd8048b257c80f3e09d0d53b1e99dcf56b22dd&hash=d7c34f2ad6b41ad1a23487157b32c2d8b8b0fe2be0ba87879b3ea1be0e2ec1d2&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0022000008000767&tid=spdf-9a324b52-9dd4-4829-9545-cd0888f885c9&sid=6751f14093211949f59b6846edd048eb1c89gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=060e5d0a550907050455&rr=9d5fede19b2c3b5e&cc=ch). Specifically, they will have to:
        - Read carefully and understand the theoretical analysis.
        - Read and understand about the other studied baselines (E$^3$, R-max).
        - Implement the baselines, MBIE and its variant (MBIE-EB).
        - Reproduce the Benchmarks from the paper (RiverSwim and SixArms MDPs).
    - II. **Extensions**:
Next, students will be able to extend their project in at least one out of many possible directions
        - Novel exploration schemes.
        - Scaling to larger problems.
        - Compare with additional baselines.
        - Benchmark on harder MDPs.
        - Model-free exploration.
        - etc.
  
**NB**: students will receive a pdf with a more formal description of the paper and tasks, as well as some link that will help to understand some concept.
