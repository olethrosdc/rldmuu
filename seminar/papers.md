
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



## P2: Matching and Combinatorial Multi-Armed Bandits
- **Supervisor** : Hortence Nana

- **Email**: hortence.yiepnou@unine.ch
  
- **PAPER** : Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits

  - 👨‍🏫: ⭐
  - 🖥️: ⭐⭐⭐

  - **Brief description**: The paper studies an online learning setting called stochastic combinatorial semi-bandits. In this setting, a learning agent repeatedly selects subsets of items (subject to combinatorial constraints), observes their individual stochastic rewards, and aims to maximize cumulative reward over time. The key contribution of the paper is an analysis of the CombUCB1 algorithm, a UCB-like algorithm for this setting, and the derivation of tight regret bounds that are both gap-dependent and gap-free.
 
  - **Link** : [https://proceedings.mlr.press/v38/kveton15.pdf]
  
  - **How to reproduce**: Student should implement the CombUCB1 . concretely, carefully read and understand the problem definition (Section 2), the CombUCB1 algorithm and its initialization, and the implementation part. Implement CombUCB1 as described in Algorithm 1 and make some experiments.
    
  - **How to extend**: As extention, students will apply that setting to many-to-one matching and derive non linear rewards.
   
  
**NB**: students will receive a pdf with a more formal description of the paper and tasks, as well as some link that will help to understand some concept.




## P3 : RL (Zero-Sum Matrix Games with Bandit Feedback)

- **Supervisor:** Elif
- **Email**: elif.yilmaz@unine.ch

- **General description:**  
  This topic covers multi-armed bandit (MAB) algorithms with a focus on incorporating additional considerations such as trustworthiness and differential privacy. Students will investigate standard MAB algorithms and their performances in a zero-sum game setting, understanding how the setting dynamics influence decision-making and regret performance. Through simulations and experiments, students will gain experience in evaluating algorithm performance under different environments and constraints. Overall, the topic emphasizes both the theoretical foundations and practical implementation challenges of MAB algorithms.

- 👨‍🏫: ⭐⭐
- 🖥️: ⭐

- **Paper**
  - **Link:** [On the Limitations and Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games Under Noisy Feedback](https://arxiv.org/pdf/2306.13233)
  - **How to reproduce:**
    Students should begin by implementing the standard MAB algorithms and then apply them in a zero-sum game setting. They should reproduce the experimental setup by simulating environments (Section 3 and 4). Then, they should compare the performance of the algorithms by generating regret curves and analyzing whether the empirical results align with the theoretical regret bounds.
  - **How to extend:**

