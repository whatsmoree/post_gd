## 0. 主要工作

### 0.0 主要思路

想着结合 Fairwalk 和 CNARW，然后拓展到动态网络应用（基于 dynnode2vec）。这两周看到了上个月发的 DeepHub（基于 dynnode2vec）可以拿来比较，大概率效果比 dynamic fair-CNARW 要好，然后发现 DeepHub 可以拓展。故有了这份方案稿。

### 0.1 方法论

1. 提出 Fair-CNARW（1.3）
2. 拓展 Dynamic Fair-CNARW（2.2）
3. 提出 DeepHub+（2.4）& DeepHub-Eigen（2.5）

### 0.2 结果（拟）

1. 社区检测
2. 公平性评估
3. 计算效率
4. 方法比较（3.3）

## 1. 静态网络

### 1.1 CNARW

#### 论文链接
[Walking with Perception: Efficient Random Walk Sampling via Common Neighbor Awareness](https://www.cse.cuhk.edu.hk/~cslui/PUBLICATION/ICDE_2019_B.pdf#:~:text=common%20neighbor%20aware%20random%20walk%20framework%20called%20CNARW%2C,next-hop%20candidate%20nodes%20to%20speed%20up%20the%20convergence.)

#### 核心思想

CNARW 提出了改进传统随机游走（Random Walk）采样的方法，以解决其**收敛速度慢、查询成本高**的问题。核心策略是：在选择下一跳节点时，不再均匀随机地选择邻居，而是结合**当前节点与候选节点之间的共同邻居数量**进行加权，从而提高收敛效率，减少查询次数。

CNARW 认为，如果一个候选节点具有更高的度但与当前节点更少的共同邻居，那么它更有可能通往未访问的节点，从而提供更高的探索机会。因此，CNARW 会以更高的概率选择这样的节点作为下一跳。

#### 算法流程与核心公式

CNARW通过考虑共同邻居来改进随机游走的转移概率。其核心公式为：

**转移概率计算：**

$$P(v_j | v_i) = \frac{1 - \frac{C_{ij}}{\min(\deg(v_i), \deg(v_j))}}{Z}$$

其中：
- $C_{ij}$：节点$v_i$和$v_j$的共同邻居数量
- $\deg(v_i)$：节点$v_i$的度数
- $Z$：归一化因子

**算法流程：**
1. 对于当前节点$v_i$，获取所有邻居节点集合$N(v_i)$
2. 对每个邻居$v_j \in N(v_i)$，计算共同邻居数量$C_{ij}$
3. 根据公式计算转移概率$P(v_j | v_i)$
4. 基于概率分布选择下一个节点
5. 重复步骤1-4，直到游走长度达到预设值

### 1.2 FairWalk

#### 论文链接
[Fairwalk: Towards Fair Graph Embedding](https://www.ijcai.org/proceedings/2019/0456.pdf)

#### 核心思想

FairWalk 指出主流图嵌入算法（如 Node2vec）存在对少数群体的偏见，尤其在**好友推荐等社交网络任务中会加剧原有的不平等**。为此，提出了一个公平性感知的节点嵌入方法，旨在解决传统随机游走方法在处理敏感属性（如性别、种族等）时的偏见问题。核心思想是通过调整随机游走的路径，使得不同群体的节点在嵌入空间中得到公平的表示。

#### 算法流程与核心公式

FairWalk通过群体感知的随机游走来确保公平性。其核心公式为：

**群体平衡的转移概率：**

$$P(v_j | v_i) = \frac{1}{|Z_{v_i}|} \cdot \frac{1}{|w_z (v_i)|} $$
<!-- $$P(v_j | v_i) \propto \exp(-\lambda \cdot d(G_i, G_j))$$ -->

其中：
- $Z_{v_i}$：包含有节点$v_i$邻居的所有群体的集合
- $w_z (v_i)$：属于群体$z$的当前节点$v_i$的邻居集合
<!-- - $\lambda$：公平性控制参数 -->

**算法流程：**
1. 首先为每个节点分配群体标签
2. 对于当前节点$v_i$，识别所有邻居的群体
3. 等概率选择群体
4. 群体内均匀随机选择节点
<!-- 3. 根据群体距离调整转移概率
4. 执行偏向于公平性的随机游走
5. 使用游走序列训练Skip-gram模型 -->

### 1.3 结合 CNARW 和 FairWalk: Fair-CNARW

#### 核心思想

提出的结合方法可以在 CNARW 的高效采样基础上，加入 FairWalk 的公平性考量。具体来说，可以设计一个新的随机游走权重函数，既考虑共同邻居的影响，又确保不同群体的节点在嵌入空间中得到公平的表示。

#### 算法流程与核心公式

Fair-CNARW结合了CNARW的效率和FairWalk的公平性。其核心是统一权重函数：

**统一权重函数：**

$$w_{uv} = \alpha \cdot \text{CNARW\_weight}(u,v) + (1-\alpha) \cdot \text{Fairness\_weight}(v)$$

其中：
- **CNARW权重**：$\text{CNARW\_weight}(u,v) = 1 - \frac{C_{uv}}{\min(\deg(u), \deg(v))}$
- **公平性权重**：$\text{Fairness\_weight}(v)$
 <!-- = \frac{1}{|G_g|}$，其中$v \in G_g$ -->
- **$\alpha$**：效率-公平性权衡参数 $(0\leq\alpha\leq1)$

**算法流程：**
1. 预处理：计算所有节点对的共同邻居数量
2. 为每个节点分配群体标签，统计群体大小
3. 对于当前节点u的每个邻居v：
   - 计算CNARW权重（基于共同邻居）
   - 计算公平性权重（基于邻居群体）
   - 结合两个权重得到统一权重
4. 根据统一权重进行随机游走
<!-- 5. 使用游走序列训练Skip-gram模型 -->

**动态$\alpha$策略：**
- **效率优先模式**：$\alpha \to 1$，重视采样效率
- **公平性优先模式**：$\alpha \to 0$，重视群体平衡
- **自适应模式**：根据图特性动态调整$\alpha$值


### 2. 动态网络

#### 2.1 dynnode2vec

#### 论文链接
[dynnode2vec: Scalable Dynamic Network Embedding](https://arxiv.org/abs/1812.02356)

#### 核心思想

dynnode2vec 提出了一个增量式的动态图嵌入方法，旨在解决传统静态图嵌入方法在处理动态图时的局限性。它通过对节点嵌入进行增量更新，避免了每次时间步都需要重新计算所有节点的嵌入，从而提高了效率。

#### 算法流程与核心公式

dynnode2vec通过增量更新策略处理动态图。核心是演化节点识别：

**演化节点识别：**

$$\Delta V_t = V_{\text{add}} \cup \{v_i \in V_t | \exists e_i = (v_i, v_j) \in (E_{\text{add}} \cup E_{\text{del}})\}$$

其中：
- $V_{\text{add}}$：在时间$t$新增的节点
- $E_{\text{add}}$：新增的边集合
- $E_{\text{del}}$：被删除的边集合

**算法流程：**
1. **初始化 (t=1)**：
   - 对第一个图快照G₁运行标准node2vec
   - 生成初始嵌入$Z_1$和Skip-gram模型$\text{Skip}_1$

2. **增量循环 (t=2 to T)**：
   - 识别演化节点：计算$\Delta V_t$
   - 生成演化游走：仅为$\Delta V_t$中的节点生成随机游走
   - 动态训练：使用$Z_{t-1}$初始化，用新游走更新模型

<!-- **时间复杂度优化：**
- **传统方法**：$O(|V| \times n_{\text{walks}} \times \text{walk\_length})$
- **dynnode2vec**：$O(|\Delta V_t| \times n_{\text{walks}} \times \text{walk\_length})$
- **加速比**：$|V| / |\Delta V_t|$ -->

### 2.2 dynamic Fair-CNARW

#### 核心思想

dynamic Fair-CNARW 结合了 dynnode2vec 的增量更新策略和 Fair-CNARW 的公平性机制，旨在实现动态图的高效且公平的嵌入。通过在动态图中仅更新变化的节点和边，算法能够保持时间一致性，同时确保不同群体的节点得到公平的表示。

#### 算法流程与核心公式

Dynamic Fair-CNARW结合了增量更新和公平性权重。其核心算法流程为：

**统一权重函数（核心创新）：**

$$w_{uv} = \alpha \times \text{CNARW\_weight}(u,v) + (1-\alpha) \times \text{Fairness\_weight}(v)$$

结合node2vec的偏向参数：

$$w_{uv}^{\text{biased}} = w_{uv} \times \text{bias\_factor}(p, q)$$

**动态更新流程：**
1. **初始化 (t=1)**：
   - 使用Fair-CNARW对$G_1$生成初始嵌入
   - 预计算共同邻居和群体结构

2. **增量更新 (t=2 to T)**：
   - 识别演化节点：$\Delta V_t = V_{\text{add}} \cup \{\text{变化邻域的节点}\}$
   - 更新预计算结构（仅针对$\Delta V_t$）
   - 生成Fair-CNARW游走（仅针对$\Delta V_t$）
   - 增量训练Skip-gram模型

**三阶段随机游走决策：**
1. **回溯决策**（概率$p$）：返回前一个节点
2. **均匀随机决策**（概率$u$）：随机选择邻居
3. **Fair-CNARW偏向决策**：基于统一权重选择

<!-- **时间复杂度分析：**
- **每时间步**：$O(|\Delta V_t| \times n_{\text{walks}} \times \text{walk\_length})$
- **相比静态Fair-CNARW**：加速比 = $|V| / |\Delta V_t|$
- **相比dynnode2vec**：额外开销 < 20%（来自公平性计算）

**关键优势：**
- 保持dynnode2vec的可扩展性
- 引入Fair-CNARW的公平性机制
- 支持灵活的效率-公平性权衡
- 维持时间一致性 -->

### 2.3 DeepHub

#### 论文链接

[Dynamic Graph Embedding Through Hub-aware Random Walks](https://arxiv.org/pdf/2505.17764v2)

#### 核心思想

DeepHub 提出了一个基于深度学习的动态图嵌入方法，旨在解决传统方法在处理中心节点偏见时的局限性。它通过引入度中心性信息来改进节点嵌入，从而提高了对中心节点的表示能力。

#### 算法流程与核心公式

DeepHub通过中心节点感知的随机游走来解决hub节点偏见问题：

**中心节点感知权重计算：**

$$\text{score}(n) = \begin{cases}
1 + \deg(n) & \text{默认模式（吸引hub）} \\
1 + \max_{\deg} - \deg(n) & \text{逆向模式（回避hub）} \\
\log(1 + \text{score}) & \text{逆向对数模式}
\end{cases}$$
<!-- 1 + \log(1 + \max_{\deg}) - \log(1 + \deg(n)) -->

**三阶段随机游走策略：**
1. **回溯**（概率$p$）：返回前一个访问的节点
2. **均匀随机**（概率$u$）：随机选择邻居节点
3. **度数偏向**：基于中心节点感知的权重选择

**算法流程：**
1. 预计算所有节点的度数信息
2. 对于每个演化节点，生成中心节点感知的随机游走：
   - 以概率$p$返回前一个节点
   - 以概率$u$进行均匀随机移动
   - 否则根据度数权重进行偏向移动
3. 使用生成的游走训练Skip-gram模型
4. 增量更新嵌入向量

<!-- **关键发现：**
- **逆向模式最优**：在所有9个数据集上，inverse模式表现最佳
- **hub节点过度表示问题**：标准随机游走倾向于频繁访问hub节点
- **平衡表示**：通过回避hub节点，提高非hub节点的表示质量

**优势：**
- 减少对hub节点的偏见
- 提高图重构任务的F1分数
- 保持dynnode2vec的可扩展性
- 改善中心/非中心节点的性能平衡 -->


### 2.4 DeepHub+

#### 核心思想

DeepHub+ 是对原始DeepHub算法的扩展改进，引入了更精细的度数敏感策略。它在处理hub节点偏见的基础上，进一步提高了节点表示的多样性和质量。

#### 算法流程与核心公式

DeepHub+ 在DeepHub的基础上引入了动态度数阈值和自适应权重调整机制：

**动态度数阈值计算：**

$$\text{threshold} = \text{mean}(\text{degrees}) + \sigma \cdot \text{std}(\text{degrees})$$

其中$\sigma$是可调参数，控制阈值的严格程度。

**自适应权重函数：**

$$\text{adaptive\_weight}(n) = \begin{cases}
1 + \alpha \cdot \log(1 + \deg(n)) & \text{if } \deg(n) \leq \text{threshold} \\
1 + \beta \cdot \log(1 + \frac{\max_{\deg}}{\deg(n)}) & \text{if } \deg(n) > \text{threshold}
\end{cases}$$

其中$\alpha$和$\beta$是平衡参数，用于调节不同度数节点的权重分配。

**增强的随机游走策略：**

1. **度数感知初始化**：根据节点度数分布确定起始节点
2. **自适应移动**：使用动态权重进行邻居选择
3. **多样性促进**：避免过度访问高度数节点

**算法步骤：**

1. 计算图的度数分布统计信息
2. 确定动态度数阈值$\text{threshold}$
3. 对每个目标节点执行增强随机游走：
   - 计算自适应权重$\text{adaptive\_weight}(n)$
   - 根据权重概率选择下一个节点
   - 记录游走路径以训练嵌入模型
4. 使用Skip-gram模型训练节点嵌入
5. 评估并优化超参数$\alpha, \beta, \sigma$

**关键改进：**

- **自适应阈值**：根据图的度数分布动态调整策略
- **平衡权重**：更精细的权重分配机制
- **多样性增强**：确保低度数节点获得充分表示
- **计算复杂度**：只需要在每个网络快照计算图的均值方差，几乎不会影响计算复杂度。

#### 思路来源（探索方向）

DeepHub的固定偏置策略（`inverse-log`）是一种“一刀切”的方法。然而，学者在网络中的角色是多样化的。自适应游走策略能更好地捕捉这种异质性。

针对不同LID的学者采用不同游走策略。对于低LID学者，可以采用更具探索性的随机游走（类似DeepWalk），以“跳出”其所在的小圈子，探索更广阔的网络。对于高LID的“桥梁”学者，则可以采用更具结构性的游走（类似node2vec），以精确刻画其连接不同社区的独特角色。DeepHub论文本身也视其为重要的未来工作，且已有研究开始将LID应用于动态图。 [Local Intrinsic Dimensionality for Dynamic Graph Embeddings](https://arxiv.org/pdf/2411.16145)

*   **拓扑感知的自适应游走 (LID-aware Walks)：**
    *   局部内在维度（LID）衡量节点邻域的结构复杂性。在学者网络中，LID具有非常直观的解释：
        *   **低LID学者：** 处于一个结构简单、联系紧密的“学术小圈子”或实验室内部。其合作关系高度冗余（圈内成员相互都是合作者）。
        *   **高LID学者：** 扮演“桥梁”角色，连接多个不同但内部紧密的学术领域（例如，一位将机器学习应用于生物信息学的学者）。其合作关系复杂且非冗余。

### 2.5 DeepHub-Eigen

将DeepHub中的度中心性替换为特征向量中心性。

*   **概念**
    *   **度中心性**: 衡量的是合作者数量。一个学者度数高，意味着他/她有很多合作者。
    *   **特征向量中心性**: 衡量的是合作者的“质量”或影响力。一个学者特征向量中心性高，意味着他/她倾向于与本身也具有高影响力的学者合作。这在学术界是一个极其重要的信号，完美地对应了“学术声望”或“领域影响力”的概念。一个与诺贝尔奖得主合作的青年学者，其影响力传递远大于一个仅在内部小圈子发表大量论文的学者。因此，使用特征向量中心性来引导随机游走，能够更好地捕捉学术影响力的流动，而不仅仅是合作的频次。

*   **计算复杂度**
    *   **挑战：** 其计算复杂度为 $O(|E| \cdot k)$（其中 $k$ 为迭代次数），高于度中心性的 $O(|V| + |E|)$。对于覆盖数十万学者、数百万合作关系的大型网络（如DBLP），在每个时间快照（例如，每年）完全重算，成本不菲。此外，网络的谱隙（spectral gap）会影响收敛速度，当新的、相对孤立的学术社区出现时，可能会减慢计算。
    *   **对策：** 拉长网络快照周期的选择；增量更新；近似计算如幂迭代法，通过减少迭代次数 $k$ 可以直接换取速度。

## 3. 实验设计与评估

### 3.1 数据集

本研究使用了多个真实世界的动态网络数据集来评估所提出的Dynamic Fair-CNARW算法？

### 3.2 评估指标

#### 社区检测

[Network community detection via neural embeddings](https://doi.org/10.1038/s41467-024-52355-w)

#### 公平性评估

- **Demographic Parity**：群体间预测率的相等性
- **Equalized Odds**：条件公平性度量
- **Individual Fairness**：相似个体的相似待遇
- **Group Fairness Violation**：群体公平性违反程度

#### 计算效率

- **训练时间**：算法执行时间
- **内存使用**：peak内存消耗
- **可扩展性**：随图规模的性能变化
- **收敛速度**：达到稳定性能的迭代次数；SLEM

### 3.3 基线方法比较

**静态方法：**
- Node2Vec
- DeepWalk
- LINE
- FairWalk
- CNARW
- Fair-CNARW

**动态方法：**
- DynNode2Vec
- DynGEM
- DeepHub
- DynFairWalk
- DeepHub+(+)

## 4. 结果与分析


## 5. 结论与未来工作

### 5.1 主要贡献

本研究提出的Dynamic Fair-CNARW算法实现了以下重要贡献：

1. **算法创新**：提出Fair-CNARW并引入动态图嵌入，结合了Fair-CNARW的公平性保障与DynNode2Vec的效率优势；拓展DeepHub，提出DeepHub+(+)

2. **理论突破**：基于传统随机游走建立了动态公平性的理论框架，证明了增量公平性更新的收敛性（现有动态公平性研究都是基于DL）

3. **实验验证**：在多个真实数据集上验证了算法的有效性（待验证）

### 5.2 局限性

当前方法仍存在一些局限性：

1. **超参数敏感性**：算法性能对超参数设置较为敏感，需要针对不同数据集进行调优

2. **计算复杂性**：虽然相比静态重训练有显著改进，但在极大规模网络上仍面临计算挑战

3. **公平性定义**：当前主要关注度数公平性，未来需要扩展到其他公平性概念

### 5.3 未来研究方向

基于当前工作，我们确定了以下重要的未来研究方向：

1. **多维公平性**：扩展到种族、性别、年龄等多维属性的公平性保障

2. **联邦学习**：在保护隐私的前提下实现分布式动态公平嵌入

3. **解释性AI**：增强算法的可解释性，帮助理解公平性决策过程

4. **实时应用**：优化算法以支持毫秒级的实时动态更新

5. **跨域泛化**：研究算法在不同领域网络间的迁移能力

## 其他文献

### 公平性

[CrossWalk: Fairness-enhanced Node Representation Learning](https://arxiv.org/pdf/2105.02725)

[Drop Edges and Adapt: a Fairness Enforcing Fine-tuning for Graph Neural Networks](https://arxiv.org/pdf/2302.11479)

[FairEdit: Preserving Fairness in Graph Neural Networks through Greedy Graph Editing](https://arxiv.org/pdf/2201.03681)

[FairMILE: Towards an Efficient Framework for Fair Graph Representation Learning](https://dl.acm.org/doi/pdf/10.1145/3617694.3623231)

[Toward Structure Fairness in Dynamic Graph Embedding: A Trend-aware Dual Debiasing Approach](https://arxiv.org/pdf/2406.13201)

[FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning](https://arxiv.org/pdf/2410.17555)