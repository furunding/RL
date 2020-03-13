[comment]: <> (最近修改日期：2020/02/13)
[cs 294课程主页](http://rail.eecs.berkeley.edu/deeprlcourse/)  
[cs 294课程视频](https://www.bilibili.com/video/av66523922?from=search&seid=15461326377484250354)  
[cs 294课程作业github](https://github.com/berkeleydeeprlcourse/homework_fall2019)  
[CS 294 深度强化学习中文笔记](https://zhuanlan.zhihu.com/c_125238795)  
[常用数学符号的 LaTeX 表示方法](http://www.mohu.org/info/symbols/symbols.htm)  


# Supervised Learning of Behavior
## 马尔科夫性
- **定义**  
  &emsp;&emsp;下一时刻的奖励和状态$r_(t+1),s_(t+1)$仅仅依赖于当前时刻的状态和动作$s_t, a_t$
- **概率图模型**  
  &emsp;&emsp;可以用概率图模型对状态、观测和动作的条件转移概率进行建模
  ![](https://pic3.zhimg.com/80/v2-d17d6d5173f20f1c72b7db7b6a60f0ae_hd.jpg)
- **状态序列满足马尔科夫性，但观测序列不一定满足**  
  &emsp;&emsp;由于观测只反映了部分事实，所以未来的观测不能由当前的观测完全确定
## 行为克隆
- **监督学习**
- **偏差积累**  
  &emsp;&emsp;由于现实的随机性或者复杂性，使得机器所采用的动作和人类的动作有偏差或者动作所产生的结果有偏差，这样在有偏差的一状态，机器还会做出有偏差的动作，使得之后状态的**偏差积累**
  ![](https://pic3.zhimg.com/80/v2-cdc2049ca175848e391180c45d6793e2_hd.jpg)
- **偏差修复**  
  &emsp;&emsp;有关汽车的自动驾驶的论文[《End to End Learning for Self-Driving Cars》](https://arxiv.org/abs/1604.07316)中，在采集数据时，汽车中间和两侧都放有摄像头，将三张图片作为观测，而相应的人类的正常驾驶行为作为标记，将这组数据打乱喂给CNN，监督学习出模仿人类的策略。出人意料的是这种简单的模仿学习实际上能成功。这是因为论文中所用的一个技巧：在标记左边摄像机画面时，它的的标记为人类正常动作加上小量的右转，而汽车右边摄像机图像的标记是人类正常动作加上小量的左转。这样机器行为在产生的向左偏差时，机器所接受到的画面就和正常情况下的左边摄像机的画面相似，而汽车左边摄像机图像的标记是人类正常动作加上小量的右转，因此机器进行有小量的右转。这样能在偏差时，检测到偏差并作出修复，也就是对的实际分布$o_t$起了稳定的作用。  
  &emsp;&emsp;对于这种想法的拓展，就是希望$o_t$的实际分布能相对学习时的分布稳定，一种方法是，学习用的数据不光是某种条件下的一条人类所走的路径，而是希望正确的路径有一个明确的概率分布（其中路径分布是指$p(\tau)=p(x_1, a_1, ..., x_T, a_T)$），然后在这个概率分布中取正确的路径，作为模仿学习的数据。因为实在正确路径的概率分布中取路径，因此其包含了许多偏差路径修正的例子作为学习数据，可从下图看出：![](https://pic2.zhimg.com/80/v2-36a19a1aa06524ffc2afa3aefd598f0d_hd.jpg)  
  &emsp;&emsp;这样实际操作中，机器由于学习了许多面对偏差的修正行为，能让实际路径分布相对学习时的分布稳定。而正确路径的概率分布的生成可以用下一节会讲的iLQR方法。
- **DAgger**  
  &emsp;&emsp;为了让$p_{data}(o_t) = p_{\pi_\theta}(o_t)$，我们从数据来源出发，想让数据来源和实际操作的分布相似，那么很直接的一个想法是直接从$p_{\pi_\theta}(o_t)$采样学习数据，做法就是实际运行策略$\pi_\theta$ ，但是需要对运行结果做标记，使其成为训练数据，然后更新策略$p_{\pi_\theta}(o_t)$，以此循环：![](https://pic4.zhimg.com/80/v2-5e67e987ff7824b5e3e42eb1adbb52db_hd.jpg)
- **偏差分析**
  ![](https://img-blog.csdnimg.cn/2019042500500948.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM1NjI4NQ==,size_16,color_FFFFFF,t_70)
- **Why might we fail to fit the expert?**  
  1. Non-Markovian behavior  
     &emsp;&emsp;人类专家通常会基于历史做决策，可以使用RNN处理历史数据，但是历史数据的使用可能会引入因果困惑问题；  
  2. Multimodal behavior  
    &emsp;&emsp;人类专家面对同一种情形时可能会做出截然不同的选择
    ![](https://img-blog.csdnimg.cn/20190211211537320.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM1NjI4NQ==,size_16,color_FFFFFF,t_70)
- **总结**  
  
  - 总是无效  
    &emsp;&emsp;数据分布不匹配
  - 有时有效
    - Hacks(e.g. left/right images)
    - Sample from a stable trajectory distribution(e.g. ask human expert to make mistakes)
    - Add more on-policy data, e.g. using Dagger
    - Better models that fit more accurately

# MDP
- **动力**  
  &emsp;&emsp;$p(s_{t+!}, r_{t+1}|s_t, a_t)$
- **Bellman期望方程**  
  $$
  v_\pi(s_t) = \max_{a_t}\,q_\pi(s_t, a_t)\\
  q_\pi(s_t, a_t) = r(s_t, a_t) + \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_\pi(s_{t+1})\\
  v_\pi(s_t) = \sum_{a_t}\,\pi(a_t|s_t)(r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_\pi(s_{t+1}))\\
  q_\pi(s_t, a_t) = r(s_t, a_t) + \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times \max_{a_t}\,q_\pi(s_{t+1}, a_{t+1})
  $$
- **Bellman最优方程** 
  $$
  v_*(s_t) = \max_{a_t}\,q_*(s_t, a_t)\\
  q_*(s_t, a_t)
  
   = r(s_t, a_t) + \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_*(s_{t+1})\\
  v_*(s_t) = \max_{a_t}\,r(s_t, a_t) + \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_*(s_{t+1})\\
  q_*(s_t, a_t) = r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times \max_{a_t}\,q_*(s_{t+1}, a_{t+1})
  $$
- **压缩映射**  
  - Bellman期望算子$t_\pi$  
    $t_\pi(v)(s_t) = \sum_{a_t}\,\pi(a_t|s_t)(r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_\pi(s_{t+1}))$
  - Bellman最优算子$t_*$  
    $t_*(v)(s_t) = \max_{a_t}\,r(s_t, a_t) + \gamma \sum_{s_{t+1}} p(s_{t+1} | s_t, a_t) \times v_*(s_{t+1})$

- **策略价值评估**  
  1. 求状态价值的线性方程组
  2. 利用Bellman期望算子迭代求解不动点

- **策略改进**  
  &emsp;&emsp;对于两个确定性策略$\pi$和${\pi}^{'}$，如果
  $$\forall s \in S, v_\pi(s) \le q_\pi (s, {\pi}^{'}(s))$$
  则$\pi \le {\pi}^{'}$，即6
  $$\forall s \in S, v_\pi(s) \le v_{{\pi}^{'}}(s)$$

- **策略迭代**  
  
  &emsp;&emsp;策略价值评估 + 策略改进

- **最优价值评估**  
  
  1. 将包含max的线性方程组转换为线性规划问题
  2. 利用Bellman最优算子迭代求解不动点

# 回合更新价值迭代
- **无模型算法**  
  
  &emsp;&emsp;在没有环境的数学描述的情况下，只依靠经验(例如轨迹的样本)学习出给定策略的价值函数和最优策略。  
  &emsp;&emsp;对于无模型的策略评估，动力$p$的表达式未知，只能用动作价值表达状态价值，反之则不行，因此动作价值函数更重要。

- **Monte Carlo方法**  
  &emsp;&emsp;对于给定策略，在许多轨迹样本中，如果某个状态或某个状态动作对出现了$c$次，其对应的回报值分别为$g_1, g_2, ..., g_c$，那么可以估计其状态价值或动作价值为$\frac{1}{c}\sum_{i=1}^{c} g_c$。  
  &emsp;&emsp;又称为回合更新，可分为每次访问回合更新和首次访问回合更新。  
- **增量法**  
  &emsp;&emsp;若前$c-1$次观察得到的回报样本是$g_1, g_2, ..., g_{c-1}$，则前$c-1$次价值函数的估计值为$\overline{g}_{c-1} = \frac{1}{c}\sum_{i=1}^{c} g_i$；若第$c$次的回报样本是$g_c$，则前c次价值函数的估计值为$\overline{g}_{c} = \overline{g}_{c-1} +\frac{1}{c}(g_{c} - \overline{g}_{c-1})$。  
  &emsp;&emsp;增量法是Robbins-Monro算法的一种。

- **同策回合更新**  
  - **局部最优**  
    &emsp;&emsp;同策算法可能会从一个并不好的策略出发，只经过那些很差的状态，然后只为那些很差的状态更新价值。  
  - **起始探索**  
    &emsp;&emsp;在回合开始时，从所有的状态动作对中随机选择，使其成为初始状态动作对。但是对于一些起始状态确定的环境并不适用。
  - **柔性策略**  
    &emsp;&emsp;对于某个策略，如果对于任意的$s \in S, a \in A$ 均有 $\pi(a | s) > 0$， 则称这个策略是柔性策略。  
    &emsp;&emsp;对于一个柔性策略，若其状态空间$S$和动作空间$A$都是有限集合，且对于任意的$s \in S, a \in A$，均有$\pi(a | s) > \varepsilon / |A(S)|$，则称策略$\pi$是$\varepsilon$柔性策略。  
    &emsp;&emsp;$\varepsilon$贪心策略是所有$\varepsilon$柔性策略中最接近确定性策略的一个
    $$
    \pi(a|s) =\left\{\begin{aligned} 1 - \varepsilon + \frac{\varepsilon}{|A(S|},\quad a = a^*\\ \frac{\varepsilon}{|A(S)|}, \quad a \neq a^*  \end{aligned} \right.
    $$
    
- **异策回合更新**  


  
# 案例研究

# 零碎知识
- 样本独立性  
  &emsp;&emsp;机器学习的模型(尤其是神经网络)要求样本是**独立同分布**的，强化学习的在线学习方式得到的样本是高度相关的，因此会给训练带来一些问题。DQN引入Replay Buffer，将新增样本丢入经验池中，训练时再从经验池中采样，这就打破了样本之间的相关性，这种方法本质上算是off-line的方法。A3C采用并行训练的方式，不同的Agent会经历不同的状态和转换，因此能够避免这种相关性。参考[一文读懂深度强化学习算法A3C算法](https://www.cnblogs.com/wangxiaocvpr/p/8110120.html)

- N-step return  
  &emsp;&emsp;优势在能够加快训练速度。设想在t时刻有一个较大的奖励，单步更新每次迭代只能影响t-1时刻的回报估计，但多步更新会影响更前面的时刻的回报估计，因此更快。
  &emsp;&emsp;劣势在于方差较大，因为一个回报估计涉及到多个状态和多个动作的选择，这些状态和动作都带有随机性，因此会引入高方差。
- 确定性策略  
  &emsp;&emsp;对于连续型任务，非确定性策略的动作空间较大，估计均值需要的样本数较多。

- 回合更新的特点  
  &emsp;&emsp;很好的收敛性，但收敛较慢；只能处理有终止 State 的任务，且较高的 Variance。

- PG总结[https://zhuanlan.zhihu.com/p/36506567]