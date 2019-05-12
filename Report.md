# Project report

## Multi Agent DDPG algorithm implementation details

The RL algorithm for the agents used is Deep Deterministic Policy Gradient with *Fixed Q-targets* and *random experience replay* as described in [original paper](https://arxiv.org/pdf/1509.02971.pdf).

#### Actor parameters:

**Model arhitecture**:
- Fully connected layer - input: 24 (state size) output: 512 activation : Relu
- Fully connected layer - input: 512 output: 256 activation : Relu
- Fully connected layer - input: 256 output: 2 (action size) activation : tanh

- Optimizer : Adam
- Learning rate = 1e-4

#### Critic parameters:

**Model arhitecture**:
- Fully connected layer - input: 24 (state size) output: 512 activation : Leaky Relu
- Fully connected layer - input: **512 (prev layer) + 2 (action size)** output: 256 activation : Leaky Relu
- Fully connected layer - input: 256 output: 2 (action size) 

- Optimizer : Adam
- Learning rate = 1e-4
- Loss : MSE

The above models are defined using **Pytorch**.

#### RL training parameters:

- Replay buffer size = 1e6
- Batch size = 512
- discount rate = 0.99
- soft update value = 0.2
- Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.) with time decay
- Maximum episodes = 30000

## Results

### Training Plot
RL training showing score vs episodes. 
![Score vs Episodes](/train_result.png)

```
Episode    0	Average Score: 0.00	Critic Loss: 0.0000000000	Actor Loss: 0.000000	t_step 21
Episode   50	Average Score: 0.00	Critic Loss: 0.0000117860	Actor Loss: -0.002961	t_step 793
Episode  100	Average Score: 0.00	Critic Loss: 0.0000069113	Actor Loss: 0.002673	t_step 1503
Episode  150	Average Score: 0.00	Critic Loss: 0.0000005561	Actor Loss: 0.004936	t_step 2213
Episode  200	Average Score: 0.00	Critic Loss: 0.0000010525	Actor Loss: 0.003575	t_step 2923
Episode  250	Average Score: 0.00	Critic Loss: 0.0000013537	Actor Loss: 0.003959	t_step 3633
Episode  300	Average Score: 0.00	Critic Loss: 0.0000032616	Actor Loss: 0.003177	t_step 4343
Episode  350	Average Score: 0.00	Critic Loss: 0.0002578712	Actor Loss: -0.009273	t_step 5101
Episode  400	Average Score: 0.00	Critic Loss: 0.0000654440	Actor Loss: 0.009304	t_step 5826
Episode  450	Average Score: 0.01	Critic Loss: 0.0003854933	Actor Loss: 0.018395	t_step 6871
Episode  500	Average Score: 0.04	Critic Loss: 0.0000569430	Actor Loss: -0.024736	t_step 8119
Episode  550	Average Score: 0.06	Critic Loss: 0.0000687173	Actor Loss: -0.048613	t_step 9517
Episode  600	Average Score: 0.08	Critic Loss: 0.0000963217	Actor Loss: -0.055150	t_step 11070
Episode  650	Average Score: 0.09	Critic Loss: 0.0001494859	Actor Loss: -0.077822	t_step 13048
Episode  700	Average Score: 0.11	Critic Loss: 0.0001418800	Actor Loss: -0.082690	t_step 15328
Episode  750	Average Score: 0.12	Critic Loss: 0.0001398104	Actor Loss: -0.085552	t_step 17665
Episode  800	Average Score: 0.12	Critic Loss: 0.0001798014	Actor Loss: -0.094822	t_step 19859
Episode  850	Average Score: 0.11	Critic Loss: 0.0001600345	Actor Loss: -0.086415	t_step 21886
Episode  900	Average Score: 0.11	Critic Loss: 0.0001728834	Actor Loss: -0.088994	t_step 24224
Episode  950	Average Score: 0.12	Critic Loss: 0.0001487788	Actor Loss: -0.093994	t_step 26834
Episode 1000	Average Score: 0.12	Critic Loss: 0.0001681939	Actor Loss: -0.098305	t_step 29089
Episode 1050	Average Score: 0.12	Critic Loss: 0.0001477163	Actor Loss: -0.099457	t_step 31405
Episode 1100	Average Score: 0.11	Critic Loss: 0.0001648622	Actor Loss: -0.097214	t_step 33604
Episode 1150	Average Score: 0.14	Critic Loss: 0.0001560414	Actor Loss: -0.099605	t_step 37061
Episode 1200	Average Score: 0.21	Critic Loss: 0.0002208860	Actor Loss: -0.123560	t_step 42091
Episode 1250	Average Score: 0.33	Critic Loss: 0.0003449130	Actor Loss: -0.202024	t_step 50601
Episode 1273	Average Score: 0.51	Score: 2.20	Critic Loss: 0.0005981674	Actor Loss: -0.279227	t_step 59533
Environment solved in 1273 episodes!	Average Score: 0.51
```

## Discussion

OU noise without decay is not allowing the DDPG to converge as the noise was forcing the agent to keep exploring only instead of exploitation. The OU noise is decayed based on the predefined total iterations (10000 in this case). The actor loss (negative of Expected return) is also decreasing along with increase in average score. 

## Ideas for future work

1. Try tuning the OU parameters for faster convergence
2. Try Prioritised Experience Replay
