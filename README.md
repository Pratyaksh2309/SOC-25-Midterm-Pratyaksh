# SOC'25
## ID:121 Reinforcement Learning in Self Driving Cars

### Name -> Pratyaksh Bhardwaj
### Roll No. -> 23B2401

<hr/>
Week 1:
This week, I began my journey into Reinforcement Learning (RL) and learned several foundational concepts. Our mentors recommended using OpenAI's Spinning Up guide as a starting point, and I genuinely enjoyed working through it. The interface and explanations were intuitive and made complex ideas much more approachable.
Out of curiosity, I also explored the Google DeepMind RL course and watched some of David Silver’s lectures (suggested by mentors). These gave me a deeper understanding of the theory behind RL and mathematics behind different things,

###### Some key concepts I learned included:

    Agent & Environment: The agent is the learner or decision-maker, and the environment is everything it interacts with. The agent takes actions, and the environment responds with new states and rewards.
    Policy: The strategy that the agent follows to choose actions. It can be deterministic or probabilistic.
    Reward Function: This provides feedback to the agent about how good or bad an action was in a particular state.
    Value Function: This estimates the expected return (total future reward) from a state or a state-action pair.

<hr/>
Week 2: 
This week, I explored several well-known RL algorithms. At first, I was intimidated by the amount of math involved, but over time, I started appreciating the logic behind each approach.

The algorithms I studied included:

    Q-Learning: A value-based method where the agent learns the value of actions in specific states.
    Monte Carlo Methods: These learn from complete episodes, estimating value functions based on actual returns.
    Proximal Policy Optimization (PPO): A popular policy gradient method known for stability and performance in continuous environments.
    Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC): Both are actor-critic methods suited for continuous action spaces.

Although the math was challenging, it was rewarding to grasp how algorithms update policies using gradients, Bellman equations, or sampling returns.

<hr/>
Week 3:
This week was all about applying theory to practice. We were given a coding assignment where we had to implement RL algorithms in OpenAI Gym environments, such as Frozen Lake and Pendulum.
Building these algorithms from scratch gave me a real sense of how the components come together. I referred to tutorials and also got help from “Johnny’s Code” repository, which helped me understand how different parts of an RL agent work in code.
I also found Gonkee’s YouTube videos very helpful—they explained complex RL implementations in a clear and engaging way. This hands-on experience made me much more comfortable with Python implementations of RL concepts and libraries like numpy, PyTorch, and Gym.

<hr/>
Week 4:
This week, we shifted our focus from RL to the mechanics of self-driving cars. We learned about their coordinate systems, motion models, and the various constraints they must follow for safe and efficient navigation.

Our mentors provided us with a curated playlist from the Tübingen Machine Learning Channel, which explained these concepts in a structured and easy-to-understand manner.

Key topics covered included:

    Kinematic & Dynamic Bicycle Models: These are simplified representations of vehicle motion used to model their behavior accurately.
    
Control Strategies:

    PID Controllers: Widely used in industry, they adjust control inputs based on proportional, integral, and derivative errors.
    Geometric Controllers: Like the Stanley Controller, often used for path tracking in autonomous vehicles.
    Optimal Control: Involves finding the control inputs that minimize a cost function, balancing performance and efficiency.



Week 5: Implementing Control in Code
This week and the current ongoing week, we are given a coding assignment where we had to implement the PID controller into a pre-written simulation. Our job is to fill in the mathematical components and tune the parameters to get the car to follow a desired path correctly.


### Over the last few weeks, I’ve learned a lot—both in theory and practice—about Reinforcement Learning and self-driving cars. RL introduced me to the idea of agents learning through interaction and delayed rewards, while self-driving car modules exposed me to real-world control systems and physical constraints.
