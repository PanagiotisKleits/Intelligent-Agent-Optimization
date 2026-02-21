3D Ball Balance: Custom Control with ML-Agents
This project is an implementation of an intelligent balancing agent using the Unity ML-Agents framework. The goal was to move beyond the standard tutorial logic and build an agent that achieves high-precision stability through custom observation handling and a proximity-based reward system.

Observation Strategy: 
Instead of a simple vector sensor, I used Reflection-based Observables with numStackedObservations: 9.

The Logic: In physics-based tasks like balancing, a single snapshot of the position isn't enough.

Temporal Context: By stacking 9 frames, the agent "senses" momentum and velocity trends, allowing it to make proactive adjustments before the ball even reaches the edge.

Focus: The observations track rotation deltas (z and x axes) and the relative position of the ball to the platform center.

Designing the Reward FunctionThe core of this agent's stability lies in its reward shaping. I replaced binary "step rewards" with a continuous mathematical function to ensure the agent is always trying to reach the absolute center, not just "stay on the board."I used the following formula for the reward (r):

r = 0.1/(1.0 + distanceToCenter)

This inverse-distance approach ensures that as the ball approaches the center , the reward increases. This avoids the "lazy agent" problem where a model settles for a sub-optimal policy just to avoid falling off.

Troubleshooting & Training Insights

This project benefited heavily from my previous analysis of failed RL environments. In past experiments, such as the BigWallJump, I identified that agents often get stuck in local minima due to a rapid drop in Entropy.

When the entropy falls too quickly, the agent stops exploring and becomes "rigid," often leading to negative cumulative rewards and idle behavior just to run out the clock. For this 3D Ball project, I closely monitored the training via TensorBoard to ensure a healthy entropy decay, allowing the agent to discover the centering strategy rather than just surviving.

Technical Setup

Framework: Unity ML-Agents.

Logic: C# script (Ball3DHardAgent.cs).

Backend: PyTorch.