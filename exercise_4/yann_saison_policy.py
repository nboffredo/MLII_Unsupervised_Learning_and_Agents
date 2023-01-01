from agent import Agent
import random

def pick_random_action(actions) -> str:
    return random.choice(list(actions))

def custom_policy(agent: Agent) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    position, known_rewards = agent.position, agent.known_rewards
    possible_actions = {"none" : known_rewards[position]}
    
    if (position > 0):
        possible_actions["left"] = known_rewards[position - 1]
    if (position < len(known_rewards) - 1):
        possible_actions["right"] = known_rewards[position + 1]

    action = "none"
    current_value = known_rewards[position]
    
    if (max(known_rewards) <= known_rewards[position]):
        action = pick_random_action(possible_actions)
    if (max(possible_actions.values())) > current_value:
        sorted_actions = sorted(
            possible_actions, key=lambda action: possible_actions[action], reverse=True
        )
        action = sorted_actions[0]
    
    assert action in possible_actions
    return action