
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    # YOUR CODE HERE
    Q = 0
    for next_state in mdp.get_next_states(state, action):
        p = mdp.get_transition_prob(state, action, next_state)
        r = mdp.get_reward(state, action, next_state)
        next_v = gamma * state_values[next_state]
        Q += p * (r + next_v)

    return Q