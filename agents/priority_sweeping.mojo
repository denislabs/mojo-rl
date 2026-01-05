from random import random_si64, random_float64
from .qlearning import QTable
from core import TabularAgent


struct PrioritySweepingAgent(TabularAgent):
    """Priority Sweeping agent: efficient model-based planning.

    Like Dyna-Q but prioritizes updates by their expected impact:
    - States with large TD errors are updated first
    - Predecessors of updated states are added to the queue
    - Much more efficient than random sampling in Dyna-Q

    Algorithm:
    1. After real experience, compute TD error |δ|
    2. If |δ| > threshold, add (s,a) to priority queue with priority |δ|
    3. While queue not empty and n < n_planning:
       - Pop highest priority (s,a)
       - Update Q(s,a) using model
       - For all predecessors (s̄,ā) that lead to s:
         - Compute their TD error
         - If |δ| > threshold, add to queue

    Benefits over Dyna-Q:
    - Focuses computation on states that need updating
    - Propagates value changes backward efficiently
    - Much faster convergence in sparse reward environments
    """

    var q_table: QTable
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64
    var num_actions: Int
    var num_states: Int
    var n_planning: Int
    var priority_threshold: Float64

    # Model: (state, action) -> (next_state, reward)
    var model_next_state: List[Int]
    var model_reward: List[Float64]

    # Priority queue: parallel lists for priorities and pair indices
    # Using simple list-based priority queue (sufficient for small state spaces)
    var pq_priorities: List[Float64]
    var pq_pairs: List[Int]
    var pq_size: Int

    # Predecessor tracking: for each state, track (prev_state, action) pairs that lead to it
    # predecessors[state] contains list of (prev_state * num_actions + action)
    var predecessors: List[List[Int]]

    fn __init__(
        out self,
        num_states: Int,
        num_actions: Int,
        n_planning: Int = 5,
        priority_threshold: Float64 = 0.0001,
        learning_rate: Float64 = 0.1,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
    ):
        self.q_table = QTable(num_states, num_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states
        self.n_planning = n_planning
        self.priority_threshold = priority_threshold

        # Initialize model
        var total_pairs = num_states * num_actions
        self.model_next_state = List[Int]()
        self.model_reward = List[Float64]()
        for _ in range(total_pairs):
            self.model_next_state.append(-1)
            self.model_reward.append(0.0)

        # Initialize priority queue
        self.pq_priorities = List[Float64]()
        self.pq_pairs = List[Int]()
        self.pq_size = 0

        # Initialize predecessors - empty list for each state
        self.predecessors = List[List[Int]]()
        for _ in range(num_states):
            self.predecessors.append(List[Int]())

    fn _pair_index(self, state: Int, action: Int) -> Int:
        return state * self.num_actions + action

    fn select_action(self, state_idx: Int) -> Int:
        var rand = random_float64()
        if rand < self.epsilon:
            # random_si64 is inclusive on both ends, so use num_actions - 1
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.q_table.get_best_action(state_idx)

    fn _add_to_pq(mut self, pair_idx: Int, priority: Float64):
        """Add or update priority in queue."""
        # Check if already in queue
        for i in range(self.pq_size):
            if self.pq_pairs[i] == pair_idx:
                # Update priority if higher
                if priority > self.pq_priorities[i]:
                    self.pq_priorities[i] = priority
                return

        # Add new entry
        if self.pq_size < len(self.pq_pairs):
            self.pq_pairs[self.pq_size] = pair_idx
            self.pq_priorities[self.pq_size] = priority
        else:
            self.pq_pairs.append(pair_idx)
            self.pq_priorities.append(priority)
        self.pq_size += 1

    fn _pop_max_pq(mut self) -> Int:
        """Pop and return pair with highest priority."""
        if self.pq_size == 0:
            return -1

        # Find max priority
        var max_idx = 0
        var max_priority = self.pq_priorities[0]
        for i in range(1, self.pq_size):
            if self.pq_priorities[i] > max_priority:
                max_priority = self.pq_priorities[i]
                max_idx = i

        var result = self.pq_pairs[max_idx]

        # Remove by swapping with last
        self.pq_size -= 1
        if max_idx < self.pq_size:
            self.pq_pairs[max_idx] = self.pq_pairs[self.pq_size]
            self.pq_priorities[max_idx] = self.pq_priorities[self.pq_size]

        return result

    fn _add_predecessor(mut self, next_state: Int, state: Int, action: Int):
        """Record that (state, action) leads to next_state."""
        var pred_pair = self._pair_index(state, action)

        # Check if already recorded
        for i in range(len(self.predecessors[next_state])):
            if self.predecessors[next_state][i] == pred_pair:
                return

        self.predecessors[next_state].append(pred_pair)

    fn update(
        mut self,
        state_idx: Int,
        action: Int,
        reward: Float64,
        next_state_idx: Int,
        done: Bool,
    ):
        """Priority Sweeping update."""
        var idx = self._pair_index(state_idx, action)

        # Update model
        self.model_next_state[idx] = next_state_idx
        self.model_reward[idx] = reward

        # Track predecessor
        self._add_predecessor(next_state_idx, state_idx, action)

        # Compute TD error (priority)
        var current_q = self.q_table.get(state_idx, action)
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table.get_max_value(next_state_idx)
        var td_error = target - current_q

        # Direct update
        var new_q = current_q + self.learning_rate * td_error
        self.q_table.set(state_idx, action, new_q)

        # Add to priority queue if significant
        var priority = td_error if td_error > 0 else -td_error  # abs
        if priority > self.priority_threshold:
            self._add_to_pq(idx, priority)

        # Planning loop
        for _ in range(self.n_planning):
            if self.pq_size == 0:
                break

            var pair_idx = self._pop_max_pq()
            if pair_idx < 0:
                break

            var s = pair_idx // self.num_actions
            var a = pair_idx % self.num_actions

            # Get model prediction
            var ns = self.model_next_state[pair_idx]
            var r = self.model_reward[pair_idx]

            if ns < 0:  # Not in model
                continue

            # Update Q-value
            var q = self.q_table.get(s, a)
            var t = r + self.discount_factor * self.q_table.get_max_value(ns)
            var delta = t - q
            self.q_table.set(s, a, q + self.learning_rate * delta)

            # Process predecessors of s
            for i in range(len(self.predecessors[s])):
                var pred_pair = self.predecessors[s][i]
                var ps = pred_pair // self.num_actions
                var pa = pred_pair % self.num_actions

                # Check if in model
                if self.model_next_state[pred_pair] < 0:
                    continue

                var pr = self.model_reward[pred_pair]
                var pq = self.q_table.get(ps, pa)
                var pt = pr + self.discount_factor * self.q_table.get_max_value(s)
                var p_delta = pt - pq
                var p_priority = p_delta if p_delta > 0 else -p_delta

                if p_priority > self.priority_threshold:
                    self._add_to_pq(pred_pair, p_priority)

    fn decay_epsilon(mut self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        return self.epsilon

    fn get_best_action(self, state_idx: Int) -> Int:
        return self.q_table.get_best_action(state_idx)
