import pygame
import sys
import random
from collections import deque
import heapq
import timeit
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import defaultdict


class EightPuzzle:
    def __init__(self, initial, goal):
        self.initial = tuple(map(tuple, initial))
        self.goal = tuple(map(tuple, goal))
        self.rows, self.cols = 3, 3

    # tìm vị trí ô trống
    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j
        return None

    # tìm các trạng thái có thể đạt được từ trạng thái hiện tại
    def get_neighbors(self, state):
        row, col = self.find_blank(state)
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        state_list = [list(row) for row in state]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row[:] for row in state_list]
                new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                moves.append(tuple(map(tuple, new_state)))
        return moves

    # chuyển trạng thái sang list
    def state_to_list(self, state):
        return [num for row in state for num in row]

    # chuyển list sang trạng thái
    def list_to_state(self, lst):
        return tuple(tuple(lst[i * 3:(i + 1) * 3]) for i in range(3))

    # tạo trạng thái ngẫu nhiên, đảm bảo có thể giải được
    def generate_random_state(self):
        numbers = list(range(9))
        random.shuffle(numbers)
        state = self.list_to_state(numbers)
        while not self.is_solvable(state):
            random.shuffle(numbers)
            state = self.list_to_state(numbers)
        return state


    # tính độ phù hợp trạng thái, càng gần 0 thì càng gần mục tiêu
    def fitness(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return -distance

    # thực hiện phép lai cha mẹ trả về trạng thái con
    def crossover(self, parent1, parent2):
        p1_list = self.state_to_list(parent1)
        p2_list = self.state_to_list(parent2)
        crossover_point = random.randint(1, 7)
        child = p1_list[:crossover_point]
        seen = set(child)
        for num in p2_list:
            if num not in seen:
                child.append(num)
                seen.add(num)
        return self.list_to_state(child)

    # thực hiện phép đột biến bằng cách hoán đổi 2 vị trí (nếu nhỏ hơn xác suất đột hiến)
    def mutate(self, state, mutation_rate=0.05):
        state_list = self.state_to_list(state)
        if random.random() < mutation_rate:
            i, j = random.sample(range(9), 2)
            state_list[i], state_list[j] = state_list[j], state_list[i]
        return self.list_to_state(state_list)


    # xây dựng đường đi từ trạng thái ban đầu đến trạng thái cuối cùng
    def reconstruct_path(self, final_state):
        path = [self.initial]
        current = self.initial
        while current != final_state:
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda x: self.heuristic(x), default=final_state)
            path.append(current)
            if current == final_state:
                break
        return path

    # thuật toán di truyền
    def genetic_algorithm(self, population_size=50, max_generations=500):
        population = [self.generate_random_state() for _ in range(population_size)]
        explored_states = []
        best_fitness = float('-inf')
        no_improvement_count = 0
        max_no_improvement = 100

        for generation in range(max_generations):
            population = sorted(population, key=self.fitness, reverse=True)
            explored_states.extend(population[:5])
            best_state = population[0]
            current_fitness = self.fitness(best_state)

            if best_state == self.goal:
                path = self.reconstruct_path(best_state)
                return path, explored_states

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                break

            new_population = population[:population_size // 2]
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(new_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        return None, explored_states

    def bfs(self):
        queue = deque([(self.initial, [])])
        visited = {self.initial}
        explored_states = []

        while queue:
            state, path = queue.popleft()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))
        return None, explored_states


    def dfs(self):
        stack = [(self.initial, [])]
        visited = {self.initial}
        explored_states = []
        while stack:
            state, path = stack.pop()
            explored_states.append(state)
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [state]))
        return None, explored_states

    def ucs(self):
        open_list = []
        heapq.heappush(open_list, (0, self.initial, []))
        visited = {}
        explored_states = []

        while open_list:
            cost, state, path = heapq.heappop(open_list)
            explored_states.append(state)
            if state in visited and visited[state] < cost:
                continue
            visited[state] = cost
            if state == self.goal:
                return path + [state], explored_states
            for neighbor in self.get_neighbors(state):
                new_cost = cost + 1
                if neighbor not in visited or new_cost < visited[neighbor]:
                    heapq.heappush(open_list, (new_cost, neighbor, path + [state]))
        return None, explored_states

    def depth_limited_search(self, state, path, depth, visited):
        explored_states = []
        if state == self.goal:
            return path + [state], explored_states
        if depth == 0:
            return None, explored_states
        visited.add(state)
        explored_states.append(state)
        for neighbor in self.get_neighbors(state):
            if neighbor not in visited:
                result, sub_explored = self.depth_limited_search(neighbor, path + [state], depth - 1, visited)
                explored_states.extend(sub_explored)
                if result:
                    return result, explored_states
        return None, explored_states

    def ids(self):
        max_depth = 100
        depth = 0
        while depth <= max_depth:
            visited = set()
            result, explored_states = self.depth_limited_search(self.initial, [], depth, visited)
            if result:
                return result, explored_states
            depth += 1
        return None, []

    def heuristic(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] and state[i][j] != self.goal[i][j]:
                    value = state[i][j]
                    goal_i, goal_j = divmod(value - 1, 3)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def greedy_search(self, timeout=10.0):
        start_time = time.time()
        pq = []
        heapq.heappush(pq, (self.heuristic(self.initial), self.initial, []))
        visited = set()
        explored_states = []

        while pq:
            if time.time() - start_time > timeout:
                print("GBFS timeout after", timeout, "seconds")
                return None, explored_states

            _, state, path = heapq.heappop(pq)
            if state in visited:  # Kiểm tra visited ngay sau khi pop để bỏ qua nếu đã thăm
                continue

            explored_states.append(state)
            visited.add(state)

            if state == self.goal:
                return path + [state], explored_states

            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    heapq.heappush(pq, (self.heuristic(neighbor), neighbor, path + [state]))

        return None, explored_states

    def a_star(self, timeout=10.0):
        start_time = time.time()
        pq = []
        heapq.heappush(pq, (self.heuristic(self.initial), 0, self.initial, []))
        explored_states = []

        while pq:
            if time.time() - start_time > timeout:
                print("A* timeout after", timeout, "seconds")
                return None, explored_states

            f, g, state, path = heapq.heappop(pq)
            explored_states.append(state)

            if state == self.goal:
                return path + [state], explored_states

            for neighbor in self.get_neighbors(state):
                new_g = g + 1
                new_f = new_g + self.heuristic(neighbor)
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [state]))

        return None, explored_states

    def ida_star(self, timeout=10.0):
        start_time = time.time()
        threshold = self.heuristic(self.initial)
        explored_states = []
        iteration = 0

        while True:
            if time.time() - start_time > timeout:
                print("IDA* timeout after", timeout, "seconds")
                return None, explored_states

            iteration += 1
            print(f"IDA* iteration {iteration}, threshold = {threshold}")  # Thông báo tiến trình
            result, new_threshold = self.ida_star_recursive(self.initial, [], 0, threshold, explored_states)
            if result:
                return result, explored_states
            if new_threshold == float('inf'):
                return None, explored_states
            threshold = new_threshold

    def ida_star_recursive(self, state, path, g, threshold, explored_states):
        f = g + self.heuristic(state)
        explored_states.append(state)

        if f > threshold:
            return None, f
        if state == self.goal:
            return path + [state], threshold

        min_threshold = float('inf')
        for neighbor in self.get_neighbors(state):
            result, new_threshold = self.ida_star_recursive(neighbor, path + [state], g + 1, threshold, explored_states)
            if result:
                return result, threshold
            min_threshold = min(min_threshold, new_threshold)

        return None, min_threshold

    def simple_hill_climbing(self):
        current = self.initial
        path = [current]
        explored_states = []

        while True:
            explored_states.append(current)
            neighbors = self.get_neighbors(current)
            next_state = None
            current_heuristic = self.heuristic(current)

            for neighbor in neighbors:
                if self.heuristic(neighbor) < current_heuristic:
                    next_state = neighbor
                    break

            if next_state is None:
                break

            current = next_state
            path.append(current)

            if current == self.goal:
                return path, explored_states

        return None, explored_states

    def steepest_ascent_hill_climbing(self):
        current = self.initial
        path = [current]
        explored_states = []

        while True:
            explored_states.append(current)
            neighbors = self.get_neighbors(current)
            next_state = None
            best_heuristic = float('inf')

            for neighbor in neighbors:
                neighbor_heuristic = self.heuristic(neighbor)
                if neighbor_heuristic < best_heuristic:
                    best_heuristic = neighbor_heuristic
                    next_state = neighbor

            if next_state is None or best_heuristic >= self.heuristic(current):
                break

            current = next_state
            path.append(current)

            if current == self.goal:
                return path, explored_states

        return None, explored_states

    def random_hill_climbing(self):
        current = self.initial
        path = [current]
        explored_states = []

        while True:
            explored_states.append(current)
            neighbors = self.get_neighbors(current)
            next_states = [neighbor for neighbor in neighbors if self.heuristic(neighbor) < self.heuristic(current)]

            if not next_states:
                break

            next_state = random.choice(next_states)
            current = next_state
            path.append(current)

            if current == self.goal:
                return path, explored_states

        return None, explored_states

    def is_solvable(self, state=None):
        if state is None:
            state = self.initial
        state_list = [num for row in state for num in row if num != 0]
        inversions = 0
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] > state_list[j]:
                    inversions += 1
        return inversions % 2 == 0

    def simulated_annealing(self):
        current = self.initial
        path = [current]
        explored_states = set()
        current_heuristic = self.heuristic(current)
        best_state = current
        best_heuristic = current_heuristic

        temperature = 1000.0  # Tăng nhiệt độ ban đầu
        cooling_rate = 0.99  # Giảm tốc độ làm nguội
        no_improvement_count = 0
        max_no_improvement = 2000  # Tăng giới hạn không cải thiện

        while temperature > 0.01 and no_improvement_count < max_no_improvement:
            explored_states.add(current)

            neighbors = self.get_neighbors(current)
            if not neighbors:
                break

            neighbor_heuristic_pairs = [(neighbor, self.heuristic(neighbor)) for neighbor in neighbors]
            neighbor_heuristic_pairs.sort(key=lambda x: x[1])
            next_state, next_heuristic = neighbor_heuristic_pairs[0]

            if next_heuristic >= current_heuristic:
                delta = next_heuristic - current_heuristic
                acceptance_probability = math.exp(-delta / temperature)
                if random.uniform(0, 1) > acceptance_probability:
                    next_state, next_heuristic = random.choice(neighbor_heuristic_pairs)

            if next_heuristic < current_heuristic:
                no_improvement_count = 0
                if next_heuristic < best_heuristic:
                    best_state = next_state
                    best_heuristic = next_heuristic
            else:
                no_improvement_count += 1

            current = next_state
            current_heuristic = next_heuristic
            path.append(current)

            if current == self.goal:
                return path, list(explored_states)

            temperature *= cooling_rate

        if best_state == self.goal:
            return self.reconstruct_path(best_state), list(explored_states)
        return None, list(explored_states)

    def beam_search(self, beam_width=5):  # Tăng beam_width
        initial_state = self.initial
        if initial_state == self.goal:
            return [initial_state], []

        current_states = [initial_state]
        path = {initial_state: []}

        while current_states:
            next_states = []
            for state in current_states:
                neighbors = self.get_neighbors(state)
                for neighbor in neighbors:
                    if neighbor not in path:
                        path[neighbor] = path[state] + [state]
                        next_states.append(neighbor)

            evaluated = [(self.heuristic(state), state) for state in next_states]
            evaluated.sort(key=lambda x: x[0])

            current_states = [state for (_, state) in evaluated[:beam_width]]

            if self.goal in current_states:
                return path[self.goal] + [self.goal], list(path.keys())

        return None, list(path.keys())

    def and_or_search(self):
        from collections import deque

        # Khởi tạo hàng đợi với trạng thái ban đầu và đường đi tương ứng
        queue = deque([(self.initial, [])])
        visited = set()
        explored_states = []

        while queue:
            state, path = queue.popleft()
            explored_states.append(state)

            # Kiểm tra xem trạng thái hiện tại có phải là mục tiêu không
            if state == self.goal:
                return path + [state], explored_states

            # Nếu trạng thái chưa được thăm
            if state not in visited:
                visited.add(state)

                # Lấy tất cả các trạng thái lân cận (các nhánh OR)
                neighbors = self.get_neighbors(state)
                # Thêm tất cả các trạng thái lân cận vào hàng đợi (AND: cần khám phá tất cả)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [state]))

        return None, explored_states

    def generate_random_state(self, max_depth=5):
        """
        Tạo trạng thái ngẫu nhiên gần initial_state trong max_depth bước.
        """
        current_state = self.initial
        for _ in range(random.randint(1, max_depth)):
            neighbors = self.get_neighbors(current_state)
            current_state = random.choice(neighbors)
        return current_state

    def bfs_for_belief(self, start_state, max_depth=10):
        """
        Chạy BFS từ trạng thái start_state để tìm các trạng thái lân cận trong max_depth bước.
        Trả về: Tập hợp các trạng thái lân cận (giới hạn tối đa 5 trạng thái).
        """
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = set()

        while queue and len(states) < 5:  # Giới hạn số trạng thái
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.add(neighbor)
        return states

    def belief_state_search(self, initial_belief, max_steps=5000):
        initial_belief = set(initial_belief)
        explored = set()
        num_explored_states = 0
        belief_states_path = [list(initial_belief)]
        total_steps = 0
        best_heuristic = float('inf')
        no_improvement_steps = 0

        # Initialize explored states
        for state in initial_belief:
            explored.add(state)
            num_explored_states += 1
            h = self.heuristic(state)
            if h < best_heuristic:
                best_heuristic = h

        belief_queue = []
        representative_state = min(initial_belief, key=self.heuristic)
        heapq.heappush(belief_queue, (self.heuristic(representative_state), initial_belief, []))
        visited = set()

        while belief_queue and num_explored_states < max_steps:
            _, belief_state, path = heapq.heappop(belief_queue)
            belief_state_tuple = frozenset(belief_state)

            for state in belief_state:
                if state == self.goal:
                    for initial_state in initial_belief:
                        self.initial = initial_state
                        solution, _ = self.bfs()
                        if solution:
                            total_steps += len(solution) - 1
                        else:
                            return None, explored, 0
                    belief_states_path.append([self.goal] * len(initial_belief))
                    return belief_states_path, explored, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            action_scores = []
            for action in range(4):
                new_belief = set()
                min_heuristic = float('inf')
                count = 0
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_states = self.optimized_bfs_for_belief(state, max_depth=1)
                        new_belief.update(next_states)
                        for next_state in next_states:
                            h = self.heuristic(next_state)
                            min_heuristic = min(min_heuristic, h)
                            count += 1
                if new_belief:
                    score = min_heuristic + len(new_belief) * 0.1
                    action_scores.append((action, score, new_belief))

            if not action_scores:
                continue

            action_scores.sort(key=lambda x: x[1])

            if random.random() < 0.1:
                action, _, new_belief = random.choice(action_scores)
            else:
                action, _, new_belief = action_scores[0]

            if len(new_belief) > 10:
                new_belief = set(sorted(new_belief, key=self.heuristic)[:10])

            if new_belief:
                representative_state = min(new_belief, key=self.heuristic)
                new_heuristic = self.heuristic(representative_state)
                if new_heuristic < best_heuristic:
                    best_heuristic = new_heuristic
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1

                if no_improvement_steps > 500 or len(new_belief) > 50:
                    return None, explored, 0

                for state in new_belief:
                    if state not in explored:
                        explored.add(state)
                        num_explored_states += 1
                new_path = path + [representative_state]
                heapq.heappush(belief_queue, (new_heuristic, new_belief, new_path))
                belief_states_path.append(list(new_belief))

        return None, explored, 0

    def optimized_bfs_for_belief(self, start_state, max_depth=1):
        queue = deque([(start_state, 0)])
        visited = {start_state}
        states = [(self.heuristic(start_state), start_state)]

        while queue:
            state, depth = queue.popleft()
            if depth < max_depth:
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        states.append((self.heuristic(neighbor), neighbor))

        states.sort()
        return {state for _, state in states[:10]}



    def get_observation(self, state):
        """
        Giả lập quan sát một phần: chỉ quan sát được vị trí của ô số 1.
        Trả về vị trí (row, col) của ô số 1 trong trạng thái.
        """
        for i in range(3):
            for j in range(3):
                if state[i][j] == 1:
                    return (i, j)
        return None

    def find_states_with_one_at_00(self, start_state, max_states=6):
        """
        Tìm các trạng thái có số 1 ở vị trí (0,0) bằng BFS.
        start_state: Trạng thái ban đầu.
        max_states: Số trạng thái tối đa cần tìm.
        Trả về: Danh sách các trạng thái (dạng tuple) có số 1 ở (0,0).
        """
        queue = deque([(start_state, [])])
        visited = {start_state}
        states_with_one_at_00 = []

        while queue and len(states_with_one_at_00) < max_states:
            state, path = queue.popleft()
            if self.get_observation(state) == (0, 0):
                states_with_one_at_00.append(state)
                if len(states_with_one_at_00) >= max_states:
                    break
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [state]))

        while len(states_with_one_at_00) < max_states:
            numbers = list(range(9))
            random.shuffle(numbers)
            numbers[0] = 1
            remaining_numbers = [num for num in numbers[1:] if num != 1]
            if len(remaining_numbers) < 8:
                remaining_numbers.append(0)
            numbers = [1] + remaining_numbers[:8]
            state = self.list_to_state(numbers)
            if self.is_solvable(state) and state not in states_with_one_at_00:
                states_with_one_at_00.append(state)

        return states_with_one_at_00[:max_states]

    def partial_observable_search(self):
        """
        Partial Observable Search: Tìm kiếm trên không gian belief states với số 1 ở (0,0).
        Trả về: (belief_states_path, explored_states, total_steps) hoặc (None, explored_states, 0).
        """
        # Khởi tạo initial_belief với 6 trạng thái có số 1 ở (0,0)
        initial_belief = self.find_states_with_one_at_00(self.initial, max_states=6)
        queue = deque([(set(initial_belief), [], 0)])
        visited = set()
        explored_states = []
        belief_states_path = [list(initial_belief)[:6]]
        max_steps = 1000

        while queue and len(queue) < max_steps:
            belief_state, path, steps = queue.popleft()
            belief_state_tuple = frozenset(belief_state)
            explored_states.extend(belief_state)

            # Kiểm tra điều kiện mục tiêu: Chỉ cần một trạng thái là goal
            for state in belief_state:
                if state == self.goal:
                    # Tính tổng số bước bằng BFS từ initial_belief đến goal
                    total_steps = 0
                    for initial_state in initial_belief:
                        self.initial = initial_state
                        solution, _ = self.bfs()
                        if solution:
                            total_steps += len(solution) - 1
                        else:
                            return None, explored_states, 0
                    belief_states_path.append([self.goal] * 6)  # Hiển thị goal state cho tất cả Belief States
                    return belief_states_path, explored_states, total_steps

            if belief_state_tuple in visited:
                continue
            visited.add(belief_state_tuple)

            actions = set()
            for state in belief_state:
                neighbors = self.get_neighbors(state)
                for action in range(len(neighbors)):
                    actions.add(action)

            for action in actions:
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_states = self.bfs_for_belief(state, max_depth=1)
                        new_belief.update(next_states)

                # Lọc new_belief để chỉ giữ trạng thái có số 1 ở (0,0)
                new_belief = set(state for state in new_belief if self.get_observation(state) == (0, 0))

                if new_belief and len(new_belief) <= 20:  # Giới hạn kích thước new_belief
                    representative_state = min(belief_state, key=self.heuristic)
                    new_path = path + [representative_state]
                    queue.append((new_belief, new_path, steps + 1))
                    belief_states_path.append(list(new_belief)[:6])  # Lưu belief state mới

        return None, explored_states, 0

    def is_valid_assignment(self, state, pos, value):
        """
        Kiểm tra xem việc gán giá trị cho ô pos có thỏa mãn các ràng buộc không.
        state: Ma trận hiện tại (có thể chứa None).
        pos: Vị trí ô (i,j).
        value: Giá trị cần gán (0-8).
        Trả về: True nếu hợp lệ, False nếu không.
        """
        i, j = pos
        # Ràng buộc: Ô (0,0) phải là 1
        if i == 0 and j == 0 and value != 1:
            return False

        # Ràng buộc: Mỗi số chỉ xuất hiện một lần
        for r in range(3):
            for c in range(3):
                if (r, c) != pos and state[r][c] == value:
                    return False

        # Ràng buộc theo hàng: ô(i,j+1) = ô(i,j) + 1 (trừ ô trống)
        if j > 0 and state[i][j - 1] is not None and value != 0 and state[i][j - 1] != value - 1:
            return False
        if j < 2 and value != 0 and state[i][j + 1] is not None and state[i][j + 1] != value + 1:
            return False

        # Ràng buộc theo cột: ô(i+1,j) = ô(i,j) + 3 (trừ ô trống)
        if i > 0 and state[i - 1][j] is not None and value != 0 and state[i - 1][j] != value - 3:
            return False
        if i < 2 and value != 0 and state[i + 1][j] is not None and state[i + 1][j] != value + 3:
            return False

        return True

    def is_solvable(self, state):
        """
        Kiểm tra xem ma trận có solvable không (số nghịch đảo chẵn).
        state: Ma trận 3x3 (có thể chứa None).
        """
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None and state[i][j] != 0]
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0

    def backtracking_search(self, depth_limit=9):
        """
        Backtracking Search cho CSP: Gán giá trị cho các ô từ ma trận rỗng.
        depth_limit: Số ô tối đa được gán (9 cho ma trận 3x3).
        Trả về: (path, explored_states) hoặc (None, explored_states).
        """
        visited = set()  # Lưu các trạng thái đã thăm
        explored_states = []  # Lưu các trạng thái đã khám phá
        path = []  # Lưu đường đi từ rỗng đến mục tiêu

        def backtrack(state, assigned, pos_index):
            """
            Hàm đệ quy để thực hiện backtracking.
            state: Ma trận hiện tại (có thể chứa None).
            assigned: Tập các giá trị đã gán.
            pos_index: Chỉ số ô đang gán (0-8).
            """
            # Kiểm tra trạng thái mục tiêu
            if pos_index == 9:  # Đã gán hết 9 ô
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            # Lấy vị trí ô tiếp theo
            i, j = divmod(pos_index, 3)
            if i >= 3 or j >= 3:
                return None

            # Tạo bản sao trạng thái
            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            # Thử gán các giá trị còn lại
            for value in range(9):
                if value not in assigned:
                    if self.is_valid_assignment(state, (i, j), value):
                        # Gán giá trị
                        new_state = [row[:] for row in state]
                        new_state[i][j] = value
                        new_assigned = assigned | {value}
                        path.append(state_tuple)  # Thêm trạng thái trước khi gán
                        result = backtrack(new_state, new_assigned, pos_index + 1)
                        if result is not None:
                            return result
                        path.pop()  # Quay lui: xóa trạng thái nếu không thành công

            return None

        # Khởi tạo ma trận rỗng
        empty_state = [[None for _ in range(3)] for _ in range(3)]
        result = backtrack(empty_state, set(), 0)
        return result, explored_states

    def forward_checking_search(self, depth_limit=9):
        """
        Forward Checking Search cho CSP: Gán giá trị cho các ô từ ma trận rỗng với Forward Checking, MRV, và LCV.
        depth_limit: Số ô tối đa được gán (9 cho ma trận 3x3).
        Trả về: (path, explored_states) hoặc (None, explored_states).
        """
        visited = set()  # Lưu các trạng thái đã thăm
        explored_states = []  # Lưu các trạng thái đã khám phá
        path = []  # Lưu đường đi từ rỗng đến mục tiêu

        def get_domain(state, pos, assigned):
            """
            Lấy tập giá trị hợp lệ cho ô tại pos.
            state: Ma trận hiện tại.
            pos: Vị trí ô (i,j).
            assigned: Tập các giá trị đã gán.
            Trả về: Danh sách các giá trị hợp lệ.
            """
            domain = []
            for value in range(9):
                if value not in assigned and self.is_valid_assignment(state, pos, value):
                    domain.append(value)
            return domain

        def forward_check(state, pos, value, domains, assigned):
            """
            Thực hiện Forward Checking: Cập nhật tập giá trị của các ô chưa gán.
            state: Ma trận hiện tại.
            pos: Vị trí ô vừa gán (i,j).
            value: Giá trị vừa gán.
            domains: Từ điển chứa tập giá trị hợp lệ cho các ô.
            assigned: Tập các vị trí và giá trị đã gán.
            Trả về: True nếu tất cả ô chưa gán còn giá trị hợp lệ, False nếu không.
            """
            new_domains = {k: v[:] for k, v in domains.items()}
            used_values = set(state[r][c] for r in range(3) for c in range(3) if state[r][c] is not None)
            related_positions = [(r, c) for r in range(3) for c in range(3) if
                                 (r, c) not in assigned and (r == pos[0] or c == pos[1])]
            for other_pos in related_positions:
                if other_pos != pos:
                    i, j = other_pos
                    new_domain = [val for val in new_domains[other_pos] if val not in used_values]
                    if pos == (0, 0) and value == 1:
                        if other_pos == (0, 1):
                            new_domain = [2]
                        elif other_pos == (1, 0):
                            new_domain = [4]
                    elif value != 0:  # Chỉ áp dụng ràng buộc hàng/cột nếu value không phải ô trống
                        if j > 0 and state[i][j - 1] is not None and state[i][j - 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[i][j - 1] == val - 1]
                        if j < 2 and state[i][j + 1] is not None and state[i][j + 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[i][j + 1] == val + 1]
                        if i > 0 and state[i - 1][j] is not None and state[i - 1][j] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[i - 1][j] == val - 3]
                        if i < 2 and state[i + 1][j] is not None and state[i][j + 1] != 0:
                            new_domain = [val for val in new_domain if val == 0 or state[i][j + 1] == val + 3]
                    new_domains[other_pos] = new_domain
                    if not new_domain:
                        return False, domains
            return True, new_domains

        def select_mrv_variable(positions, domains, state):
            """
            Chọn ô có ít giá trị hợp lệ nhất (MRV), ưu tiên ô có nhiều ràng buộc trực tiếp.
            positions: Danh sách các ô chưa gán.
            domains: Từ điển chứa tập giá trị hợp lệ.
            state: Ma trận hiện tại.
            Trả về: Vị trí ô được chọn.
            """
            max_constraints = -1
            selected_pos = None
            for pos in positions:
                i, j = pos
                constraints = sum(1 for r, c in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)] if
                                  0 <= r < 3 and 0 <= c < 3 and state[r][c] is not None)
                domain_size = len(domains[pos])
                if constraints > max_constraints or (constraints == max_constraints and domain_size < (
                len(domains[selected_pos]) if selected_pos else float('inf'))):
                    max_constraints = constraints
                    selected_pos = pos
            return selected_pos

        def select_lcv_value(pos, domain, state, domains, assigned):
            """
            Chọn giá trị ít ràng buộc nhất (LCV), ưu tiên giá trị trong goal_state.
            pos: Vị trí ô đang gán.
            domain: Tập giá trị hợp lệ của ô.
            state: Ma trận hiện tại.
            domains: Từ điển chứa tập giá trị hợp lệ.
            assigned: Tập các giá trị đã gán.
            Trả về: Danh sách giá trị sắp xếp theo số giá trị bị loại bỏ (ít nhất đến nhiều nhất).
            """
            value_scores = []
            for value in domain:
                temp_state = [row[:] for row in state]
                temp_state[pos[0]][pos[1]] = value
                _, new_domains = forward_check(temp_state, pos, value, domains, assigned)
                eliminated = sum(len(domains[p]) - len(new_domains[p]) for p in new_domains if p != pos)
                if value == self.goal[pos[0]][pos[1]]:  # Ưu tiên giá trị trong goal_state
                    eliminated -= 10
                value_scores.append((eliminated, value))
            value_scores.sort()
            return [value for _, value in value_scores]

        def backtrack_with_fc(state, assigned, positions, domains):
            """
            Hàm đệ quy để thực hiện backtracking với Forward Checking.
            state: Ma trận hiện tại (có thể chứa None).
            assigned: Tập các vị trí và giá trị đã gán.
            positions: Danh sách các ô chưa gán.
            domains: Từ điển chứa tập giá trị hợp lệ cho các ô.
            """
            if len(assigned) == 9:  # Đã gán hết 9 ô
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            # Kiểm tra sớm trạng thái mục tiêu khi gán từ 7 ô trở lên
            if len(assigned) >= 7:
                temp_state = [row[:] for row in state]
                temp_assigned = assigned.copy()
                temp_positions = [p for p in positions if p not in assigned]
                temp_domains = {k: v[:] for k, v in domains.items()}
                for p in temp_positions:
                    remaining_values = [v for v in range(9) if v not in temp_assigned.values()]
                    if not remaining_values:
                        return None
                    value = remaining_values[0]  # Chọn giá trị đầu tiên
                    temp_state[p[0]][p[1]] = value
                    temp_assigned[p] = value
                    temp_tuple = tuple(tuple(row) for row in temp_state)
                    path.append(temp_tuple)  # Thêm trạng thái trung gian
                    success, temp_domains = forward_check(temp_state, p, value, temp_domains, temp_assigned)
                    if not success:
                        path.pop()
                        return None
                state_tuple = tuple(tuple(row) for row in temp_state)
                if state_tuple == self.goal and self.is_solvable(temp_state):
                    return path
                path.pop(len(temp_positions))  # Xóa các trạng thái trung gian nếu thất bại
                return None

            # Chọn ô có ít giá trị hợp lệ nhất (MRV)
            pos = select_mrv_variable(positions, domains, state)
            if pos is None:
                return None

            # Lấy tập giá trị hợp lệ và sắp xếp theo LCV
            domain = get_domain(state, pos, set(assigned.values()))
            sorted_values = select_lcv_value(pos, domain, state, domains, assigned)

            # Tạo bản sao trạng thái
            state_tuple = tuple(tuple(row if row is not None else (None, None, None)) for row in state)
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            explored_states.append(state_tuple)

            # Thử gán các giá trị theo thứ tự LCV
            for value in sorted_values:
                new_state = [row[:] for row in state]
                new_state[pos[0]][pos[1]] = value
                new_assigned = assigned.copy()
                new_assigned[pos] = value
                new_positions = [p for p in positions if p != pos]
                path.append(state_tuple)  # Thêm trạng thái trước khi gán

                # Thực hiện Forward Checking
                success, new_domains = forward_check(new_state, pos, value, domains, new_assigned)
                if success:
                    result = backtrack_with_fc(new_state, new_assigned, new_positions, new_domains)
                    if result is not None:
                        return result
                path.pop()  # Quay lui: xóa trạng thái nếu không thành công

            return None

        # Khởi tạo ma trận rỗng và tập giá trị ban đầu
        empty_state = [[None for _ in range(3)] for _ in range(3)]
        positions = [(i, j) for i in range(3) for j in range(3)]
        domains = {(i, j): list(range(9)) for i in range(3) for j in range(3)}
        assigned = {}
        result = backtrack_with_fc(empty_state, assigned, positions, domains)
        return result, explored_states

    def min_conflicts_search(self, max_iterations=1000, max_no_improvement=20, timeout=2.0):
        """
        Optimized Min-Conflicts Search for CSP starting from an empty state.

        Args:
            max_iterations (int): Maximum number of iterations.
            max_no_improvement (int): Maximum iterations without improvement before restart.
            timeout (float): Maximum running time in seconds.

        Returns:
            tuple: (path, num_explored_states) if solution found,
                   (None, num_explored_states) otherwise.
        """

        def count_conflicts(state):
            """
            Count conflicts with optimized weights and Manhattan heuristic.
            """
            conflicts = 0
            manhattan_distance = 0
            value_counts = defaultdict(int)

            # Constraint: (0,0) must be 1
            if state[0][0] is not None and state[0][0] != 1:
                conflicts += 5

            # Constraint: Each number appears once
            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    if val is not None:
                        value_counts[val] += 1
                        if value_counts[val] > 1:
                            conflicts += value_counts[val] - 1
                        # Manhattan distance
                        if val != 0 and val != self.goal[i][j]:
                            goal_i, goal_j = divmod(val - 1, 3)
                            manhattan_distance += abs(i - goal_i) + abs(j - goal_j)

            # Row constraint: state[i][j+1] = state[i][j] + 1 (except blank)
            for i in range(3):
                for j in range(2):
                    if (state[i][j] is not None and state[i][j] != 0 and
                            state[i][j + 1] is not None and state[i][j + 1] != 0):
                        if state[i][j + 1] != state[i][j] + 1:
                            conflicts += 1

            # Column constraint: state[i+1][j] = state[i][j] + 3 (except blank)
            for j in range(3):
                for i in range(2):
                    if (state[i][j] is not None and state[i][j] != 0 and
                            state[i + 1][j] is not None and state[i + 1][j] != 0):
                        if state[i + 1][j] != state[i][j] + 3:
                            conflicts += 1

            # Solvability constraint (only check if state is complete)
            if all(state[i][j] is not None for i in range(3) for j in range(3)) and not self.is_solvable(state):
                conflicts += 5

            # Add Manhattan heuristic
            conflicts += manhattan_distance * 0.3
            return conflicts

        def select_position(state, assigned_positions):
            """Select unassigned position or position with highest conflicts."""
            # Prefer unassigned positions
            for i in range(3):
                for j in range(3):
                    if (i, j) not in assigned_positions:
                        return i, j
            # If all assigned, select position with highest conflicts
            conflicts = []
            for i in range(3):
                for j in range(3):
                    temp_state = [row[:] for row in state]
                    original_value = temp_state[i][j]
                    temp_state[i][j] = None
                    base_conflicts = count_conflicts(tuple(tuple(row) for row in temp_state))
                    temp_state[i][j] = original_value
                    current_conflicts = count_conflicts(tuple(tuple(row) for row in temp_state))
                    if current_conflicts > base_conflicts:
                        conflicts.append((i, j, current_conflicts))
            if conflicts:
                return max(conflicts, key=lambda x: x[2])[:2]
            return None

        def select_value(state, i, j, current_value, assigned_values):
            """Select value with least conflicts, prioritizing goal state value."""
            value_scores = []
            for value in range(9):
                if value not in assigned_values - ({current_value} if current_value is not None else set()):
                    if (i, j) == (0, 0) and value != 1:
                        continue
                    temp_state = [row[:] for row in state]
                    temp_state[i][j] = value
                    conflicts = count_conflicts(tuple(tuple(row) for row in temp_state))
                    if value == self.goal[i][j]:
                        conflicts -= 3  # Prioritize goal state value
                    value_scores.append((conflicts, value))
            if not value_scores:
                return None
            value_scores.sort()
            return value_scores[0][1]

        start_time = time.time()
        current_state = [[None for _ in range(3)] for _ in range(3)]  # Empty state
        path = [tuple(tuple(None if x is None else x for x in row) for row in current_state)]
        num_explored_states = 1
        assigned_values = set()
        assigned_positions = set()
        best_conflicts = float('inf')
        best_state = current_state
        no_improvement_count = 0
        temperature = 5.0  # Reduced initial temperature

        # Initialize (0,0) with 1 to satisfy constraint
        current_state[0][0] = 1
        assigned_values.add(1)
        assigned_positions.add((0, 0))
        path.append(tuple(tuple(None if x is None else x for x in row) for row in current_state))
        num_explored_states += 1

        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                print("Timeout reached")
                return None, num_explored_states

            current_state_tuple = tuple(tuple(None if x is None else x for x in row) for row in current_state)
            if current_state_tuple == self.goal and all(
                    current_state[i][j] is not None for i in range(3) for j in range(3)):
                print(f"Solution found after {iteration} iterations")
                return path, num_explored_states

            conflicts = count_conflicts(current_state_tuple)
            if conflicts == 0 and all(current_state[i][j] is not None for i in range(3) for j in range(3)):
                if self.is_solvable(current_state):
                    print(f"Solution found after {iteration} iterations")
                    return path, num_explored_states

            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_state = [row[:] for row in current_state]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_no_improvement:
                current_state = [[None for _ in range(3)] for _ in range(3)]
                assigned_values = set()
                assigned_positions = set()
                current_state[0][0] = 1
                assigned_values.add(1)
                assigned_positions.add((0, 0))
                current_state_tuple = tuple(tuple(None if x is None else x for x in row) for row in current_state)
                path.append(current_state_tuple)
                num_explored_states += 1
                conflicts = count_conflicts(current_state_tuple)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_state = [row[:] for row in current_state]
                no_improvement_count = 0
                temperature = 5.0
                continue

            pos = select_position(current_state, assigned_positions)
            if not pos:
                current_state = [[None for _ in range(3)] for _ in range(3)]
                assigned_values = set()
                assigned_positions = set()
                current_state[0][0] = 1
                assigned_values.add(1)
                assigned_positions.add((0, 0))
                current_state_tuple = tuple(tuple(None if x is None else x for x in row) for row in current_state)
                path.append(current_state_tuple)
                num_explored_states += 1
                continue

            i, j = pos
            current_value = current_state[i][j]

            # Simulated Annealing with lower probability for random choices
            if random.random() < min(0.2, temperature / 20.0):
                remaining = [v for v in range(9) if
                             v not in assigned_values - ({current_value} if current_value is not None else set())]
                if remaining and (i != 0 or j != 0 or 1 in remaining):
                    value = random.choice(remaining)
                else:
                    continue
            else:
                value = select_value(current_state, i, j, current_value, assigned_values)
                if value is None:
                    continue

            current_state_list = [row[:] for row in current_state]
            current_state_list[i][j] = value
            if current_value is not None:
                assigned_values.remove(current_value)
                assigned_positions.remove((i, j))
            assigned_values.add(value)
            assigned_positions.add((i, j))
            current_state = current_state_list
            current_state_tuple = tuple(tuple(None if x is None else x for x in row) for row in current_state)
            path.append(current_state_tuple)
            num_explored_states += 1

            temperature *= 0.99  # Faster cooling

        if tuple(tuple(row) for row in best_state) == self.goal and self.is_solvable(best_state):
            print("Returning best state as solution")
            return path, num_explored_states
        print("No solution found")
        return None, num_explored_states

    # Định nghĩa q_table toàn cục
    q_table = defaultdict(lambda: {a: 0.0 for a in range(4)})  # {state: {action: q_value}}

    # Hàm phụ
    @staticmethod
    def get_action_from_direction(dx, dy):
        """Chuyển đổi hướng di chuyển thành hành động (0: lên, 1: xuống, 2: phải, 3: trái)."""
        if dx == 0 and dy == -1:
            return 0  # Lên
        elif dx == 0 and dy == 1:
            return 1  # Xuống
        elif dx == 1 and dy == 0:
            return 2  # Phải
        elif dx == -1 and dy == 0:
            return 3  # Trái
        return None

    @staticmethod
    def get_direction_from_action(action):
        """Chuyển đổi hành động thành hướng di chuyển (dx, dy)."""
        if action == 0:  # Lên
            return 0, -1
        elif action == 1:  # Xuống
            return 0, 1
        elif action == 2:  # Phải
            return 1, 0
        elif action == 3:  # Trái
            return -1, 0
        return 0, 0

    def q_learning_search(self, max_episodes=1000, max_steps=100):
        """
        Q-Learning Search cho 8-puzzle: Học chính sách di chuyển ô trống để đạt trạng thái mục tiêu.
        max_episodes: Số tập tối đa.
        max_steps: Số bước tối đa mỗi tập.
        Trả về: (path, states_explored) hoặc (None, states_explored) nếu không tìm thấy giải pháp.
        """
        # Kiểm tra tính solvable của initial_state
        if not self.is_solvable(self.initial):
            return None, 0

        alpha = 0.2  # Tỷ lệ học
        gamma = 0.9  # Hệ số chiết khấu
        epsilon = 0.3  # Tỷ lệ khám phá ban đầu
        convergence_threshold = 0.1
        states_explored = 0
        path = [self.initial]

        def hamming_distance(state):
            """Tính số ô không đúng vị trí so với goal_state (trừ ô trống)."""
            distance = 0
            for i in range(3):
                for j in range(3):
                    if state[i][j] != 0 and state[i][j] != self.goal[i][j]:
                        distance += 1
            return distance

        def get_neighbors(state):
            """Lấy các trạng thái lân cận bằng cách di chuyển ô trống."""
            i, j = None, None
            for r in range(3):
                for c in range(3):
                    if state[r][c] == 0:
                        i, j = r, c
                        break
            neighbors = []
            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Lên, xuống, phải, trái
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    new_state = [list(row) for row in state]
                    new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                    neighbors.append(tuple(tuple(row) for row in new_state))
            return neighbors

        for episode in range(max_episodes):
            current_state = self.initial
            max_delta = 0
            episode_states_explored = 0
            epsilon = max(0.05, epsilon * 0.995)  # Giảm epsilon dần

            for step in range(max_steps):
                state_tuple = current_state

                # Epsilon-greedy: Đảm bảo action luôn được gán giá trị
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    if not self.q_table[state_tuple]:  # Nếu dictionary rỗng
                        action = random.randint(0, 3)  # Chọn ngẫu nhiên
                    else:
                        action = max(self.q_table[state_tuple], key=self.q_table[state_tuple].get)

                # Thực hiện hành động
                dx, dy = self.get_direction_from_action(action)
                i, j = None, None
                for r in range(3):
                    for c in range(3):
                        if current_state[r][c] == 0:
                            i, j = r, c
                            break
                next_i, next_j = i + dy, j + dx  # dx, dy tương ứng với thay đổi cột, hàng
                neighbors = get_neighbors(current_state)

                # Tính phần thưởng
                if not (0 <= next_i < 3 and 0 <= next_j < 3):
                    reward = -10  # Phạt nếu di chuyển không hợp lệ
                    next_state = current_state
                else:
                    next_state = [list(row) for row in current_state]
                    next_state[i][j], next_state[next_i][next_j] = next_state[next_i][next_j], next_state[i][j]
                    next_state = tuple(tuple(row) for row in next_state)
                    if next_state not in neighbors:
                        reward = -10
                        next_state = current_state
                    else:
                        # Tính phần thưởng dựa trên số ô sai vị trí
                        distance_before = hamming_distance(current_state)
                        distance_after = hamming_distance(next_state)
                        reward = -0.5 + (distance_before - distance_after) * 5
                        if next_state == self.goal:
                            reward = 100

                # Cập nhật Q-value
                next_state_tuple = next_state
                old_value = self.q_table[state_tuple][action]
                max_future_q = max(self.q_table[next_state_tuple].values()) if self.q_table[next_state_tuple] else 0.0
                self.q_table[state_tuple][action] = old_value + alpha * (reward + gamma * max_future_q - old_value)
                max_delta = max(max_delta, abs(old_value - self.q_table[state_tuple][action]))

                states_explored += 1
                episode_states_explored += 1
                current_state = next_state

                if current_state == self.goal:
                    break

            # Kiểm tra hội tụ
            if max_delta < convergence_threshold:
                print(f"Q-Learning converged after {episode + 1} episodes")
                break

        # Trích xuất đường đi
        path = [self.initial]
        current_state = self.initial
        visited = set([current_state])
        steps = 0
        while current_state != self.goal and steps < max_steps:
            state_tuple = current_state
            if state_tuple not in self.q_table or not self.q_table[state_tuple]:
                break
            action = max(self.q_table[state_tuple], key=self.q_table[state_tuple].get)
            dx, dy = self.get_direction_from_action(action)
            i, j = None, None
            for r in range(3):
                for c in range(3):
                    if current_state[r][c] == 0:
                        i, j = r, c
                        break
            next_i, next_j = i + dy, j + dx
            if not (0 <= next_i < 3 and 0 <= next_j < 3):
                break
            next_state = [list(row) for row in current_state]
            next_state[i][j], next_state[next_i][next_j] = next_state[next_i][next_j], next_state[i][j]
            next_state = tuple(tuple(row) for row in next_state)
            if next_state not in get_neighbors(current_state) or next_state in visited:
                break
            visited.add(next_state)
            path.append(next_state)
            current_state = next_state
            steps += 1

        if current_state == self.goal:
            return path, states_explored
        return None, states_explored


# Hàm chọn trạng thái ban đầu bằng Pygame
def initial_state_selector(goal_state):
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Initial State Selector")

    try:
        background = pygame.image.load(r"asset\anh_backgound\map3.png")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except pygame.error:
        print("Cannot find background image, using white background")
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))

    # Sử dụng font Comic Sans MS (phong cách vui nhộn, gần giống Anime Ace)
    title_font = pygame.font.SysFont("Comic Sans MS", 70, bold=True)  # Font lớn hơn cho tiêu đề
    label_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)  # Font cho nhãn và nút
    error_font = pygame.font.SysFont("Comic Sans MS", 40, bold=True)  # Font cho thông báo lỗi

    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    tile_size = 100
    grid_offset_x = 250
    grid_offset_y = 300
    goal_offset_x = 650
    goal_offset_y = 300
    selected_cell = None

    button_right_rect = pygame.Rect(300, 620, 200, 50)
    button_confirm_rect = pygame.Rect(700, 620, 200, 50)

    def draw_grid(state, offset_x, offset_y, tile_size, selected=None):
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
                if selected == (i, j):
                    pygame.draw.rect(screen, (255, 255, 0), rect)
                else:
                    pygame.draw.rect(screen, (33, 172, 255), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                if state[i][j] != 0:
                    text = label_font.render(str(state[i][j]), True, (228, 61, 48))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def is_valid_state(state):
        flat = [num for row in state for num in row]
        return sorted(flat) == list(range(9))

    # Hàm vẽ chữ với viền đen
    def render_text_with_border(surface, text, font, color, border_color, pos):
        text_surface = font.render(text, True, color)
        border_surface = font.render(text, True, border_color)
        text_rect = text_surface.get_rect(center=pos)

        # Vẽ viền bằng cách vẽ nhiều lần lệch vị trí
        offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]
        for dx, dy in offsets:
            border_rect = text_rect.copy()
            border_rect.x += dx
            border_rect.y += dy
            surface.blit(border_surface, border_rect)

        surface.blit(text_surface, text_rect)

    puzzle_temp = EightPuzzle(initial_state, goal_state)

    running = True
    while running:
        screen.blit(background, (0, 0))

        # Tiêu đề
        render_text_with_border(screen, "Select State", title_font, (255, 0, 0), (0, 0, 0), (WIDTH // 2, 50))

        # Vẽ lưới trạng thái ban đầu
        render_text_with_border(screen, "Initial State", label_font, (0, 255, 0), (0, 0, 0),
                                (grid_offset_x + 150, grid_offset_y - 60))  # 150 = 300/2 (chiều rộng ma trận)
        draw_grid(initial_state, grid_offset_x, grid_offset_y, tile_size, selected_cell)

        # Vẽ lưới trạng thái mục tiêu
        render_text_with_border(screen, "Goal State", label_font, (0, 255, 0), (0, 0, 0),
                                (goal_offset_x + 150, goal_offset_y - 60))
        draw_grid(goal_state, goal_offset_x, goal_offset_y, tile_size)

        # Vẽ các nút

        render_text_with_border(screen, "->", label_font, (255, 255, 255), (0, 0, 0),
                                (button_right_rect.centerx, button_right_rect.centery ))
        render_text_with_border(screen, "Confirm", label_font, (255, 255, 255), (0, 0, 0),
                                (button_confirm_rect.centerx, button_confirm_rect.centery ))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for i in range(3):
                    for j in range(3):
                        rect = pygame.Rect(grid_offset_x + j * tile_size, grid_offset_y + i * tile_size, tile_size,
                                           tile_size)
                        if rect.collidepoint(mouse_pos):
                            selected_cell = (i, j)
                            break
                if button_right_rect.collidepoint(mouse_pos):
                    initial_state = list(map(list, puzzle_temp.generate_random_state()))
                    selected_cell = None
                if button_confirm_rect.collidepoint(mouse_pos):
                    if is_valid_state(initial_state) and puzzle_temp.is_solvable(initial_state):
                        pygame.quit()
                        return initial_state
                    else:
                        render_text_with_border(screen, "Invalid State!", error_font, (255, 0, 0), (0, 0, 0),
                                                (WIDTH // 2, 600))
                        pygame.display.flip()
                        pygame.time.delay(1000)

            elif event.type == pygame.KEYDOWN and selected_cell:
                i, j = selected_cell
                if event.unicode.isdigit() and 0 <= int(event.unicode) <= 8:
                    initial_state[i][j] = int(event.unicode)
                elif event.key == pygame.K_BACKSPACE:
                    initial_state[i][j] = 0

    pygame.quit()
    return initial_state

def plot_performance(performance_history):
    """Tạo biểu đồ cột cho States Explored và Runtime bằng Plotly với số ở phía trên cột."""
    algorithms = []
    avg_states_explored = []
    avg_runtimes = []

    for algo, runs in performance_history.items():
        if runs:  # Chỉ xử lý thuật toán đã chạy ít nhất 1 lần
            algorithms.append(algo)
            states = [run["states_explored"] for run in runs]
            states = [len(s) if isinstance(s, list) else s for s in states]
            avg_states_explored.append(sum(states) / len(states) if states else 0)
            runtimes = [run["runtime"] for run in runs]
            avg_runtimes.append(sum(runtimes) / len(runtimes) if runtimes else 0)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("States Explored", "Runtime"))

    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=avg_states_explored,
            name="States Explored",
            marker_color="lightblue",
            text=[f"{val:.0f}" for val in avg_states_explored],
            textposition="auto",
            textfont=dict(size=14, family="Arial", weight="bold"),  # In đậm
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="States Explored", row=1, col=1, tickfont=dict(weight="bold"))  # In đậm trục y

    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=avg_runtimes,
            name="Runtime (ms)",
            marker_color="lightcoral",
            text=[f"{val:.2f}" for val in avg_runtimes],
            textposition="auto",
            textfont=dict(size=14, family="Arial", weight="bold"),  # In đậm
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1, tickfont=dict(weight="bold"))  # In đậm trục y

    fig.update_xaxes(tickfont=dict(weight="bold"))  # In đậm tên thuật toán
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Performance Comparison",
        title_font=dict(size=16, weight="bold"),  # In đậm tiêu đề
        showlegend=False,
        bargap=0.2,
    )
    fig.show()


def main_game(initial_state, goal_state):
    """
    Main function to run the 8-puzzle solver interface.

    Args:
        initial_state: Initial state of the puzzle.
        goal_state: Goal state of the puzzle.

    Returns:
        str: "BACK" if BACK button is pressed, None otherwise.
    """
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Solver")

    try:
        background = pygame.image.load(r"asset\anh_backgound\map5.png")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except pygame.error:
        print("Cannot find background image, using white background")
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))

    # Font chữ
    title_font = pygame.font.SysFont("Comic Sans MS", 70, bold=True)
    label_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
    button_font = pygame.font.SysFont("Comic Sans MS", 20, bold=True)
    info_font = pygame.font.SysFont("Comic Sans MS", 25, bold=True)
    number_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)

    puzzle = EightPuzzle(initial_state, goal_state)

    # Vị trí và kích thước lưới trạng thái
    small_tile_size = 40
    initial_grid_x, initial_grid_y = 750, 120
    goal_grid_x, goal_grid_y = 950, 120

    # Vùng thuật toán chạy (bên trái, lớn)
    algo_tile_size = 150
    algo_grid_x, algo_grid_y = 50, 100

    # Nút thuật toán
    button_width, button_height = 100, 40
    button_spacing_x, button_spacing_y = 110, 50
    start_x, start_y = 750, 300
    buttons = [
        ("BFS", pygame.Rect(start_x, start_y, button_width, button_height)),
        ("DFS", pygame.Rect(start_x + button_spacing_x, start_y, button_width, button_height)),
        ("UCS", pygame.Rect(start_x + 2 * button_spacing_x, start_y, button_width, button_height)),
        ("IDS", pygame.Rect(start_x + 3 * button_spacing_x, start_y, button_width, button_height)),
        ("Greedy", pygame.Rect(start_x, start_y + button_spacing_y, button_width, button_height)),
        ("A*", pygame.Rect(start_x + button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("IDA*", pygame.Rect(start_x + 2 * button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("SHC", pygame.Rect(start_x + 3 * button_spacing_x, start_y + button_spacing_y, button_width, button_height)),
        ("SAHC", pygame.Rect(start_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("RHC", pygame.Rect(start_x + button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("SAS",
         pygame.Rect(start_x + 2 * button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("Beam",
         pygame.Rect(start_x + 3 * button_spacing_x, start_y + 2 * button_spacing_y, button_width, button_height)),
        ("Genetic", pygame.Rect(start_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        (
        "AND-OR", pygame.Rect(start_x + button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Belief",
         pygame.Rect(start_x + 2 * button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("POS",
         pygame.Rect(start_x + 3 * button_spacing_x, start_y + 3 * button_spacing_y, button_width, button_height)),
        ("Backtrack", pygame.Rect(start_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("Forward",
         pygame.Rect(start_x + button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("MinConf",
         pygame.Rect(start_x + 2 * button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height)),
        ("QLearn",
         pygame.Rect(start_x + 3 * button_spacing_x, start_y + 4 * button_spacing_y, button_width, button_height))
    ]

    # Nút BACK, RESET, VIEW và INFO ở góc dưới bên phải
    back_button_rect = pygame.Rect(WIDTH - 120, HEIGHT - 70, 100, 40)
    reset_button_rect = pygame.Rect(WIDTH - 230, HEIGHT - 70, 100, 40)
    view_button_rect = pygame.Rect(WIDTH - 340, HEIGHT - 70, 100, 40)
    info_button_rect = pygame.Rect(WIDTH - 450, HEIGHT - 70, 100, 40)  # Nút Info bên trái View

    # Lưu trữ hiệu suất thuật toán
    performance_history = {
        "BFS": [], "DFS": [], "UCS": [], "IDS": [], "Greedy": [], "A*": [], "IDA*": [],
        "SHC": [], "SAHC": [], "RHC": [], "SAS": [], "Beam": [], "Genetic": [], "AND-OR": [],
        "Belief": [], "POS": [], "Backtrack": [], "Forward": [], "MinConf": [], "QLearn": []
    }

    solution = None
    solution_index = 0
    elapsed_time = 0
    steps = 0
    error_message = None
    error_timer = 0
    selected_button = None
    display_state = initial_state  # Trạng thái hiển thị trước khi chạy thuật toán

    def draw_grid(state, offset_x, offset_y, tile_size):
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
                if state[i][j] == 0 or state[i][j] is None:
                    pygame.draw.rect(screen, (255, 255, 255), rect)
                else:
                    pygame.draw.rect(screen, (33, 172, 255), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                if state[i][j] is not None and state[i][j] != 0:
                    num_font = number_font if tile_size > 50 else pygame.font.SysFont("Comic Sans MS", 30, bold=True)
                    text = num_font.render(str(state[i][j]), True, (228, 61, 48))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def render_text_with_border(surface, text, font, color, border_color, pos):
        text_surface = font.render(text, True, color)
        border_surface = font.render(text, True, border_color)
        text_rect = text_surface.get_rect(center=pos)

        offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]
        for dx, dy in offsets:
            border_rect = text_rect.copy()
            border_rect.x += dx
            border_rect.y += dy
            surface.blit(border_surface, border_rect)

        surface.blit(text_surface, text_rect)

    def display_algorithm_info(performance_history):
        """Ghi thông tin chi tiết về các thuật toán đã chạy vào file txt."""
        with open("algorithm_info.txt", "w") as file:
            file.write("=== Algorithm Performance Info ===\n\n")
            for algo, runs in performance_history.items():
                if not runs:
                    continue
                file.write(f"Algorithm: {algo}\n")
                for i, run in enumerate(runs, 1):
                    file.write(f"  Run {i}:\n")
                    file.write(f"    Steps: {run['steps']}\n")
                    file.write(f"    States Explored: {run['states_explored']}\n")
                    file.write(f"    Runtime: {run['runtime']:.2f} ms\n")
                    file.write(f"    Path: {'Found' if run['path'] else 'Not Found'}\n")
                    if run['path']:
                        file.write("    Path Details:\n")
                        for j, state in enumerate(run['path'][:5]):  # Giới hạn 5 trạng thái đầu để tóm tắt
                            file.write(f"      State {j + 1}: {state}\n")
                        if len(run['path']) > 5:
                            file.write(f"      ... (Total {len(run['path'])} states)\n")
                file.write("\n")
            file.write("================================\n")

    running = True
    clock = pygame.time.Clock()
    while running:
        screen.blit(background, (0, 0))

        render_text_with_border(screen, "8-Puzzle Solver", title_font, (255, 0, 0), (0, 0, 0), (WIDTH // 2, 50))

        render_text_with_border(screen, "Initial State", label_font, (0, 255, 0), (0, 0, 0),
                                (initial_grid_x + (small_tile_size * 3) // 2, initial_grid_y - 20))
        draw_grid(initial_state, initial_grid_x, initial_grid_y, small_tile_size)

        render_text_with_border(screen, "Goal State", label_font, (0, 255, 0), (0, 0, 0),
                                (goal_grid_x + (small_tile_size * 3) // 2, goal_grid_y - 20))
        draw_grid(goal_state, goal_grid_x, goal_grid_y, small_tile_size)

        # Hiển thị trạng thái thuật toán
        if solution:
            if solution_index < len(solution):
                draw_grid(solution[solution_index], algo_grid_x, algo_grid_y, algo_tile_size)
                solution_index += 1
                pygame.time.delay(500)
            else:
                draw_grid(solution[-1], algo_grid_x, algo_grid_y, algo_tile_size)
        else:
            draw_grid(display_state, algo_grid_x, algo_grid_y, algo_tile_size)

        render_text_with_border(screen, f"Running Time: {elapsed_time:.2f} ms", info_font, (0, 128, 0), (0, 0, 0),
                                (algo_grid_x + 225, algo_grid_y + 500))
        render_text_with_border(screen, f"Steps: {steps}", info_font, (0, 128, 0), (0, 0, 0),
                                (algo_grid_x + 225, algo_grid_y + 550))

        if error_message and pygame.time.get_ticks() - error_timer < 1000:
            render_text_with_border(screen, "No Solution Found!", info_font, (255, 0, 0), (0, 0, 0),
                                    (algo_grid_x + 225, algo_grid_y + 575))
        else:
            error_message = None

        for label, rect in buttons:
            corner_radius = 10
            button_color = (255, 255, 0) if selected_button == label else (26, 125, 255)
            pygame.draw.rect(screen, button_color, rect, border_radius=corner_radius)
            pygame.draw.rect(screen, (41, 128, 185), rect, 2, border_radius=corner_radius)
            render_text_with_border(screen, label, button_font, (255, 255, 255), (0, 0, 0),
                                    (rect.centerx, rect.centery))

        corner_radius = 10
        # Vẽ nút Info
        info_button_color = (255, 255, 0) if selected_button == "INFO" else (26, 125, 255)
        pygame.draw.rect(screen, info_button_color, info_button_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), info_button_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "INFO", button_font, (255, 255, 255), (0, 0, 0),
                                (info_button_rect.centerx, info_button_rect.centery))

        view_button_color = (255, 255, 0) if selected_button == "VIEW" else (26, 125, 255)
        pygame.draw.rect(screen, view_button_color, view_button_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), view_button_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "VIEW", button_font, (255, 255, 255), (0, 0, 0),
                                (view_button_rect.centerx, view_button_rect.centery))

        reset_button_color = (255, 255, 0) if selected_button == "RESET" else (26, 125, 255)
        pygame.draw.rect(screen, reset_button_color, reset_button_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), reset_button_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "RESET", button_font, (255, 255, 255), (0, 0, 0),
                                (reset_button_rect.centerx, reset_button_rect.centery))

        back_button_color = (255, 255, 0) if selected_button == "BACK" else (26, 125, 255)
        pygame.draw.rect(screen, back_button_color, back_button_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), back_button_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "BACK", button_font, (255, 255, 255), (0, 0, 0),
                                (back_button_rect.centerx, back_button_rect.centery))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if info_button_rect.collidepoint(mouse_pos):
                    selected_button = "INFO"
                    pygame.display.flip()
                    pygame.time.delay(200)
                    display_algorithm_info(performance_history)  # Ghi thông tin vào file txt
                    import platform
                    import os
                    file_path = "algorithm_info.txt"
                    try:
                        if platform.system() == "Windows":
                            os.startfile(file_path)
                        elif platform.system() == "Darwin":  # macOS
                            os.system(f"open {file_path}")
                        else:  # Linux và các hệ điều hành khác
                            os.system(f"xdg-open {file_path}")
                    except Exception as e:
                        print(f"Error opening file: {e}")
                    continue

                if view_button_rect.collidepoint(mouse_pos):
                    selected_button = "VIEW"
                    pygame.display.flip()
                    pygame.time.delay(200)
                    plot_performance(performance_history)
                    continue

                if reset_button_rect.collidepoint(mouse_pos):
                    selected_button = "RESET"
                    solution = None
                    solution_index = 0
                    elapsed_time = 0
                    steps = 0
                    error_message = None
                    display_state = initial_state
                    pygame.display.flip()
                    pygame.time.delay(200)
                    continue

                if back_button_rect.collidepoint(mouse_pos):
                    selected_button = "BACK"
                    pygame.display.flip()
                    pygame.time.delay(200)
                    pygame.quit()
                    return "BACK"

                for label, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        selected_button = label
                        solution = None
                        solution_index = 0
                        elapsed_time = 0
                        steps = 0
                        error_message = None
                        start_time = timeit.default_timer()

                        # Set display_state based on algorithm
                        if label in ["MinConf", "Backtrack", "Forward"]:
                            display_state = [[None, None, None], [None, None, None], [None, None, None]]
                            draw_grid(display_state, algo_grid_x, algo_grid_y, algo_tile_size)
                            pygame.display.flip()
                            pygame.time.delay(200)
                        else:
                            display_state = initial_state

                        if label == "MinConf":
                            puzzle = EightPuzzle([[None, None, None], [None, None, None], [None, None, None]],
                                                 goal_state)
                            solution, num_explored_states = puzzle.min_conflicts_search()
                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history["MinConf"].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": num_explored_states,
                                "path": solution if solution else []
                            })
                        elif label == "QLearn":
                            puzzle = EightPuzzle(initial_state, goal_state)
                            solution, states_explored = puzzle.q_learning_search()
                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history["QLearn"].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": states_explored if states_explored is not None else 0,
                                "path": solution if solution else []
                            })
                        elif label == "Backtrack":
                            puzzle = EightPuzzle([[None, None, None], [None, None, None], [None, None, None]],
                                                 goal_state)
                            solution, explored_states = puzzle.backtracking_search()
                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history["Backtrack"].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": explored_states if explored_states is not None else 0,
                                "path": solution if solution else []
                            })
                        elif label == "Forward":
                            puzzle = EightPuzzle([[None, None, None], [None, None, None], [None, None, None]],
                                                 goal_state)
                            solution, explored_states = puzzle.forward_checking_search()
                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history["Forward"].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": explored_states if explored_states is not None else 0,
                                "path": solution if solution else []
                            })
                        elif label in ["BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*", "SHC", "SAHC", "RHC",
                                       "SAS", "Beam", "Genetic", "AND-OR"]:
                            puzzle = EightPuzzle(initial_state, goal_state)
                            if label == "BFS":
                                solution, explored_states = puzzle.bfs()
                            elif label == "DFS":
                                solution, explored_states = puzzle.dfs()
                            elif label == "UCS":
                                solution, explored_states = puzzle.ucs()
                            elif label == "IDS":
                                solution, explored_states = puzzle.ids()
                            elif label == "Greedy":
                                solution, explored_states = puzzle.greedy_search()
                            elif label == "A*":
                                solution, explored_states = puzzle.a_star()
                            elif label == "IDA*":
                                solution, explored_states = puzzle.ida_star()
                            elif label == "SHC":
                                solution, explored_states = puzzle.simple_hill_climbing()
                            elif label == "SAHC":
                                solution, explored_states = puzzle.steepest_ascent_hill_climbing()
                            elif label == "RHC":
                                solution, explored_states = puzzle.random_hill_climbing()
                            elif label == "SAS":
                                solution, explored_states = puzzle.simulated_annealing()
                            elif label == "Beam":
                                solution, explored_states = puzzle.beam_search()
                            elif label == "Genetic":
                                solution, explored_states = puzzle.genetic_algorithm()
                            elif label == "AND-OR":
                                solution, explored_states = puzzle.and_or_search()
                            elapsed_time = (timeit.default_timer() - start_time) * 1000
                            steps = len(solution) - 1 if solution else 0
                            if not solution:
                                error_message = True
                                error_timer = pygame.time.get_ticks()
                            performance_history[label].append({
                                "runtime": elapsed_time,
                                "steps": steps,
                                "states_explored": explored_states if explored_states is not None else 0,
                                "path": solution if solution else []
                            })
                        elif label == "Belief":
                            print("Calling belief_state_interface for Belief")
                            initial_state_tuple = tuple(tuple(row) for row in initial_state)
                            initial_belief = {initial_state_tuple}
                            neighbors = puzzle.get_neighbors(initial_state_tuple)
                            for neighbor in neighbors[:5]:
                                initial_belief.add(neighbor)
                            while len(initial_belief) < 6:
                                random_state = puzzle.generate_random_state()
                                if random_state not in initial_belief:
                                    initial_belief.add(random_state)
                            initial_belief = list(initial_belief)
                            result = belief_state_interface(initial_belief, goal_state, "Belief", performance_history)
                            print(f"Returned from belief_state_interface: {result}")
                            if result == "BACK":
                                selected_button = None
                                solution = None
                                solution_index = 0
                                elapsed_time = 0
                                steps = 0
                                error_message = None
                                display_state = initial_state
                                pygame.init()
                                screen = pygame.display.set_mode((WIDTH, HEIGHT))
                                pygame.display.set_caption("8-Puzzle Solver")
                                try:
                                    background = pygame.image.load(r"asset\anh_backgound\map5.png")
                                    background = pygame.transform.scale(background, (WIDTH, HEIGHT))
                                except pygame.error:
                                    print("Cannot find background image, using white background")
                                    background = pygame.Surface((WIDTH, HEIGHT))
                                    background.fill((255, 255, 255))
                                title_font = pygame.font.SysFont("Comic Sans MS", 70, bold=True)
                                label_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
                                button_font = pygame.font.SysFont("Comic Sans MS", 20, bold=True)
                                info_font = pygame.font.SysFont("Comic Sans MS", 25, bold=True)
                                number_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)
                                screen.blit(background, (0, 0))
                                render_text_with_border(screen, "8-Puzzle Solver", title_font, (255, 0, 0), (0, 0, 0),
                                                        (WIDTH // 2, 50))
                                render_text_with_border(screen, "Initial State", label_font, (0, 255, 0), (0, 0, 0),
                                                        (initial_grid_x + (small_tile_size * 3) // 2,
                                                         initial_grid_y - 20))
                                draw_grid(initial_state, initial_grid_x, initial_grid_y, small_tile_size)
                                render_text_with_border(screen, "Goal State", label_font, (0, 255, 0), (0, 0, 0),
                                                        (goal_grid_x + (small_tile_size * 3) // 2, goal_grid_y - 20))
                                draw_grid(goal_state, goal_grid_x, goal_grid_y, small_tile_size)
                                draw_grid(initial_state, algo_grid_x, algo_grid_y, algo_tile_size)
                                for label, rect in buttons:
                                    corner_radius = 10
                                    button_color = (255, 255, 0) if selected_button == label else (26, 125, 255)
                                    pygame.draw.rect(screen, button_color, rect, border_radius=corner_radius)
                                    pygame.draw.rect(screen, (41, 128, 185), rect, 2, border_radius=corner_radius)
                                    render_text_with_border(screen, label, button_font, (255, 255, 0), (0, 0, 0),
                                                            (rect.centerx, rect.centery))
                                corner_radius = 10
                                info_button_color = (255, 255, 0) if selected_button == "INFO" else (26, 125, 255)
                                pygame.draw.rect(screen, info_button_color, info_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), info_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "INFO", button_font, (255, 255, 255), (0, 0, 0),
                                                        (info_button_rect.centerx, info_button_rect.centery))
                                view_button_color = (255, 255, 0) if selected_button == "VIEW" else (26, 125, 255)
                                pygame.draw.rect(screen, view_button_color, view_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), view_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "VIEW", button_font, (255, 255, 255), (0, 0, 0),
                                                        (view_button_rect.centerx, view_button_rect.centery))
                                reset_button_color = (255, 255, 0) if selected_button == "RESET" else (26, 125, 255)
                                pygame.draw.rect(screen, reset_button_color, reset_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), reset_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "RESET", button_font, (255, 255, 255), (0, 0, 0),
                                                        (reset_button_rect.centerx, reset_button_rect.centery))
                                back_button_color = (255, 255, 0) if selected_button == "BACK" else (26, 125, 255)
                                pygame.draw.rect(screen, back_button_color, back_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), back_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "BACK", button_font, (255, 255, 255), (0, 0, 0),
                                                        (back_button_rect.centerx, back_button_rect.centery))
                                pygame.display.flip()
                                continue
                        elif label == "POS":
                            print("Calling belief_state_interface for POS")
                            initial_state_tuple = tuple(tuple(row) for row in initial_state)
                            initial_belief = puzzle.find_states_with_one_at_00(initial_state_tuple, max_states=6)
                            result = belief_state_interface(initial_belief, goal_state, "POS", performance_history)
                            print(f"Returned from belief_state_interface: {result}")
                            if result == "BACK":
                                selected_button = None
                                solution = None
                                solution_index = 0
                                elapsed_time = 0
                                steps = 0
                                error_message = None
                                display_state = initial_state
                                pygame.init()
                                screen = pygame.display.set_mode((WIDTH, HEIGHT))
                                pygame.display.set_caption("8-Puzzle Solver")
                                try:
                                    background = pygame.image.load(r"asset\anh_backgound\map5.png")
                                    background = pygame.transform.scale(background, (WIDTH, HEIGHT))
                                except pygame.error:
                                    print("Cannot find background image, using white background")
                                    background = pygame.Surface((WIDTH, HEIGHT))
                                    background.fill((255, 255, 255))
                                title_font = pygame.font.SysFont("Comic Sans MS", 70, bold=True)
                                label_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
                                button_font = pygame.font.SysFont("Comic Sans MS", 20, bold=True)
                                info_font = pygame.font.SysFont("Comic Sans MS", 25, bold=True)
                                number_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)
                                screen.blit(background, (0, 0))
                                render_text_with_border(screen, "8-Puzzle Solver", title_font, (255, 0, 0), (0, 0, 0),
                                                        (WIDTH // 2, 50))
                                render_text_with_border(screen, "Initial State", label_font, (0, 255, 0), (0, 0, 0),
                                                        (initial_grid_x + (small_tile_size * 3) // 2,
                                                         initial_grid_y - 20))
                                draw_grid(initial_state, initial_grid_x, initial_grid_y, small_tile_size)
                                render_text_with_border(screen, "Goal State", label_font, (0, 255, 0), (0, 0, 0),
                                                        (goal_grid_x + (small_tile_size * 3) // 2, goal_grid_y - 20))
                                draw_grid(goal_state, goal_grid_x, goal_grid_y, small_tile_size)
                                draw_grid(initial_state, algo_grid_x, algo_grid_y, algo_tile_size)
                                for label, rect in buttons:
                                    corner_radius = 10
                                    button_color = (255, 255, 0) if selected_button == label else (26, 125, 255)
                                    pygame.draw.rect(screen, button_color, rect, border_radius=corner_radius)
                                    pygame.draw.rect(screen, (41, 128, 185), rect, 2, border_radius=corner_radius)
                                    render_text_with_border(screen, label, button_font, (255, 255, 255), (0, 0, 0),
                                                            (rect.centerx, rect.centery))
                                corner_radius = 10
                                info_button_color = (255, 255, 0) if selected_button == "INFO" else (26, 125, 255)
                                pygame.draw.rect(screen, info_button_color, info_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), info_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "INFO", button_font, (255, 255, 255), (0, 0, 0),
                                                        (info_button_rect.centerx, info_button_rect.centery))
                                view_button_color = (255, 255, 0) if selected_button == "VIEW" else (26, 125, 255)
                                pygame.draw.rect(screen, view_button_color, view_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), view_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "VIEW", button_font, (255, 255, 255), (0, 0, 0),
                                                        (view_button_rect.centerx, view_button_rect.centery))
                                reset_button_color = (255, 255, 0) if selected_button == "RESET" else (26, 125, 255)
                                pygame.draw.rect(screen, reset_button_color, reset_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), reset_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "RESET", button_font, (255, 255, 255), (0, 0, 0),
                                                        (reset_button_rect.centerx, reset_button_rect.centery))
                                back_button_color = (255, 255, 0) if selected_button == "BACK" else (26, 125, 255)
                                pygame.draw.rect(screen, back_button_color, back_button_rect,
                                                 border_radius=corner_radius)
                                pygame.draw.rect(screen, (41, 128, 185), back_button_rect, 2,
                                                 border_radius=corner_radius)
                                render_text_with_border(screen, "BACK", button_font, (255, 255, 255), (0, 0, 0),
                                                        (back_button_rect.centerx, back_button_rect.centery))
                                pygame.display.flip()
                                continue

    pygame.quit()
    return None




def belief_state_interface(initial_belief, goal_state, algorithm_type, performance_history):
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"{algorithm_type} State Search Interface")

    try:
        background = pygame.image.load(r"asset\anh_backgound\map5.png")
        background = pygame.transform.scale(background, (WIDTH, HEIGHT))
    except pygame.error:
        print("Cannot find background image, using white background")
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill((255, 255, 255))

    title_font = pygame.font.SysFont("Comic Sans MS", 70, bold=True)
    label_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
    button_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
    info_font = pygame.font.SysFont("Comic Sans MS", 30, bold=True)
    number_font = pygame.font.SysFont("Comic Sans MS", 50, bold=True)

    puzzle = EightPuzzle(list(map(list, initial_belief[0])), goal_state)

    tile_size = 80
    belief_matrices = list(initial_belief)
    belief_positions = []
    num_matrices = len(belief_matrices)
    max_per_row = 3
    offset_x = 50
    goal_pos_y = HEIGHT // 2 - (tile_size * 3) // 2 - 80
    offset_y = goal_pos_y

    for idx in range(num_matrices):
        row = idx // max_per_row
        col = idx % max_per_row
        pos_x = offset_x + col * (tile_size * 3 + 20)
        pos_y = offset_y + row * (tile_size * 3 + 40)
        belief_positions.append((pos_x, pos_y))

    goal_pos_x = WIDTH // 2 - (tile_size * 3) // 2 + 400

    button_run_rect = pygame.Rect(WIDTH - 300, HEIGHT - 70, 100, 40)
    button_back_rect = pygame.Rect(WIDTH - 150, HEIGHT - 70, 100, 40)

    solutions = [None] * num_matrices  # Lưu đường đi giải pháp cho từng Belief State
    solution_indices = [0] * num_matrices  # Chỉ số bước hiện tại của từng Belief State
    elapsed_time = 0
    total_steps = 0
    num_explored_states = 0
    error_message = None
    error_timer = 0
    selected_button = None
    running_solution = False  # Kiểm soát trạng thái chạy giải pháp

    def draw_grid(state, offset_x, offset_y, tile_size):
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size)
                if state[i][j] == 0:
                    pygame.draw.rect(screen, (255, 255, 255), rect)
                else:
                    pygame.draw.rect(screen, (33, 172, 255), rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                if state[i][j] != 0:
                    text = number_font.render(str(state[i][j]), True, (228, 61, 48))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def render_text_with_border(surface, text, font, color, border_color, pos):
        text_surface = font.render(text, True, color)
        border_surface = font.render(text, True, border_color)
        text_rect = text_surface.get_rect(center=pos)

        offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]
        for dx, dy in offsets:
            border_rect = text_rect.copy()
            border_rect.x += dx
            border_rect.y += dy
            surface.blit(border_surface, border_rect)

        surface.blit(text_surface, text_rect)

    running = True
    clock = pygame.time.Clock()
    last_update_time = pygame.time.get_ticks()  # Thời điểm cập nhật bước cuối cùng

    while running:
        screen.blit(background, (0, 0))

        render_text_with_border(screen, f"{algorithm_type} State Search", title_font, (255, 0, 0), (0, 0, 0), (WIDTH // 2, 50))

        # Hiển thị các Belief States
        for idx, (pos_x, pos_y) in enumerate(belief_positions):
            if solutions[idx] is not None:
                if solution_indices[idx] < len(solutions[idx]):
                    state = solutions[idx][solution_indices[idx]]
                else:
                    state = solutions[idx][-1]  # Hiển thị trạng thái cuối (goal state)
            else:
                state = belief_matrices[idx]
            render_text_with_border(screen, f"Belief {idx+1}", label_font, (0, 255, 0), (0, 0, 0),
                                    (pos_x + (tile_size * 3) // 2, pos_y - 20))
            draw_grid(state, pos_x, pos_y, tile_size)

        render_text_with_border(screen, "Goal State", label_font, (0, 255, 0), (0, 0, 0),
                                (goal_pos_x + (tile_size * 3) // 2, goal_pos_y - 20))
        draw_grid(goal_state, goal_pos_x, goal_pos_y, tile_size)

        goal_bottom_y = goal_pos_y + (tile_size * 3)
        if elapsed_time > 1000:
            display_time = f"Running Time: {elapsed_time / 1000:.2f} s"
        else:
            display_time = f"Running Time: {elapsed_time:.2f} ms"
        render_text_with_border(screen, display_time, info_font, (0, 128, 0), (0, 0, 0),
                                (WIDTH // 2 + 400, goal_bottom_y + 40))
        render_text_with_border(screen, f"Steps: {total_steps}", info_font, (0, 128, 0), (0, 0, 0),
                                (WIDTH // 2 + 400, goal_bottom_y + 80))

        if error_message and pygame.time.get_ticks() - error_timer < 1000:
            if elapsed_time > 10000:
                render_text_with_border(screen, "Timeout: Too Long!", info_font, (255, 0, 0), (0, 0, 0),
                                        (WIDTH // 2, goal_bottom_y + 120))
            else:
                render_text_with_border(screen, "No Solution Found!", info_font, (255, 0, 0), (0, 0, 0),
                                        (WIDTH // 2, goal_bottom_y + 120))
        else:
            error_message = None

        corner_radius = 10
        run_button_color = (255, 255, 0) if selected_button == "Run" else (26, 125, 255)
        pygame.draw.rect(screen, run_button_color, button_run_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), button_run_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "Run", button_font, (255, 0, 0), (0, 0, 0),
                                (button_run_rect.centerx, button_run_rect.centery))

        back_button_color = (255, 255, 0) if selected_button == "Back" else (26, 125, 255)
        pygame.draw.rect(screen, back_button_color, button_back_rect, border_radius=corner_radius)
        pygame.draw.rect(screen, (41, 128, 185), button_back_rect, 2, border_radius=corner_radius)
        render_text_with_border(screen, "Back", button_font, (255, 0, 0), (0, 0, 0),
                                (button_back_rect.centerx, button_back_rect.centery))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button_run_rect.collidepoint(mouse_pos):
                    selected_button = "Run"
                    solutions = [None] * num_matrices
                    solution_indices = [0] * num_matrices
                    elapsed_time = 0
                    total_steps = 0
                    num_explored_states = 0
                    error_message = None
                    running_solution = True
                    last_update_time = pygame.time.get_ticks()

                    start_time = timeit.default_timer()
                    if algorithm_type == "Belief":
                        puzzle = EightPuzzle(list(map(list, belief_matrices[0])), goal_state)
                        belief_states_path, explored_states, total_steps = puzzle.belief_state_search(initial_belief)
                        if belief_states_path:
                            solutions = [[] for _ in range(num_matrices)]
                            for belief_set in belief_states_path:
                                for idx in range(min(num_matrices, len(belief_set))):
                                    solutions[idx].append(belief_set[idx])
                        else:
                            error_message = True
                            error_timer = pygame.time.get_ticks()
                            running_solution = False
                        num_explored_states = len(explored_states)
                    elif algorithm_type == "POS":
                        puzzle = EightPuzzle(list(map(list, belief_matrices[0])), goal_state)
                        belief_states_path, explored_states, total_steps = puzzle.partial_observable_search()
                        num_explored_states = len(explored_states)
                        if belief_states_path:
                            # Chuyển belief_states_path thành danh sách các trạng thái cho mỗi Belief State
                            solutions = [[] for _ in range(num_matrices)]
                            for belief_set in belief_states_path:
                                for idx in range(min(num_matrices, len(belief_set))):
                                    solutions[idx].append(belief_set[idx])
                        else:
                            error_message = True
                            error_timer = pygame.time.get_ticks()
                            running_solution = False

                    elapsed_time = (timeit.default_timer() - start_time) * 1000

                    # Lưu hiệu suất vào performance_history
                    performance_history[algorithm_type].append({
                        "runtime": elapsed_time,
                        "steps": total_steps,
                        "states_explored": num_explored_states,
                        "path": solutions if solutions else []
                    })

                if button_back_rect.collidepoint(mouse_pos):
                    selected_button = "Back"
                    pygame.display.flip()
                    pygame.time.delay(200)
                    pygame.quit()
                    return "BACK"

        # Cập nhật giải pháp cho tất cả Belief States đồng thời
        if running_solution:
            current_time = pygame.time.get_ticks()
            if current_time - last_update_time >= 500:  # Cập nhật mỗi 500ms
                all_finished = True
                for idx in range(num_matrices):
                    if solutions[idx] and solution_indices[idx] < len(solutions[idx]):
                        solution_indices[idx] += 1
                        all_finished = False
                last_update_time = current_time
                if all_finished:
                    running_solution = False

    pygame.quit()
    return None


# Hàm chính
if __name__ == "__main__":
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    initial_state = initial_state_selector(goal_state)
    while True:
        result = main_game(initial_state, goal_state)
        if result == "BACK":
            initial_state = initial_state_selector(goal_state)
        else:
            break