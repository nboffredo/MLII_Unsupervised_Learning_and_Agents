import matplotlib.pyplot as plt
import numpy as np

nb_data = 5000
age_max = 110

# 0 = blue, green = 1, brown = 2, hazelnut = 3, other = 4
eye_colors = [0, 1, 2, 3, 4]
eye_colors_probs = [0.1, 0.02, 0.8, 0.05, 0.03]

style = {"facecolor": "blue", "alpha": 0.2, "pad": 10}
data = (np.random.randint(0, age_max, nb_data), np.random.choice(eye_colors, size=nb_data, p=eye_colors_probs))

exp_value_age = 0;
exp_value_color = 0;

for i in range(age_max):
    exp_value_age += i * (np.count_nonzero(data[0] == i) / nb_data)
for i in range(len(eye_colors)):
    exp_value_color += i * eye_colors_probs[i]

exp_value_color_test = 0
for i in range(len(eye_colors)):
    exp_value_color_test += i * (np.count_nonzero(data[1] == i) / nb_data)

# Thanks to linearity of expected value:
exp_value = exp_value_color + exp_value_age

n = 5000

plt.plot(data[0][:n], data[1][:n], "o")
plt.title("exercise_1")
plt.xlabel("age")
plt.ylabel("eye_color")
plt.savefig(f"results/unif_discrete_{n}_values.pdf")
plt.close()

eucl_distance = []
for i in range(n):
    if (i == 0):
        continue
    empirical_mean = (np.mean(data[0][:i]) + np.mean(data[1][:i]))
    eucl_distance.append(abs(exp_value - empirical_mean))

start_point = 10

plt.plot(range(start_point + 1, n), eucl_distance[start_point:], "o")
plt.title("Euclidian distance to the expected value as a function of n")
plt.xlabel("number of value n")
plt.ylabel("Euclidian distance to the expected value")
plt.savefig(f"results/eucl_distance_to_the_expected_value_as_a_function_of_n_with_5000_values.pdf")
plt.close()