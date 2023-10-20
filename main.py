import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# rounds 为仿真次数
ROUNDS = 100
# max_iterations 为最大迭代次数
MAX_ITERATIONS = 10000
# test_number 为一个样本测试多少次
TEST_NUMBER = 100
# n 为样本数
N = 20
# DEBUG 为是否输出中间内容
DEBUG = False

P = 1e9 + 7


def generate_numbers(n, x, s=1):
    """
    :param n: 个数
    :param x: 总概率
    :param s: ni不能超过s
    :return: 一个list满足sum ni=x，且ni均匀随机分布
    """
    random.seed(time.time_ns() % P)
    if n * s < x:
        raise ValueError("It's impossible to generate numbers with the given constraints.")

    points = [0]
    sum_so_far = 0
    cnt = 0
    pesudo_random = False
    while len(points) < n:
        next_point = sum_so_far + random.uniform(0, min(s, x - sum_so_far))

        # 如果下一个点导致我们超过了限制，我们重新调整它
        if len(points) == n - 1 and next_point != x:
            next_point = x

        if next_point - sum_so_far <= s and next_point <= x:
            sum_so_far = next_point
            points.append(next_point)
        cnt += 1
        if cnt > 10000:
            # 这个概率太难生成了
            pesudo_random = True
            break

    if pesudo_random:
        points = [0]
        step = x / n
        for i in range(n):
            points.append(points[-1] + step)
    return [points[i] - points[i - 1] for i in range(1, len(points))]


def random_index_based_on_probability(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)


def update_probabilities(probabilities, chosen_index, increment=0.01):
    if probabilities[chosen_index] + increment > 1:
        increment = 1 - probabilities[chosen_index]

    probabilities[chosen_index] += increment

    scaling_factor = (1 - increment) / (1 - probabilities[chosen_index])
    for i in range(len(probabilities)):
        if i != chosen_index:
            probabilities[i] -= (probabilities[i] / (1 - probabilities[chosen_index])) * increment
    # 确保精度问题不会导致概率之和不为1
    total_prob = sum(probabilities)
    if not np.isclose(total_prob, 1.0, atol=1e-10,rtol=0):
        probabilities[-1] += (1.0 - total_prob)

    return probabilities


def simulate(A, n, max_iterations):
    """
    :param A: A 的概率
    :param n: 总个数
    :param max_iterations: 最大迭代次数
    :return: bool, A是否能在最大迭代次数后得到50%以上
    """
    remain = 1 - A
    probabilities = generate_numbers(n - 1, remain, A)
    probabilities = [A] + probabilities
    while A != max(probabilities):
        probabilities = generate_numbers(n - 1, remain, A)
        probabilities = [A] + probabilities
    cakes = 100
    for i in range(max_iterations):
        choice = random_index_based_on_probability(probabilities)
        probabilities = update_probabilities(probabilities, choice, 1 / cakes)
        cakes += 1
        # 至于这里选0.1是否科学有待商榷
        if probabilities[0] < 0.1 or probabilities[0] > 0.5:
            break
        # print(probabilities)
    if DEBUG:
        if probabilities[0] < 0.1:
            print("A 几乎不可能达到0.5了")
        else:
            if probabilities[0] > 0.5:
                print("A 达到0.5了")
            else:
                print("A 最终也没能确定是否能到达0.5，它的值为", A)
    if probabilities[0] > 0.5:
        return True
    else:
        return False


def test(n, A):
    """
    :param n: 可以选择的个数
    :param A: A的概率
    :return: A成功到达50%的概率
    """
    succeed = 0
    failure = 0

    for i in range(100):
        if simulate(A, n, MAX_ITERATIONS):
            succeed += 1
        else:
            failure += 1

    print("成功次数为", succeed, "  失败次数为", failure)
    return succeed / (succeed + failure)


def run():
    n = 20
    x = np.array([])
    y = np.array([])
    for val in np.arange(0.2, 0.55, 0.05):
        x = np.append(x, val)
        y = np.append(y, test(n, val))
    # 原始数据
    print(x)
    print(y)
    # 创建更细的x值
    xnew = np.linspace(np.min(x), np.max(x), 300)

    # 使用spline插值
    spl = make_interp_spline(x, y, k=3)  # k=3表示三次样条插值
    ynew = spl(xnew)

    plt.plot(xnew, ynew, label="Smoothed Curve")
    plt.scatter(x, y, color='red', label="Original Data", zorder=5)
    plt.legend()
    plt.show()


run()
