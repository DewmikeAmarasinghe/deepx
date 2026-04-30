from math import gcd

# Project Euler solutions for problems 1-6

def problem1():
    # Sum of multiples of 3 or 5 below 1000
    limit = 1000
    def sum_of_multiples(n, a, b):
        def sum_arith(m, d):
            k = (m - 1) // d
            return d * k * (k + 1) // 2
        l = a * b // gcd(a, b)
        return sum_arith(n, a) + sum_arith(n, b) - sum_arith(n, l)
    return sum_of_multiples(limit, 3, 5)


def problem2():
    # Sum of even-valued Fibonacci terms not exceeding four million
    limit = 4_000_000
    a, b = 1, 2
    total = 0
    while a <= limit:
        if a % 2 == 0:
            total += a
        a, b = b, a + b
    return total


def problem3():
    # Largest prime factor of 600851475143
    n = 600851475143
    last_factor = 1
    # remove factors of 2
    while n % 2 == 0:
        last_factor = 2
        n //= 2
    # now odd factors
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            last_factor = factor
            while n % factor == 0:
                n //= factor
        factor += 2
    if n > 1:
        return n
    return last_factor


def problem4():
    # Largest palindrome made from product of two 3-digit numbers
    max_pal = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):
            prod = i * j
            if prod <= max_pal:
                break
            s = str(prod)
            if s == s[::-1]:
                max_pal = prod
                break
    return max_pal


def problem5():
    # Smallest multiple of 1..20
    def lcm(a, b):
        return a // gcd(a, b) * b
    res = 1
    for i in range(1, 21):
        res = lcm(res, i)
    return res


def problem6():
    # Difference between sum of squares and square of sum for 1..100
    n = 100
    sum_squares = sum(i * i for i in range(1, n + 1))
    sum_n = sum(i for i in range(1, n + 1))
    square_of_sum = sum_n * sum_n
    return square_of_sum - sum_squares


if __name__ == "__main__":
    results = {
        'Problem 1': problem1(),
        'Problem 2': problem2(),
        'Problem 3': problem3(),
        'Problem 4': problem4(),
        'Problem 5': problem5(),
        'Problem 6': problem6(),
    }
    for k, v in results.items():
        print(f"{k}: {v}")
