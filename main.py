import math
#  QUESTION 01
# Mean
def mean(numbers):
  return sum(numbers) / len(numbers)
print("Mean",mean([1,4,5,7,8,9,10]))
# Median
def median(numbers):
  numbers.sort()
  n = len(numbers)
  if n % 2 == 0:
      return (numbers[n//2 - 1] + numbers[n//2]) / 2
  else:
      return numbers[n//2]
print("Median",median([1,4,5,7,8,9,10]))
# Mode
def mode(numbers):
  frequency = {} 
  for num in numbers:
      if num in frequency:
          frequency[num] += 1
      else:
          frequency[num] = 1
  max_count = max(frequency.values())
  mode = []
  for key in frequency:
      if frequency[key] == max_count:
          mode.append(key)
  if len(mode) == 1:
      return mode[0]
  else:
      return mode
numbers = [1,4,5,7,8,9,10]
print("Mode:",mode(numbers))


# QUESTION 02
# Variance
def variance(numbers):
  total = 0
  for num in numbers:
      total += num
  mean = total / len(numbers)
  squared_differences = []
  for num in numbers:
      squared_differences.append((num - mean) ** 2)
  total_squared_diff = 0
  for value in squared_differences:
      total_squared_diff += value
  variance = total_squared_diff / len(numbers)
  return variance
numbers = [1,4,5,7,8,9,10]
print("Variance:", variance(numbers))

# QUESTION 03
# Standard Deviation
def standard_deviation(numbers):
  total = 0
  for num in numbers:
      total += num
  mean = total / len(numbers)
  squared_differences = []
  for num in numbers:
      squared_differences.append((num - mean) ** 2)
  total_squared_diff = 0
  for value in squared_differences:
      total_squared_diff += value
  variance = total_squared_diff / len(numbers)
  standard_deviation = math.sqrt(variance)
  return standard_deviation
numbers = [1,4,5,7,8,9,10]
print("Standard Deviation:", standard_deviation(numbers))

# QUESTION 04
 # Euclidean Distance
def euclidean_distance(a, b):
  if isinstance(a, list) and isinstance(b, list):
      return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
  else:
      return abs(a - b)
print("Euclidean Distance:", euclidean_distance([3, 4], [6, 8]))

# QUESTION 05
#Sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
print("Sigmoid:", sigmoid(5))
