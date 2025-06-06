# Task 1.1
student_scores = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78,
    "Dana": 82,
    "Emily": 95
}

namelist = ["Alice", "Brian", "Charlie", "David", "Emily", "Fred"]

for name in namelist:
    if name in student_scores:
        print(f"{name} {student_scores[name]}")
    else:
        print(f"{name} added")
        student_scores[name] = 0
print(student_scores)

# Task 1.2
student_scores = {
    "Alice": [85, 87, 92, 96],
    "Bob": [92, 91, 94, 84],
    "Charlie": [78, 80, 82, 84]
}

print("Bob's grade on assignment 2:", student_scores["Bob"][1])

student_scores["Charlie"][2] = 85

print("Charlie's grades after update:", student_scores["Charlie"])