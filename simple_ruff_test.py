# I: import order / multiple imports

# F: unused variable
unused_var = 42  # F841

# E/W: pycodestyle whitespace issues / indentation


def hello_world():
    print("Hello Ruff!")  # E201/E202 extra spaces + single quotes (should be double)
    print("bad indent")


# C: flake8-comprehensions
squares = [x * x for x in range(5)]  # C
squares2 = [x * x for x in range(5) if True]  # C

# B: flake8-bugbear
if x := 10:  # B010 / B018
    print(x)


# UP: pyupgrade
def old_style_default(x=1):  # UP006: spaces around default argument
    return x


# line too long (should be ignored by E501)
long_line = "This is a very long line that should exceed 100 characters to test if line-length and ignore rules are working properly in Ruff."

hello_world()
old_style_default()
