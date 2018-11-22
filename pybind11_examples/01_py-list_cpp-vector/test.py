import sys

sys.path.append("build")
import example

A = [1., 2., 3., 4.]

B = example.modify(A)

print(B)
