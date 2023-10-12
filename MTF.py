from datetime import datetime, timedelta

n = datetime.now()

b = n + timedelta(hours=24)


print(n)
print(b)

print(n>b)
print(n<b)