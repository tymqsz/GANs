from data import MonetDataset

d = MonetDataset(None, "images")

i = iter(d)

print(next(i).shape)