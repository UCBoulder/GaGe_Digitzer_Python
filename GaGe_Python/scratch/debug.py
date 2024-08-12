def f(*args, **kwargs):
    print("arguments are", args)
    print("keyword arguments are", kwargs)
    return args, kwargs

args, kwargs = f("1", "2", "3", one="one", two="two")
