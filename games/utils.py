import inspect
from collections import OrderedDict
from functools import update_wrapper


def fill_args_with_defaults(f):
    signature = inspect.signature(f).parameters.items()

    def wrapper(*args, **kwargs):
        args_with_defaults = OrderedDict([(pn, p.default) for pn, p in signature])

        for arg, argname in zip(args, args_with_defaults):
            args_with_defaults[argname] = arg

        args_with_defaults.update(kwargs)
        return f(*args_with_defaults.values())

    update_wrapper(wrapper, f)

    return wrapper

def fill_kwargs_with_defaults(f):
    signature = inspect.signature(f).parameters.items()

    def wrapper(*args, **kwargs):
        kwargs_with_defaults = OrderedDict([(pn, p.default) for pn, p in signature])

        for arg, argname in zip(args, kwargs_with_defaults):
            kwargs_with_defaults[argname] = arg

        kwargs_with_defaults.update(kwargs)
        return f(**kwargs_with_defaults)

    update_wrapper(wrapper, f)

    return wrapper


# TEST

if __name__ == '__main__':
    @fill_kwargs_with_defaults
    def some_function(val1, val2, kwarg1=4, kwarg2=None):
        print(locals())
        return

    some_function(1, 2, 3, 4)
    some_function(1, val2=2, kwarg1=3, kwarg2=4)
    some_function(val1=1, val2=2, kwarg1=3)
