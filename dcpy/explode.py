# http://blog.musicallyut.in/2016/11/17/explode-out-of-your-function.html
class Exploded(BaseException):
    pass


def explode():
    import inspect

    for k, v in inspect.currentframe.f_back.f_locals.items():
        globals()[k] = v

    raise Exploded()
