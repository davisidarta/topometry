import inspect
from functools import partial
from types import ModuleType, MethodType
from typing import Union, Callable, Optional
from weakref import WeakSet


def save_pkl(TopOGraph, wd=None, filename='topograph.pkl'):
    try:
        import pickle
    except ImportError:
        return (print('Pickle is needed for saving the TopOGraph. Please install it with `pip3 install pickle`'))

    if wd is None:
        import os
        wd = os.getcwd()
    with open(wd + filename, 'wb') as output:
        pickle.dump(TopOGraph, output, pickle.HIGHEST_PROTOCOL)
    return print('TopOGraph saved at ' + wd + filename)

def read_pkl(wd=None, filename='topograph.pkl'):
    try:
        import pickle
    except ImportError:
        return (print('Pickle is needed for loading the TopOGraph. Please install it with `pip3 install pickle`'))

    if wd is None:
        import os
        wd = os.getcwd()
    with open(wd + filename, 'rb') as input:
        TopOGraph = pickle.load(input)
    return TopOGraph

def _one_of_ours(obj, root: str):
    return (
        hasattr(obj, "__name__")
        and not obj.__name__.split(".")[-1].startswith("_")
        and getattr(
            obj, '__module__', getattr(obj, '__qualname__', obj.__name__)
        ).startswith(root)
    )

def descend_classes_and_funcs(mod: ModuleType, root: str, encountered=None):
    if encountered is None:
        encountered = WeakSet()
    for obj in vars(mod).values():
        if not _one_of_ours(obj, root):
            continue
        if callable(obj) and not isinstance(obj, MethodType):
            yield obj
            if isinstance(obj, type):
                for m in vars(obj).values():
                    if callable(m) and _one_of_ours(m, root):
                        yield m
        elif isinstance(obj, ModuleType) and obj not in encountered:
            if obj.__name__.startswith('scanpy.tests'):
                # Python’s import mechanism seems to add this to `scanpy`’s attributes
                continue
            encountered.add(obj)
            yield from descend_classes_and_funcs(obj, root, encountered)

def getdoc(c_or_f: Union[Callable, type]) -> Optional[str]:
    if getattr(c_or_f, '__doc__', None) is None:
        return None
    doc = inspect.getdoc(c_or_f)
    if isinstance(c_or_f, type) and hasattr(c_or_f, '__init__'):
        sig = inspect.signature(c_or_f.__init__)
    else:
        sig = inspect.signature(c_or_f)

    def type_doc(name: str):
        param: inspect.Parameter = sig.parameters[name]
        cls = getattr(param.annotation, '__qualname__', repr(param.annotation))
        if param.default is not param.empty:
            return f'{cls}, optional (default: {param.default!r})'
        else:
            return cls

    return '\n'.join(
        f'{line} : {type_doc(line)}' if line.strip() in sig.parameters else line
        for line in doc.split('\n')
    )

def annotate_doc_types(mod: ModuleType, root: str):
    for c_or_f in descend_classes_and_funcs(mod, root):
        c_or_f.getdoc = partial(getdoc, c_or_f)