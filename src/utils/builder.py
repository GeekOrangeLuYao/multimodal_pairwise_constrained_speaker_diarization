import importlib

from utils.config import Config


def dynamic_import(import_path, alias=None):
    if alias is None:
        alias = dict()
    if import_path not in alias and ':' not in import_path:
        raise ValueError(f"import_path should be one of {alias} or "
                         f"include ':'")

    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, obj_name = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, obj_name)


def deep_build(config):
    if config is None:
        return None
    if not isinstance(config, Config):
        return config
    elif config.exist_key("module"):
        module_cls = dynamic_import(config.module)
        if config.exist_key("parameters"):
            return module_cls(**deep_build(config.parameters))
        else:
            return module_cls()
    elif config.exist_key("function"):
        func = dynamic_import(config.function)
        return func
    else:
        res = dict()
        for key, value in config.items():
            res[key] = deep_build(value)
        return res
