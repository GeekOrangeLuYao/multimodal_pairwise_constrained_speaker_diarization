import argparse
import yaml
import json

parser = argparse.ArgumentParser(description="read config")
parser.add_argument("--conf_file", required=True, help="Config file")


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(**value)
            else:
                self.__dict__[key] = value

    def is_leaf(self):
        # the key -> value, value cannot be Config
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                return False
        return True

    def kwargs(self):
        res = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                continue
            res[key] = value
        return res

    def to_string(self, step):
        s = "\n"
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                s = f"{s}{'  ' * step}{key}:{value.to_string(step + 1)}"
            else:
                s = f"{s}{'  ' * step}{key}: {value}\n"
        return s

    def __str__(self):
        return self.to_string(0)

    def exist_key(self, key):
        if key in self.__dict__:
            return True
        else:
            return False

    def items(self):
        return self.__dict__.items()


class ConfigLoader(object):
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.conf_dict = dict()
        self.load_data()

    def instance(self) -> Config:
        return Config(**self.conf_dict)

    def load_data(self):
        """
        Load the data to dict, core function
        :return:
        """
        pass

    def __repr__(self):
        return f"{self.conf_file}\n{repr(self.conf_dict)}"


class YamlConfigLoader(ConfigLoader):
    """
        Using .yml or .yaml filebin
        Common selection
    """

    def __init__(self, conf_file):
        super(YamlConfigLoader, self).__init__(conf_file)

    def load_data(self):
        with open(self.conf_file, "r") as fr:
            self.conf_dict = yaml.full_load(fr)


class JsonConfigLoader(ConfigLoader):
    """
        Using .json filebin
    """

    def __init__(self, conf_file):
        super(JsonConfigLoader, self).__init__(conf_file)

    def load_data(self):
        with open(self.conf_file, "r") as fr:
            self.conf_dict = json.load(fr)


def build_config(config_file):
    if str(config_file).endswith(".json"):
        loader = JsonConfigLoader(config_file)
    elif str(config_file).endswith(".yaml"):
        loader = YamlConfigLoader(config_file)
    else:
        raise ValueError("Unknown config file format")

    config = loader.instance()
    return config


def main(args):
    conf_file = args.conf_file
    loader = None
    if str(conf_file).endswith(".json"):
        loader = JsonConfigLoader(conf_file)
    elif str(conf_file).endswith(".yaml"):
        loader = YamlConfigLoader(conf_file)
    print(loader)
    config = loader.instance()
    print(config)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
