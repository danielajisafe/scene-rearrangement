import yaml


class Config(object):
    """
    Class for all attributes and functionalities related to a particular training run.
    """
    def __init__(self, cfg_file: str, params: dict):
        self.cfg_file = cfg_file
        self.__dict__.update(params)


def cfg_parser(cfg_file: str) -> dict:
    """
    This functions reads an input config file and instantiates objects of
    Config types.
    args:
        cfg_file (string): path to cfg file
    returns:
        data_cfg (Config)
        model_cfg (Config)
        exp_cfg (Config)
    """
    cfg = yaml.load(open(cfg_file, "r"), Loader=yaml.FullLoader)

    exp_cfg = Config(cfg_file, cfg["experiment"])
    data_cfg = Config(cfg_file, cfg["data"])
    model_cfg = Config(cfg_file, cfg["model"])

    return {"data_cfg": data_cfg, "model_cfg": model_cfg, "exp_cfg": exp_cfg}
