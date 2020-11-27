import sys
import yaml
import shutil
from os import makedirs
from os.path import join, basename, splitext, exists


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
    exp_cfg.__dict__.update(get_outdir(exp_cfg))
    data_cfg = Config(cfg_file, cfg["data"])
    model_cfg = Config(cfg_file, cfg["model"])

    return {"data_cfg": data_cfg, "model_cfg": model_cfg, "exp_cfg": exp_cfg}

def get_outdir(exp_cfg):
    outdir = join(exp_cfg.output_location, splitext(basename(exp_cfg.cfg_file))[0])
    if exists(outdir):
        res = input(
                "Version already exists and will be overwritten. Are you sure you want to continue?(y/n) "
            )
        if res == 'n':
            print("Exiting program.")
            sys.exit(0)
        else:
            shutil.rmtree(outdir)

    makedirs(outdir, exist_ok=True)
    print("models will be saved to:", outdir)
    return {"output_location": outdir}