from config import pou_pipeline_config as config
from pipeline.pou_pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(config.__dict__, pde_problem=config.pde)