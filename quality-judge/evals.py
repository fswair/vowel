import os
import pathlib

import dotenv

from vowel.codemode import CodeModeGenerator
from vowel.runner import Function

dotenv.load_dotenv()

SPEC_MODEL = os.getenv("SPEC_MODEL")
EXPLORATION_MODEL = os.getenv("EXPLORATION_MODEL")

generator = CodeModeGenerator(
    spec_model=SPEC_MODEL,
    exploration_model=EXPLORATION_MODEL,
    generation_id="largest_color_value_judge_spec_quality",
)


async def generate_spec(fn: Function):
    # check for code can compile (it will be executed in monty anyways)
    _ = fn.impl
    result = await generator.generate(fn, save_to_file=True)
    print(result)
    generator.print_total_cost()
    return result.yaml_spec


async def generate_spec_mock(fn: Function):
    return pathlib.Path(
        "/Users/mert/Desktop/LIP/evalspec/quality-judge/largestPathValue_evals.yml"
    ).read_text()
