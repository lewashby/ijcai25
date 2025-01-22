import typer
import dataclasses
import random
from pathlib import Path
from typing import Optional
from itertools import groupby
from dumbo_utils.console import console
from llmasp import llm
import json
from rich.progress import Progress
from rich.table import Table
from rich.live import Live
from dumbo_asp.primitives.atoms import GroundAtom
from dumbo_asp.primitives.models import Model
from enum import Enum

from .utils import load_data, save_results, evaluate_model

ROOT_PATH = Path(__file__).parent.parent.parent.resolve()

@dataclasses.dataclass(frozen=True)
class AppOptions:
    model: str
    server: str
    ollama_key: str
    behavior_file: Path
    application_file: Path
    dataset: Path
    single_pass: bool
    output_file: Path
    stats_file: Path
    log_prompts: bool
    debug: bool

class Level(str, Enum):
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5

app_options: Optional[AppOptions] = None
app = typer.Typer()

def make_table(title, table_data):
    table = Table(title=title)
    for idx, col in enumerate(table_data[0]):
        table.add_column(col, justify="left" if idx == 0 else "right")
    for row in table_data[1:-1]:
        table.add_row(*row)
    table.add_row()
    table.add_row(*table_data[-1])
    return table

@app.callback()
def main(
        model: str = typer.Option(..., "--model", "-m", help="Ollama model ID"),
        server: str = typer.Option("http://localhost:11434/v1", "--server", "-s",
                                   envvar="OLLAMA_SERVER", help="Ollama server URL"),
        ollama_key: str = typer.Option("ollama", "--ollama-key", "-ok", envvar="OLLAMA_API_KEY",
                                       help="Ollama API key"),
        behavior_file: Path = typer.Option(..., "--behavior-file", "-bf",
                                           help="Path for behavior file"),
        application_file: Path = typer.Option(..., "--application-file", "-af",
                                              help="Path for application file (or directory)"),
        dataset: Path = typer.Option(ROOT_PATH / "iaspllm/data/dataset.json", "--dataset", "-d", help="Path for dataset file"),
        single_pass: bool = typer.Option(False, "--single-pass", "-sp",
                                         help="Make single queries to llm"),
        output_file: Path = typer.Option(ROOT_PATH / "iaspllm/results/llm_output.json",
                                         "--output-file", "-of", help="Output file path for llm response"),
        stats_file: Path = typer.Option(ROOT_PATH / "iaspllm/results/llm_output_stats.csv",
                                        "--stats-file", "-sf", help="Statistics file path for llm response"),
        log_prompts: bool = typer.Option(False, "--log-prompts", help="Log prompts and responses"),
        debug: bool = typer.Option(False, "--debug", help="Print debug error messages"),
):
    """
    CLI for LLMASP
    """
    global app_options

    app_options = AppOptions(
        model=model,
        server=server,
        ollama_key=ollama_key,
        behavior_file=behavior_file,
        application_file=application_file,
        dataset=dataset,
        single_pass=single_pass,
        output_file=output_file,
        stats_file=stats_file,
        log_prompts=log_prompts,
        debug=debug,
    )

@app.command(name="llmasp-run-selected")
def llmasp_command_run_selected(
        problem_name: str = typer.Option(..., "--problem-name", "-pn", help="Problem name"),
        quantity: int = typer.Option(..., "--quantity", "-q", help="Instances quantity"),
        random_instances: bool = typer.Option(False, "--random-instances", "-rn",
                                              help="Take random instances"),
) -> None:
    """
    Run selected testcases with LLMASP.
    """
    assert app_options is not None

    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        problem_name = problem_name.replace(" ", "")
        problem_instances = [instance for instance in data
                             if instance["problem_name"].replace(" ", "") == problem_name]
        if random_instances:
            problem_instances = random.sample(problem_instances, quantity)
        else:
            problem_instances = problem_instances[:quantity]
        console.log("Data loaded!")
    console.log(f"Problem: {problem_name}; Instances: {quantity}")
    console.log(f"Testing model: {app_options.model}")
    results = test_problem(
        app_options.model,
        app_options.server,
        app_options.ollama_key,
        app_options.behavior_file,
        app_options.application_file,
        problem_instances,
        single_pass=app_options.single_pass
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="llmasp-full-test")
def llmasp_command_full_test():
    """
    Run for the entire dataset with LLMASP.
    """
    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        console.log("Data loaded!")
    console.log(f"Testing model: {app_options.model}")
    results = test_llmasp_dataset(
        app_options.model,
        app_options.server,
        app_options.ollama_key,
        app_options.behavior_file,
        app_options.application_file,
        data,
        single_pass=app_options.single_pass
    )
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="llama-run-selected")
def llama_command_run_selected(
        problem_name: str = typer.Option(..., "--problem-name", "-pn", help="Problem name"),
        quantity: int = typer.Option(..., "--quantity", "-q", help="Instances quantity"),
        prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                           help="1-Text \n2-Text+description \n3-Text+format \n4-Text+Encoding \n5-Text+description+format"),
        random_instances: bool = typer.Option(False, "--random-instances", "-rn",
                                              help="Take random instances"),
) -> None:
    """
    Run selected testcases with Llama model.
    """
    assert app_options is not None

    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        problem_name = problem_name.replace(" ", "")
        problem_instances = [instance for instance in data
                             if instance["problem_name"].replace(" ", "") == problem_name]
        if random_instances:
            problem_instances = random.sample(problem_instances, quantity)
        else:
            problem_instances = problem_instances[:quantity]
        console.log("Data loaded!")
    console.log(f"Problem: {problem_name}; Instances: {quantity}")
    console.log(f"Testing model: {app_options.model}")

    results = test_model(
        app_options.model,
        problem_instances,
        app_options.server,
        app_options.ollama_key,
        prompt_level=prompt_level
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

@app.command(name="llama-full-test")
def llama_command_full_test(
    prompt_level: Level = typer.Option(Level.five, "--prompt-level", "-pl", show_default=False,
                                    help="1-Text \n2-Text+description \n3-Text+format \n4-Text+Encoding \n5-Text+description+format"),
):
    """
    Run for the entire dataset with Llama model.
    """
    with console.status("Loading data..."):
        data = load_data(app_options.dataset)
        console.log("Data loaded!")
    console.log(f"Testing model: {app_options.model}")
    results = test_llama_dataset(
        app_options.model,
        app_options.server,
        app_options.ollama_key,
        data,
        prompt_level=prompt_level
    )
    console.log("Done with the model")
    save_results(results, file_path=app_options.output_file)
    console.print('---------------------------------Evaluate responses--------------------------------------')
    evaluate_model(app_options.model, results, app_options.stats_file)
    console.print('----------------------------------------Done---------------------------------------------')

def log_history(history):
    for item in history:
        console.log("[red]" + item[-2]["content"].replace('[', '\\['))
        console.log("[blue]" + item[-1]["content"].replace('[', '\\['))

def create_prompt(instance, prompt_level):
    # only text
    if prompt_level=='1':
        prompt = f"Extract the datalog facts from this text: \n ```{instance['text']}"
    # description and text
    elif prompt_level=='2':
        prompt = f"Given the following problem description between triple backtips: \n```{instance['description']}```\nExtract the datalog facts from this text: \n ```{instance['text']}```"
    # format and text
    elif prompt_level=='3':
        prompt = f"Given the following specification for the predicates format between triple backtips: \n```{instance['format']}```\nExtract the datalog facts from this text: \n ```{instance['text']}```"
    # only encoding
    elif prompt_level=='4':
        with open('data/dataset_encodings.json') as f:
            dataset_encodings = json.load(f)
        prompt = "Consider the following Datalog encoding for the problem, provided below within triple backticks: \n ```" + dataset_encodings[instance['problem_name']]['encoding'].strip() + "```\n"
        prompt += "Extract the datalog facts from this text: \n```" + instance['text'].strip() + "```\n"
        prompt += 'Output the result in datalog with no comments, no space between facts and only one fact per line.'
    # all
    else:
        prompt = instance['prompt']
    return prompt

def test_tool(tool, data: list, single_pass: bool=False):
    progress = Progress(console=console)
    task_id = progress.add_task("Running...", completed=0, total=len(data))

    def live_grid(completed, table_data=None):
        progress.update(task_id, completed=completed)
        grid = Table.grid()
        grid.add_row(progress)
        if table_data is not None:
            grid.add_row(make_table("Stats", table_data))

        return grid

    results = []
    model = tool.llm.model
    with Live(console=console) as live:
        live.update(live_grid(0))
        for index, instance in enumerate(data):
            query = instance["text"]
            created_facts, _, history, _ = tool.natural_to_asp(query, single_pass=single_pass)
            if tool.config["knowledge_base"]:
                facts = []
                for fact in created_facts.split("\n"):
                    try:
                        facts.append(str(GroundAtom.parse(fact[:-1])))
                    except:
                        console.log(f"Ignored atom {fact}")
                created_facts = Model.of_program(
                    Model.of_atoms(facts).as_facts,
                    tool.config["knowledge_base"]
                ).as_facts
            results.append({**instance, **{model: created_facts}, **{"history": history}})
            if app_options.log_prompts:
                log_history(history)

            live.update(live_grid(index + 1, evaluate_model(app_options.model, results, None)))

    return results

def test_problem(model, server, ollama_key, behavior_file, application_file, data: list, single_pass: bool = False):
    llm_handler = llm.LLMHandler(model, server_url=server, api_key=ollama_key)    
    tool = llm.LLMASP(application_file, behavior_file, llm_handler, None)
    results = test_tool(tool, data, single_pass=single_pass)
    return results

def test_llmasp_dataset(model, server, ollama_key, behavior_file, application_files_folder, data: list, single_pass: bool=False):
    results = []
    llm_handler = llm.LLMHandler(model, server_url=server, api_key=ollama_key)
    data.sort(key=lambda i: i["problem_name"])
    grouped_problems = groupby(data, key=lambda p: p["problem_name"])
    for problem_name, instances in grouped_problems:
        instances = list(instances)
        console.log(f"Problem: {problem_name}; Instances: {len(instances)}")
        problem_name = problem_name.replace(" ", "")
        application_file = f"{application_files_folder}/{problem_name}.yml"
        tool = llm.LLMASP(application_file, behavior_file, llm_handler, None)
        problem_results = test_tool(tool, instances, single_pass=single_pass)
        results.extend(problem_results)
    return results

def test_model(model_name, data, server_url: str="http://localhost:11434/v1", ollama_key="ollama", prompt_level=5):
    llm_handler = llm.LLMHandler(model_name, server_url=server_url, api_key=ollama_key)
    results = []
    progress = Progress(console=console)
    task_id = progress.add_task("Running...", completed=0, total=len(data))

    def live_grid(completed, table_data=None):
        progress.update(task_id, completed=completed)
        grid = Table.grid()
        grid.add_row(progress)
        if table_data is not None:
            grid.add_row(make_table("Stats", table_data))

        return grid

    with Live(console=console) as live:
        live.update(live_grid(0))
        for index, instance in enumerate(data):
            prompt = create_prompt(instance, prompt_level)
            messages=[
            {
                "role": "system",
                "content": "You will be provided with unstructured data, and your task is to parse it into datalog facts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
            completion, _ = llm_handler.call(messages)
            results.append({**instance, **{model_name: completion}, **{"history": messages}})
            live.update(live_grid(index + 1, evaluate_model(app_options.model, results, None)))

    return results

def test_llama_dataset(model, server, ollama_key, data, prompt_level):
    results = []
    data.sort(key=lambda i: i["problem_name"])
    grouped_problems = groupby(data, key=lambda p: p["problem_name"])
    for problem_name, instances in grouped_problems:
        instances = list(instances)[:2]
        print(len(instances))
        console.log(f"Problem: {problem_name}; Instances: {len(instances)}")
        problem_name = problem_name.replace(" ", "")
        problem_results = test_model(model, instances, server, ollama_key, prompt_level)
        
        results.extend(problem_results)
    return results