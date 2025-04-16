import sys
from enum import Enum
from operator import itemgetter
from typing import Callable

import modal

from .common import GraphState, image

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field


class Nodes:
    def __init__(
        self,
        context: str,
        sb: modal.Sandbox,
        run: Callable[[str, modal.Sandbox], tuple[str, str]],
        debug: bool = False,
    ):
        self.context = context
        self.debug = debug
        self.model = "gpt-4o-2024-08-06" if not self.debug else "gpt-4o-mini-2024-07-18"
        self.node_map = {
            "generate": self.generate,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "evaluate_execution": self.evaluate_execution,  # New node
            "finish": self.finish,
        }

        self.sb = sb
        self.run = run

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on docs and the input question
        with optional feedback from code execution tests

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        question = state_dict["question"]
        iter = state_dict["iterations"]

        ## Data model
        class Code(BaseModel):
            """Code output"""

            prefix: str = Field(description="Description of the problem and approach")
            imports: str = Field(description="Code block import statements")
            code: str = Field(description="Code block not including import statements")

        ## LLM
        llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)

        # Tool
        code_tool_oai = convert_to_openai_tool(Code)

        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[code_tool_oai],
            tool_choice={"type": "function", "function": {"name": "Code"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[Code])

        ## Prompt
        template = """
You are a coding assistant with expertise in Python.
You are able to execute Python code in a sandbox environment.
You are tasked with responding to the following user question: {question}
Your response will be shown to the user.
Here is a full set of documentation:

-------
{context}
-------

Answer the user question based on the above provided documentation.
Ensure any code you provide can be executed with all required imports and variables defined.
Structure your answer as a description of the code solution,
then a list of the imports, and then finally list the functioning code block.
Here is the user question again:

--- --- ---
{question}"""

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            # Update prompt
            addendum = """You previously tried to solve this problem. Here is your solution:

{generation}

Here is the resulting error from code execution:

{error}

Please re-try to answer this. Structure your answer with a description of the code solution.
Then list the imports. And finally list the functioning code block. Structure your answer with a description of
the code solution. Then list the imports. And finally list the functioning code block.

Here is the user question:

{question}"""
            template = template + addendum

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "generation", "error"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                    "generation": itemgetter("generation"),
                    "error": itemgetter("error"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": str(code_solution[0]),
                    "error": error,
                }
            )

        else:
            print("---GENERATE SOLUTION---")

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke({"question": question})

        iter = iter + 1
        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "iterations": iter,
            }
        }

    def check_code_imports(self, state: GraphState) -> GraphState:
        """
        Check imports

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE IMPORTS---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iter = state_dict["iterations"]

        # Attempt to execute the imports
        output, error = self.run(imports, self.sb)
        if error:
            print("---CODE IMPORT CHECK: FAILED---")
            # Catch any error during execution (e.g., ImportError, SyntaxError)
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = f"""
{error_prev_runs}

--- Most recent run output and error ---
------ output ------
{output}
------ error ------
{error}
"""
        else:
            print("---CODE IMPORT CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "iterations": iter,
            }
        }

    def check_code_execution(self, state: GraphState) -> GraphState:
        """
        Check code block execution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE EXECUTION---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iter = state_dict["iterations"]

        output, error = self.run(code_block, self.sb)
        if error:
            print("---CODE BLOCK CHECK: FAILED---")
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE BLOCK CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "output": output,
                "iterations": iter,
            }
        }

    def evaluate_execution(self, state: GraphState) -> GraphState:
        """
        Evaluate the code execution results and determine whether to finish or retry.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updated state with decision to finish or retry
        """
        print("---EVALUATING EXECUTION---")

        state_dict = state["keys"]
        output = state_dict["output"]
        error = state_dict["error"]

        code_solution = state_dict["generation"][0]
        code = code_solution.code

        class Decision(str, Enum):
            FINISH = "finish"
            RETRY = "retry"

        class ExecutionEvaluation(BaseModel):
            """Evaluation of code execution"""

            decision: Decision = Field(description="Decision to finish or retry")
            explanation: str = Field(description="Explanation for the decision")

        llm = ChatOpenAI(temperature=0, model=self.model)
        evaluation_tool = convert_to_openai_tool(ExecutionEvaluation)
        llm_with_tool = llm.bind(
            tools=[evaluation_tool],
            tool_choice={
                "type": "function",
                "function": {"name": "ExecutionEvaluation"},
            },
        )
        parser_tool = PydanticToolsParser(tools=[ExecutionEvaluation])

        template = """
You are an expert code evaluator. Analyze the following code execution results and determine if the execution was successful.

Code:
{code}

Output:
{output}

Error:
{error}

Decide whether to finish (if the execution was successful) or retry (if there were errors or unexpected results).
Provide a brief explanation for your decision.
        """.strip()

        prompt = PromptTemplate(
            template=template,
            input_variables=["code", "output", "error"],
        )

        chain = prompt | llm_with_tool | parser_tool

        evaluation = chain.invoke({"code": code, "output": output, "error": error})

        return {
            "keys": {
                **state_dict,
                "evaluation": evaluation[0],
            }
        }

    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """

        print("---FINISHING---")

        response = extract_response(state)

        self.sb.terminate()

        return {"keys": {"response": response}}


def extract_response(state: GraphState) -> str:
    """
    Extract the response from the graph state

    Args:
        state (dict): The current graph state

    Returns:
        str: The response
    """

    state_dict = state["keys"]
    code_solution = state_dict["generation"][0]

    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code

    code_output = state_dict["output"]

    return f"""{prefix}

{imports}
{code}

Result of code execution:
{code_output}
"""
