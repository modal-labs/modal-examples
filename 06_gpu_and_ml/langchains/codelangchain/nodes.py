import sys
from operator import itemgetter

import sandbox
from common import GraphState, agent_image, image

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI


class Nodes:
    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug
        self.model = (
            "gpt-4-0125-preview" if not self.debug else "gpt-3.5-turbo-0125"
        )
        self.node_map = {
            "generate": self.generate,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "finish": self.finish,
        }

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on LCEL docs and the input question
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
        class code(BaseModel):
            """Code output"""

            prefix: str = Field(
                description="Description of the problem and approach"
            )
            imports: str = Field(description="Code block import statements")
            code: str = Field(
                description="Code block not including import statements"
            )

        ## LLM
        llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)

        # Tool
        code_tool_oai = convert_to_openai_tool(code)

        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[code_tool_oai],
            tool_choice={"type": "function", "function": {"name": "code"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[code])

        ## Prompt
        template = (
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
        You are able to execute Python code in a sandbox environment that was constructed by chaining together the following two Dockerfile commands: \n
        """
            + f"{image.dockerfile_commands()}"
            + "\n"
            + f"{agent_image.dockerfile_commands()}"
            + """
            You are tasked with responding to the following user question: {question}
            Your response will be shown to the user.
            Here is a full set of LCEL documentation:
            \n ------- \n
            {context}
            \n ------- \n
            Answer the user question based on the above provided documentation. \n
            Ensure any code you provide can be executed with all required imports and variables defined. \n
            Structure your answer as a description of the code solution, \n
            then a list of the imports, and then finally list the functioning code block. \n
            Here is the user question again: \n --- --- --- \n {question}
        """
        )

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            # Update prompt
            addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:
                        \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code
                        execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this.
                        Structure your answer with a description of the code solution. \n Then list the imports.
                        And finally list the functioning code block. Structure your answer with a description of
                        the code solution. \n Then list the imports. And finally list the functioning code block.
                        \n Here is the user question: \n --- --- --- \n {question}"""
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
        sb = sandbox.run(imports)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE IMPORT CHECK: FAILED---")
            # Catch any error during execution (e.g., ImportError, SyntaxError)
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
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
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iter = state_dict["iterations"]

        sb = sandbox.run(code_block)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE BLOCK CHECK: FAILED---")
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
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
                "prefix": prefix,
                "imports": imports,
                "iterations": iter,
                "code": code,
            }
        }

    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """

        print("---FINISHING---")
        state_dict = state["keys"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code

        response = "\n".join([prefix, imports, code])

        return {"keys": {"response": response}}
