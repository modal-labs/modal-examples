# ---
# lambda-test: false  # training script that is called from learn_math.py
# pytest: false
# ---

# # Training script for training a reasoning model using the verifiers library with sandboxed code execution

# This script is used to train a model using GRPO. This is adapted from the [verifiers library](https://github.com/willccbb/verifiers/blob/main/verifiers/examples/math_python.py) example.
# Here, we use a Modal Sandbox to execute python code during training. Modal Sandboxes offer an easy way to execute untrusted code in a completely isolated environment.
# This is a more secure way to execute python code during training.

import sys

import modal
import verifiers as vf
from verifiers.utils import load_example_dataset

# We create a Modal app and a Modal sandbox.
app = modal.App.lookup("example-trainer-script-grpo", create_if_missing=True)
sb = modal.Sandbox.create(app=app)


# We create a function that will execute the python code in a Modal Sandbox.
def sandbox_exec(code):
    try:
        process = sb.exec("python", "-c", code, timeout=10)
        process.wait()

        stdout = process.stdout.read()
        stderr = process.stderr.read()
        if stderr:
            return f"Error: {stderr.strip()}"

        output = stdout.strip() if stdout else ""
        if len(output) > 1000:
            output = output[:1000] + "... (truncated to 1000 chars)"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# We define the tool prompt for prompting the model. Then, we pass in our `sandbox_exec` function as a tool to the `ToolEnv` definition.

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\\nx = sympy.symbols('x')\\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.
"""

dataset = load_example_dataset("math", split="train").select(range(128))

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[sandbox_exec],
    max_steps=3,
)

run_id = sys.argv[2]
model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

# These parameters are adapted to test the training script via an overfitting test. We will use 128 examples from the training set and overfit the model to them.
# To learn more about the parameters, please refer to the [verifiers library](https://github.com/willccbb/verifiers/blob/main/verifiers/examples/math_python.py) example.

training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 50
training_args.max_steps = 50
training_args.per_device_train_batch_size = 4
training_args.gradient_accumulation_steps = 4
training_args.num_generations = 12
training_args.learning_rate = 1e-3
training_args.logging_steps = 1
training_args.report_to = "wandb"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()

sb.terminate()
save_path = f"/root/math_weights/{run_id}"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")
