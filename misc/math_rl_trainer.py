import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset
from verifiers.tools import python
import modal

app = modal.App.lookup("math-rl", create_if_missing=True)
sb = modal.Sandbox.create(app=app)

def sandbox_exec(code):
    try:
        process = sb.exec('python', '-c', code, timeout=10)
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

dataset = (
    load_example_dataset("math", split="train")
    .shuffle(seed=42)
    .select(range(512))
)

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[sandbox_exec],
    max_steps=3
)

model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations               = 50
training_args.max_steps                     = 50
training_args.per_device_train_batch_size  = 4
training_args.gradient_accumulation_steps  = 4
training_args.num_generations              = 12
training_args.learning_rate                = 1e-3
training_args.logging_steps                = 1
training_args.report_to                     = "wandb"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()

sb.terminate()
save_path = "/root/math_weights"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")