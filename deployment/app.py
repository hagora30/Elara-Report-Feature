
import os
import modal


MODEL_ID  = "hagora-30/Elara-14B-Merged"
MODEL_DIR = "/model"
APP_NAME  = "elara-14b"

def download_model():
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        repo_id         = MODEL_ID,
        local_dir       = MODEL_DIR,
        token           = token,
        ignore_patterns = ["*.pt", "*.bin"],
    )
    print("Model downloaded!")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.2", 
        "huggingface_hub",
        "transformers",
        "accelerate",
        "fastapi",
        "uvicorn",
    )
    .run_function(
        download_model,
        secrets = [modal.Secret.from_name("huggingface-secret")],
        timeout = 60 * 30,
    )
)


app = modal.App(APP_NAME)


@app.cls(
    gpu = "a100",
    image = vllm_image,
    secrets = [modal.Secret.from_name("huggingface-secret")],
    scaledown_window = 120,
    timeout = 600,
)
class ElaraModel:

    @modal.enter()
    def load_engine(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        print("Loading vLLM engine...")
        engine_args = AsyncEngineArgs(
            model                   = MODEL_DIR,
            dtype                   = "bfloat16",
            max_model_len           = 4096,
            gpu_memory_utilization  = 0.90,
            trust_remote_code       = True,
            tensor_parallel_size    = 1,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM engine loaded!")

    @modal.web_endpoint(method="POST", label="elara-chat")
    async def chat(self, request: dict):
        from vllm import SamplingParams
        from vllm.utils import random_uuid
        from transformers import AutoTokenizer

        messages    = request.get("messages", [])
        max_tokens  = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.3)
        top_p       = request.get("top_p", 0.9)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
        )

        sampling_params = SamplingParams(
            max_tokens         = max_tokens,
            temperature        = temperature,
            top_p              = top_p,
            repetition_penalty = 1.1,
        )

        request_id        = random_uuid()
        results_generator = self.engine.generate(
            prompt, sampling_params, request_id
        )

        final_output = None
        async for output in results_generator:
            final_output = output

        generated_text = final_output.outputs[0].text

        return {
            "id"     : f"chatcmpl-{request_id}",
            "object" : "chat.completion",
            "model"  : MODEL_ID,
            "choices": [{
                "index"        : 0,
                "message"      : {
                    "role"   : "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens"    : len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens"     : len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
            }
        }

    @modal.web_endpoint(method="GET", label="elara-health")
    async def health(self):
        return {"status": "ok", "model": MODEL_ID}
