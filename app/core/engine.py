import torch
import tempfile
from transformers import (
    AutoModel,
    AutoTokenizer,
)


class DeepSeekOCR:

    def __init__(
            self,
            model_path: str = 'app/core/model-files/deepseek-ai/DeepSeek-OCR/',
    ):
        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model.eval().to("cuda")
        self.prompt = "<image>\n<|grounding|>Convert the document to markdown."

    def infer(
            self,
            image_pth: str,
    ):
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            _ = self.model.infer(
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                image_file=image_pth,
                output_path=tmpdir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True,
            )
            torch.cuda.empty_cache()
            with open(
                file = f"{tmpdir}/result.mmd",
                mode="r",
                encoding="utf-8",
            ) as f:
                content = f.read()
        return content
