"""
Wrapper for Vision-Language Models (VLM) with automatic architecture detection.
"""

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

if TYPE_CHECKING:
    from PIL import Image


class MacVLAWrapper:
    """
    Wrapper for vision-language models with automatic layer detection.
    Optimized for Mac with MPS (Metal Performance Shaders) support.
    Supports both SmolVLM and OpenVLA models.
    """

    def __init__(self, model_name: str = "smolvlm"):
        """
        Initialize VLM wrapper.

        Args:
            model_name: Model to use. Options:
                - "smolvlm" or "smol": HuggingFaceTB/SmolVLM-Instruct (2.25B, default)
                - "openvla" or "open": openvla/openvla-7b (7B, better physics understanding)
        """
        model_configs = {
            "smolvlm": {
                "id": "HuggingFaceTB/SmolVLM-Instruct",
                "name": "SmolVLM-Instruct",
                "size": "2.25B",
            },
            "smol": {
                "id": "HuggingFaceTB/SmolVLM-Instruct",
                "name": "SmolVLM-Instruct",
                "size": "2.25B",
            },
            "openvla": {"id": "openvla/openvla-7b", "name": "OpenVLA-7B", "size": "7B"},
            "open": {"id": "openvla/openvla-7b", "name": "OpenVLA-7B", "size": "7B"},
        }

        model_name_lower = model_name.lower()
        if model_name_lower not in model_configs:
            print(f"⚠️  Unknown model '{model_name}', defaulting to SmolVLM")
            model_name_lower = "smolvlm"

        config = model_configs[model_name_lower]
        self.model_id = config["id"]
        self.model_display_name = config["name"]
        self.model_size = config["size"]

        print(f"⏳ Loading VLM ({self.model_display_name}, {self.model_size})...")
        print(f"   Model ID: {self.model_id}")

        # Determine device (prioritize Metal Performance Shaders for Mac)
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Detected Apple Silicon (MPS). Running hardware accelerated.")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("⚠️ No GPU detected. Running on CPU (will be slow).")

        # Load processor
        # OpenVLA requires trust_remote_code=True
        trust_remote = model_name_lower in ["openvla", "open"]

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=trust_remote, use_fast=True
            )
        except Exception as e:
            print(f"⚠️  Failed to load processor: {e}")
            print("   Trying alternative loading method...")
            # OpenVLA might need different processor
            from transformers import AutoImageProcessor, AutoTokenizer

            try:
                self.processor = {
                    "image_processor": AutoImageProcessor.from_pretrained(
                        self.model_id, trust_remote_code=trust_remote, use_fast=True
                    ),
                    "tokenizer": AutoTokenizer.from_pretrained(
                        self.model_id, trust_remote_code=trust_remote, use_fast=True
                    ),
                }
            except Exception as e2:
                raise RuntimeError(f"Failed to load processor: {e2}")

        # Load model with correct dtype for Mac
        # OpenVLA is loaded directly from HuggingFace using transformers
        trust_remote = model_name_lower in ["openvla", "open"]

        # Try loading OpenVLA from HuggingFace
        # OpenVLA uses AutoModelForVision2Seq with trust_remote_code=True
        try:
            if trust_remote:
                # For OpenVLA, use AutoModelForVision2Seq with trust_remote_code=True
                # This is the standard way to load OpenVLA from HuggingFace
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    dtype=torch.float16 if self.device == "mps" else torch.float32,
                    _attn_implementation="eager",  # Use eager for MPS compatibility
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)
            else:
                # For standard models, try new API first
                try:
                    from transformers import AutoModelForImageTextToText

                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        dtype=torch.float16 if self.device == "mps" else torch.float32,
                        _attn_implementation="eager",
                        trust_remote_code=trust_remote,
                    ).to(self.device)
                except (ImportError, AttributeError):
                    # Fallback to deprecated API
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_id,
                        dtype=torch.float16 if self.device == "mps" else torch.float32,
                        _attn_implementation="eager",
                        trust_remote_code=trust_remote,
                    ).to(self.device)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a TIMM version error from OpenVLA's code
            if "TIMM" in error_msg or "timm" in error_msg.lower() or "0.9.10" in error_msg:
                print("\n⚠️  OpenVLA requires TIMM >= 0.9.10 and < 1.0.0")
                print("   Attempting to install compatible TIMM version...")
                try:
                    import subprocess
                    import sys

                    # Install the correct TIMM version
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "timm>=0.9.10,<1.0.0"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print("   ✅ TIMM version installed. Retrying model load...")
                    # Retry loading the model with AutoModelForVision2Seq
                    if trust_remote:
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            self.model_id,
                            dtype=torch.float16 if self.device == "mps" else torch.float32,
                            _attn_implementation="eager",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        ).to(self.device)
                    else:
                        try:
                            from transformers import AutoModelForImageTextToText

                            self.model = AutoModelForImageTextToText.from_pretrained(
                                self.model_id,
                                dtype=torch.float16 if self.device == "mps" else torch.float32,
                                _attn_implementation="eager",
                                trust_remote_code=trust_remote,
                            ).to(self.device)
                        except (ImportError, AttributeError):
                            self.model = AutoModelForVision2Seq.from_pretrained(
                                self.model_id,
                                dtype=torch.float16 if self.device == "mps" else torch.float32,
                                _attn_implementation="eager",
                                trust_remote_code=trust_remote,
                            ).to(self.device)
                except Exception as install_error:
                    print(f"   ❌ Failed to install TIMM: {install_error}")
                    print("   Please run manually: pip install 'timm>=0.9.10,<1.0.0'")
                    print("   Or use SmolVLM: --model smolvlm")
                    raise RuntimeError(
                        "OpenVLA requires compatible TIMM version. Please install: pip install 'timm>=0.9.10,<1.0.0'"
                    )

            # For other errors, try loading via AutoModel with trust_remote_code
            # OpenVLA uses a custom model class that needs to be loaded this way
            print(f"⚠️  Standard AutoModel classes failed: {e}")
            print("   Trying AutoModel with trust_remote_code...")
            try:
                from transformers import AutoModel

                # AutoModel should work with trust_remote_code=True for custom models
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    dtype=torch.float16 if self.device == "mps" else torch.float32,
                    trust_remote_code=trust_remote,
                ).to(self.device)
            except Exception as e2:
                error_msg2 = str(e2)
                # Check for TIMM errors
                if "TIMM" in error_msg2 or "timm" in error_msg2.lower() or "0.9.10" in error_msg2:
                    print("\n⚠️  OpenVLA requires TIMM >= 0.9.10 and < 1.0.0")
                    print("   Attempting to install compatible TIMM version...")
                    try:
                        import subprocess
                        import sys

                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "timm>=0.9.10,<1.0.0"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        print("   ✅ TIMM version installed. Retrying model load...")
                        # Retry with AutoModel
                        self.model = AutoModel.from_pretrained(
                            self.model_id,
                            dtype=torch.float16 if self.device == "mps" else torch.float32,
                            trust_remote_code=trust_remote,
                        ).to(self.device)
                    except Exception as install_error:
                        print(f"   ❌ Failed to install TIMM: {install_error}")
                        print("   Please run manually: pip install 'timm>=0.9.10,<1.0.0'")
                        print("   Or use SmolVLM: --model smolvlm")
                        raise RuntimeError(
                            "OpenVLA requires compatible TIMM version. Please install: pip install 'timm>=0.9.10,<1.0.0'"
                        )
                else:
                    # Try loading the model class directly from the config
                    if trust_remote:
                        try:
                            from transformers import AutoConfig

                            config = AutoConfig.from_pretrained(
                                self.model_id, trust_remote_code=True
                            )
                            # Check auto_map for the model class
                            if hasattr(config, "auto_map") and config.auto_map:
                                # Try different possible keys in auto_map
                                for key in [
                                    "AutoModel",
                                    "AutoModelForCausalLM",
                                    "AutoModelForVision2Seq",
                                ]:
                                    if key in config.auto_map:
                                        model_class_path = config.auto_map[key]
                                        import importlib

                                        module_path, class_name = model_class_path.rsplit(".", 1)
                                        module = importlib.import_module(module_path)
                                        ModelClass = getattr(module, class_name)
                                        self.model = ModelClass.from_pretrained(
                                            self.model_id,
                                            dtype=torch.float16
                                            if self.device == "mps"
                                            else torch.float32,
                                            trust_remote_code=trust_remote,
                                        ).to(self.device)
                                        break
                                else:
                                    raise RuntimeError(
                                        f"Could not find model class in auto_map. Available keys: {list(config.auto_map.keys()) if hasattr(config, 'auto_map') else 'None'}"
                                    )
                            else:
                                raise RuntimeError("Config has no auto_map")
                        except Exception as e3:
                            raise RuntimeError(
                                f"Failed to load OpenVLA model. Error: {e3}. Original error: {e2}"
                            )
                    else:
                        raise RuntimeError(f"Failed to load model. Error: {e2}")

        # Automatically find the transformer layers
        self.layers = self._find_transformer_layers()
        print(f"✅ Architecture Identified. Found {len(self.layers)} layers.")

    def _find_transformer_layers(self):
        """
        Robustly finds the ModuleList containing the transformer layers
        by inspecting the model structure recursively.

        Returns:
            torch.nn.ModuleList: The transformer layers

        Raises:
            AttributeError: If transformer layers cannot be found
        """
        # Common paths for different VLM architectures
        candidates = [
            ["language_model", "model", "layers"],  # LLaVA
            ["model", "text_model", "layers"],  # Idefics3 / SmolVLM
            ["text_model", "layers"],  # Idefics2
            ["model", "layers"],  # Qwen-VL / Plain Transformers
            ["layers"],  # Fallback
        ]

        for path in candidates:
            curr = self.model
            valid_path = True
            for attr in path:
                if hasattr(curr, attr):
                    curr = getattr(curr, attr)
                else:
                    valid_path = False
                    break

            if valid_path and isinstance(curr, torch.nn.ModuleList):
                return curr

        # Fallback: Search all modules for a ModuleList with many layers
        for _name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 5:
                return module

        raise AttributeError(
            "Could not automatically find the transformer layer list in this model."
        )

    def get_layer(self, layer_idx=-2):
        """
        Get a specific transformer layer.

        Args:
            layer_idx: Index of the layer (negative indices supported)

        Returns:
            The transformer layer module
        """
        return self.layers[layer_idx]

    def forward_pass(self, image: Image.Image, text_prompt: str):
        """
        Run a forward pass through the model.

        Args:
            image: PIL Image to process
            text_prompt: Text prompt for the model

        Returns:
            Logits for the last token position
        """
        # Handle different processor types
        if isinstance(self.processor, dict):
            # OpenVLA-style processor (separate image processor and tokenizer)
            image_processor = self.processor["image_processor"]
            tokenizer = self.processor["tokenizer"]

            # Get model dtype to match inputs
            model_dtype = next(self.model.parameters()).dtype

            # Process image and text separately
            image_inputs = image_processor(image, return_tensors="pt").to(self.device)
            text_inputs = tokenizer(text_prompt, return_tensors="pt").to(self.device)

            # Convert to model's dtype (important for float16 models)
            if "pixel_values" in image_inputs:
                image_inputs["pixel_values"] = image_inputs["pixel_values"].to(dtype=model_dtype)
            if "input_ids" in text_inputs:
                text_inputs["input_ids"] = text_inputs["input_ids"].to(self.device)
            if "attention_mask" in text_inputs:
                text_inputs["attention_mask"] = text_inputs["attention_mask"].to(self.device)

            # Combine inputs (OpenVLA format may vary)
            inputs = {**image_inputs, **text_inputs}
        else:
            # Standard processor (SmolVLM-style)
            # Check if processor has chat template
            if (
                hasattr(self.processor, "apply_chat_template")
                and self.processor.chat_template is not None
            ):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": text_prompt}],
                        }
                    ]
                    prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                    inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                        self.device
                    )
                except (ValueError, AttributeError):
                    # Fallback: use text prompt directly if chat template fails
                    inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(
                        self.device
                    )
            else:
                # No chat template, use text prompt directly
                inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(
                    self.device
                )
                # Ensure pixel_values match model dtype
                model_dtype = next(self.model.parameters()).dtype
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Handle different output formats
        if hasattr(outputs, "logits"):
            return outputs.logits[0, -1, :]
        elif hasattr(outputs, "last_hidden_state"):
            # Some models return last_hidden_state instead of logits
            return outputs.last_hidden_state[0, -1, :]
        else:
            # Fallback: try to get logits from first output
            if isinstance(outputs, tuple):
                return outputs[0][0, -1, :]
            raise ValueError(f"Unknown output format from model: {type(outputs)}")

    def batch_decode(self, token_ids):
        """
        Decode token IDs to text, handling different processor types.

        Args:
            token_ids: Tensor of token IDs to decode

        Returns:
            List of decoded strings
        """
        if isinstance(self.processor, dict):
            return self.processor["tokenizer"].batch_decode(token_ids)
        else:
            return self.processor.batch_decode(token_ids)
