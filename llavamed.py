import json
import threading
from typing import List, Generator, Dict, Optional, Union
import os

import torch
from transformers import TextIteratorStreamer

# Import necessary components from llava
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

class SimpleLLaVaMed:
    def __init__(
        self,
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        model_base: Optional[str] = None,
        model_name: Optional[str] = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = "cuda",
        num_gpus: int = None
    ):
        self.device = device
        
        # Set up multi-GPU if requested
        if num_gpus and num_gpus > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
            print(f"Using {num_gpus} GPUs")
        
        # Clean model path
        if model_path.endswith("/"):
            model_path = model_path[:-1]
            
        # Determine model name if not provided
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name
            
        print(f"Loading model {self.model_name}...")
        
        # Load the model, tokenizer, and image processor
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device
        )
        
        # Check if multimodal
        self.is_multimodal = True  # Force multimodal mode as specified in the command
        print(f"Model loaded. Multimodal: {self.is_multimodal}")
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_new_tokens: int = 256,
        stop: Optional[str] = None,
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:

        # Process images if provided and model is multimodal
        num_image_tokens = 0
        image_args = {}
        
        if images and len(images) > 0 and self.is_multimodal:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
                
            processed_images = [load_image_from_base64(image) for image in images]
            processed_images = process_images(processed_images, self.image_processor, self.model.config)
            
            if isinstance(processed_images, list):
                processed_images = [img.to(self.device, dtype=torch.float16) for img in processed_images]
            else:
                processed_images = processed_images.to(self.device, dtype=torch.float16)
                
            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            num_image_tokens = prompt.count(replace_token) * self.model.get_vision_tower().num_patches
            image_args = {"images": processed_images}
        
        # Prepare generation parameters
        do_sample = temperature > 0.001
        
        # Tokenize input
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        # Set up stopping criteria
        max_context_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens, 1024)
        
        if max_new_tokens < 1:
            return "Exceeds max token length. Please use a shorter prompt."
        
        stopping_criteria = None
        if stop:
            keywords = [stop]
            stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)]
        
        # Generate
        if stream:
            return self._generate_stream(
                input_ids, 
                do_sample, 
                temperature, 
                top_p, 
                max_new_tokens, 
                stopping_criteria,
                image_args, 
                prompt,
                stop
            )
        else:
            return self._generate_complete(
                input_ids, 
                do_sample, 
                temperature, 
                top_p, 
                max_new_tokens, 
                stopping_criteria,
                image_args,
                stop
            )
    
    def _generate_stream(
        self, 
        input_ids, 
        do_sample, 
        temperature, 
        top_p, 
        max_new_tokens, 
        stopping_criteria,
        image_args, 
        original_prompt,
        stop
    ) -> Generator[str, None, None]:
        """Stream output token-by-token"""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            **image_args
        )
        
        # Start generation in a separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the results
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            if stop and generated_text.endswith(stop):
                generated_text = generated_text[:-len(stop)]
            yield generated_text
    
    def _generate_complete(
        self, 
        input_ids, 
        do_sample, 
        temperature, 
        top_p, 
        max_new_tokens, 
        stopping_criteria,
        image_args,
        stop
    ) -> str:
        """Generate the complete output at once"""
        with torch.no_grad():
            generation_output = self.model.generate(
                inputs=input_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                **image_args
            )
        
        # Decode the output
        text = self.tokenizer.decode(generation_output[0, input_ids.shape[1]:], skip_special_tokens=True)
        if stop and text.endswith(stop):
            text = text[:-len(stop)]
        return text
