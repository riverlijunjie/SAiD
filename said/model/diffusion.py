"""Define the diffusion models which are used as SAiD model
"""
from abc import ABC
from dataclasses import dataclass
import inspect
from typing import List, Optional, Type, Union
from diffusers import DDIMScheduler, SchedulerMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Processor,
)
from .unet_1d_condition import UNet1DConditionModel
from .wav2vec2 import ModifiedWav2Vec2Model

import openvino as ov
enable_log = False

@dataclass
class SAIDInferenceOutput:
    """
    Dataclass for the inference output
    """

    result: torch.FloatTensor  # (Batch_size, sample_seq_len, x_dim), Generated blendshape coefficients
    intermediates: List[
        torch.FloatTensor
    ]  # (Batch_size, sample_seq_len, x_dim), Intermediate blendshape coefficients


@dataclass
class SAIDNoiseAdditionOutput:
    """
    Dataclass for the noise addition output
    """

    noisy_sample: torch.FloatTensor
    noise: torch.FloatTensor
    velocity: torch.FloatTensor


class SAID(ABC, nn.Module):
    """Abstract class of SAiD models"""

    denoiser: nn.Module

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Type[SchedulerMixin] = DDIMScheduler,
        in_channels: int = 32,
        feature_dim: int = -1,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
        prediction_type: str = "epsilon",
        use_ov: bool = True,
        ov_model_path: str = "./",
        device_name: str = "GPU",
        convert_model: bool = False,
        export_model: bool = False,
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler: Type[SchedulerMixin]
            Noise scheduler, by default DDIMScheduler
        in_channels : int
            Dimension of the input, by default 32
        feature_dim : int
            Dimension of the latent feature, by default -1
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        prediction_type: str
            Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
        """
        super().__init__()

        # Audio-related
        self.audio_config = (
            audio_config if audio_config is not None else Wav2Vec2Config()
        )

        self.use_ov = use_ov
        self.convert_model = convert_model
        self.export_model = export_model
        self.ov_model_path = ov_model_path
        self.convert_unet_model = False
        self.convert_audio_model = False
        self.device_name = device_name

        if convert_model:
            self.convert_unet_model = True
            self.convert_audio_model = True
        if use_ov == True:
            print("ov::compiled_model: ", self.ov_model_path + "/ModifiedWav2Vec2Model.onnx, device_name = ", self.device_name, " ...", end="")
            core = ov.Core()
            self.ov_audio_encoder = core.compile_model(self.ov_model_path + "/ModifiedWav2Vec2Model.xml",self.device_name, config={'CACHE_DIR': self.ov_model_path})
            print("done")
            print("input:",self.ov_audio_encoder.inputs)
            print("output:",self.ov_audio_encoder.outputs)
            print()
        else:
            self.audio_encoder = ModifiedWav2Vec2Model(self.audio_config)

        self.audio_processor = (
            audio_processor
            if audio_processor is not None
            else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        )
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        self.latent_scale = latent_scale

        # Noise scheduler
        self.noise_scheduler = noise_scheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
        )

        # Feature embedding
        self.feature_dim = feature_dim
        if self.feature_dim > 0:
            self.audio_proj_layer = nn.Linear(
                self.audio_config.output_hidden_size, self.feature_dim
            )
            self.null_cond_emb = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        else:
            self.null_cond_emb = nn.Parameter(
                torch.randn(1, 1, self.audio_config.output_hidden_size)
            )

        """
        # Relieve the clipping
        self.noise_scheduler.betas = betas_for_alpha_bar(diffusion_steps, 1 - 1e-15)
        self.noise_scheduler.alphas = 1.0 - self.noise_scheduler.betas
        self.noise_scheduler.alphas_cumprod = torch.cumprod(
            self.noise_scheduler.alphas, dim=0
        )
        """

    def forward(
        self,
        noisy_samples: torch.FloatTensor,
        timesteps: torch.LongTensor,
        audio_embedding: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Return the predicted noise in the noisy samples

        Parameters
        ----------
        noisy_samples : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Sequence of noisy coefficients
        timesteps : torch.LongTensor
            (Batch_size,) or (1,), Timesteps
        audio_embedding : torch.FloatTensor
            (Batch_size, embedding_seq_len, embedding_size), Sequence of audio embeddings

        Returns
        -------
        torch.FloatTensor
            (Batch_size, coeffs_seq_len, num_coeffs), Sequence of predicted noises
        """
        timestep_size = timesteps.size()
        if len(timestep_size) == 0 or timestep_size[0] == 1:
            batch_size = noisy_samples.shape[0]
            timesteps = timesteps.repeat(batch_size)

        if enable_log == True:
            print("denoiser: ", noisy_samples.shape, timesteps, audio_embedding.shape)
        #denoiser:  torch.Size([2, 231, 32]) tensor([900, 900]) torch.Size([2, 231, 768])

        if self.use_ov == True:
            name = self.ov_denoiser.output(0)
            noise_pred = self.ov_denoiser([noisy_samples, timesteps, audio_embedding])[name]
            noise_pred = torch.tensor(noise_pred)
        else:
            noise_pred = self.denoiser(noisy_samples, timesteps, audio_embedding)

        if self.convert_unet_model == True:
            dtype_mapping = {
                torch.float32: ov.Type.f32,
                torch.int64: ov.Type.i64,
                torch.float64: ov.Type.f64,
            }

            dummy_inputs = (noisy_samples, timesteps, audio_embedding)
            input_info=[]
            if 0:
                for input_tensor in dummy_inputs:
                    shape = ov.PartialShape(input_tensor.shape)
                    element_type = dtype_mapping[input_tensor.dtype]
                    input_info.append((shape, element_type))
            else:
                input_info.append((ov.PartialShape([2,-1,32]), ov.Type.f32))
                input_info.append((ov.PartialShape([2]), ov.Type.i64))
                input_info.append((ov.PartialShape([2,-1,768]), ov.Type.f32))
            print("Convert UNet1DConditionModel to be IR ...", end="")
            with torch.no_grad():    
                ov_model = ov.convert_model(self.denoiser, example_input=dummy_inputs, input=input_info)
            ov.save_model(ov_model, self.ov_model_path + "/UNet1DConditionModel.xml")
            del ov_model
            self.convert_unet_model = False
            print(" done")

        return noise_pred

    def pred_original_sample(
        self,
        noisy_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Predict the denoised sample (x_{0}) based on the noisy samples and the noise

        Parameters
        ----------
        noisy_samples : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Noisy sample
        noise : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Noise
        timesteps : torch.LongTensor
            (Batch_size,), Current timestep

        Returns
        -------
        torch.FloatTensor
            Predicted denoised sample (x_{0})
        """
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            noisy_samples - beta_prod_t**0.5 * noise
        ) / alpha_prod_t**0.5

        return pred_original_sample

    def process_audio(
        self, waveform: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.FloatTensor:
        """Process the waveform to fit the audio encoder

        Parameters
        ----------
        waveform : Union[np.ndarray, torch.Tensor, List[np.ndarray]]
            - np.ndarray, torch.Tensor: (audio_seq_len,)
            - List[np.ndarray]: each (audio_seq_len,)

        Returns
        -------
        torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        """
        out = self.audio_processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"]
        return out

    def get_audio_embedding(
        self, waveform: torch.FloatTensor, num_frames: Optional[int]
    ) -> torch.FloatTensor:
        """Return the audio embedding of the waveform

        Parameters
        ----------
        waveform : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        num_frames: Optional[int]
            The length of output audio embedding sequence, by default None

        Returns
        -------
        torch.FloatTensor
            (Batch_size, embed_seq_len, embed_size), Generated audio embedding.
            If num_frames is not None, embed_seq_len = num_frames.
        """
        print("audio_encoder: ", waveform.shape, num_frames)
        if self.use_ov == True:
            if enable_log == True:
                print("ov::inference audio_encoder...")
            features = self.ov_audio_encoder(waveform)[self.ov_audio_encoder.output(0)]
            features = torch.tensor(features)
        else:
            features = self.audio_encoder(waveform).last_hidden_state

        if self.convert_audio_model == True:
            dtype_mapping = {
                torch.float32: ov.Type.f32,
                torch.int64: ov.Type.i64,
                torch.float64: ov.Type.f64,
            }
            dummy_inputs = (waveform)
            input_info=[]
            if 0:
                shape = ov.PartialShape(waveform.shape)
                element_type = dtype_mapping[waveform.dtype]
                input_info.append((shape, element_type))
            else:
                input_info.append((ov.PartialShape([1,-1]), ov.Type.f32))
            print("Convert ModifiedWav2Vec2Model to be IR ...",end="")
            with torch.no_grad():    
                ov_model = ov.convert_model(self.audio_encoder, example_input=dummy_inputs, input=input_info)
            ov.save_model(ov_model, self.ov_model_path + "/ModifiedWav2Vec2Model.xml")
            del ov_model
            self.convert_audio_model = False
            print(" done!")

        if self.feature_dim > 0:
            features = self.audio_proj_layer(features)
        return features

    def get_random_timesteps(self, batch_size: int) -> torch.LongTensor:
        """Return the random timesteps

        Parameters
        ----------
        batch_size : int
            Size of the batch

        Returns
        -------
        torch.LongTensor
            (batch_size,), random timesteps
        """
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            dtype=torch.long,
        )
        return timesteps

    def add_noise(
        self, sample: torch.FloatTensor, timestep: torch.LongTensor
    ) -> SAIDNoiseAdditionOutput:
        """Add the noise into the sample

        Parameters
        ----------
        sample : torch.FloatTensor
            Sample to be noised
        timestep : torch.LongTensor
            (num_timesteps,), Timestep of the noise scheduler

        Returns
        -------
        SAIDNoiseAdditionOutput
            Noisy sample and the added noise
        """
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_sample = self.noise_scheduler.add_noise(sample, noise, timestep)
        velocity = self.noise_scheduler.get_velocity(sample, noise, timestep)

        return SAIDNoiseAdditionOutput(
            noisy_sample=noisy_sample, noise=noise, velocity=velocity
        )

    def encode_samples(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        """Encode samples into latent

        Parameters
        ----------
        samples : torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Samples

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Output latent
        """
        return samples.clone()

    def decode_latent(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode latent into samples

        Parameters
        ----------
        latent : torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Latent

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Output samples
        """
        return latent.clone()

    def inference(
        self,
        waveform_processed: torch.FloatTensor,
        init_samples: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        strength: float = 1.0,
        guidance_scale: float = 2.5,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,
        fps: int = 60,
        save_intermediate: bool = False,
        show_process: bool = False,
    ) -> SAIDInferenceOutput:
        """Inference pipeline

        Parameters
        ----------
        waveform_processed : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        init_samples : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, x_dim), Starting point for the process, by default None
        mask : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, x_dim), Mask the region not to be changed, by default None
        num_inference_steps : int, optional
            The number of denoising steps, by default 100
        strength: float, optional
            How much to paint. Must be between 0 and 1, by default 1.0
        guidance_scale : float, optional
            Guidance scale in classifier-free guidance, by default 2.5
        guidance_rescale : float, optional
            Guidance rescale to control rescale strength, by default 0.0
        eta : float, optional
            Eta in DDIM, by default 0.0
        fps : int, optional
            The number of frames per second, by default 60
        save_intermediate: bool, optional
            Return the intermediate results, by default False
        show_process: bool, optional
            Visualize the inference process, by default False

        Returns
        -------
        SAIDInferenceOutput
            Inference results and the intermediates
        """
        batch_size = waveform_processed.shape[0]
        waveform_len = waveform_processed.shape[1]
        in_channels = self.denoiser.in_channels
        device = waveform_processed.device
        do_classifier_free_guidance = guidance_scale > 1.0
        window_size = int(waveform_len / self.sampling_rate * fps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        latents = (
            torch.randn(batch_size, window_size, in_channels, device=device)
            if init_samples is None
            else self.encode_samples(init_samples)
        )

        # Scaling the latent
        latents *= self.latent_scale * self.noise_scheduler.init_noise_sigma

        init_latents = latents.clone()
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # Add additional noise
        noise = None
        if init_samples is not None:
            timestep = self.noise_scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor(
                [timestep] * batch_size, dtype=torch.long, device=device
            )

            noise_output = self.add_noise(latents, timesteps)
            latents = noise_output.noisy_sample
            noise = noise_output.noise

        audio_embedding = self.get_audio_embedding(waveform_processed, window_size)
        if do_classifier_free_guidance:
            """
            uncond_waveform = [np.zeros((waveform_len)) for _ in range(batch_size)]
            uncond_waveform_processed = self.process_audio(uncond_waveform).to(device)
            uncond_audio_embedding = self.get_audio_embedding(
                uncond_waveform_processed, window_size
            )
            """
            # uncond_audio_embedding = torch.zeros_like(audio_embedding)
            uncond_audio_embedding = self.null_cond_emb.repeat(
                batch_size, audio_embedding.shape[1], 1
            )
            audio_embedding = torch.cat([uncond_audio_embedding, audio_embedding])

        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta

        intermediates = []

        t_start = num_inference_steps - init_timestep

        for idx, t in enumerate(
            tqdm(
                self.noise_scheduler.timesteps[t_start:],
                disable=not show_process,
            )
        ):
            if save_intermediate:
                interm = self.decode_latent(latents / self.latent_scale)
                intermediates.append(interm)

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            noise_pred = self.forward(latent_model_input, t, audio_embedding)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_audio + guidance_scale * (
                    noise_pred_audio - noise_pred_uncond
                )

                if guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_audio, guidance_rescale
                    )

            latents = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            # Masking
            if init_samples is not None and mask is not None:
                init_latents_noisy = init_latents

                tdx_next = t_start + idx + 1
                if tdx_next < num_inference_steps:
                    t_next = self.noise_scheduler.timesteps[tdx_next]
                    init_latents_noisy = self.noise_scheduler.add_noise(
                        init_latents, noise, t_next
                    )

                latents = init_latents_noisy * mask + latents * (1 - mask)

            # Start clipping after 90% done
            """
            if idx / init_timestep > 0.9:
                latents = (
                    self.encode_samples(
                        self.decode_latent(latents / self.latent_scale).clamp(0, 1)
                    )
                    * self.latent_scale
                )
            """

        # Re-scaling & clipping the latent
        result = self.decode_latent(latents / self.latent_scale).clamp(0, 1)

        if self.export_model == True:
            print("Export torch model to be ModifiedWav2Vec2Model.onnx ...")
            torch.onnx.export(self.audio_encoder, (torch.rand(1, 61600), ), self.ov_model_path + '/ModifiedWav2Vec2Model.onnx')
            print("Export torch model to be UNet1DConditionModel.onnx ...", end="")
            torch.onnx.export(self.denoiser, (torch.rand(2,231,32), torch.tensor([100, 100]), torch.rand(2,231,768),), 
                             self.ov_model_path + '/UNet1DConditionModel.onnx')
            print(" done")
            self.export_model = False
        return SAIDInferenceOutput(result=result, intermediates=intermediates)


class SAID_UNet1D(SAID):
    """SAiD model implemented using U-Net 1D model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Type[SchedulerMixin] = DDIMScheduler,
        in_channels: int = 32,
        feature_dim: int = -1,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
        prediction_type: str = "epsilon",
        use_ov: bool = True,
        ov_model_path: str = "./",
        device_name: str = "GPU",
        convert_model: bool = False,
        export_model: bool = False,
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler: Type[SchedulerMixin]
            Noise scheduler, by default DDIMScheduler
        in_channels : int
            Dimension of the input, by default 32
        feature_dim : int
            Dimension of the latent feature, by default -1
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        prediction_type: str
            Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
        """
        super().__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            in_channels=in_channels,
            feature_dim=feature_dim,
            diffusion_steps=diffusion_steps,
            latent_scale=latent_scale,
            prediction_type=prediction_type,
            use_ov=use_ov,
            ov_model_path= ov_model_path,
            device_name=device_name,
            convert_model=convert_model,
            export_model=export_model,
        )

        # Denoiser
        if use_ov == True:
            core = ov.Core()
            print("ov::compiled_model: ", self.ov_model_path + "/UNet1DConditionModel.xml, device_name = ", self.device_name, " ...", end="")
            self.ov_denoiser = core.compile_model(self.ov_model_path + "/UNet1DConditionModel.xml",self.device_name, config={'CACHE_DIR': self.ov_model_path})
            print("done")
            print("input:",self.ov_denoiser.inputs)
            print("output:",self.ov_denoiser.outputs)
            print()

        self.denoiser = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=self.feature_dim
            if self.feature_dim > 0
            else self.audio_config.hidden_size,
        )
