"""Define the conditional 1D UNet model
"""
import torch
from torch import nn
from .ldm.openaimodel import UNetModel
import openvino as ov

class UNet1DConditionModel(nn.Module):
    """Conditional 1D UNet model"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cross_attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Constructor of the UNet1DConditionModel

        Parameters
        ----------
        in_channels : int
            The number of channels in the input sample
        out_channels : int
            The number of channels in the output
        cross_attention_dim : int
            The dimension of the cross attention features
        dropout : float
            Dropout rate, by default 0.1
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim

        #self.ov_model = None,
        #core = ov.Core()
        #self.ov_model = core.compile_model("/home/openvino-ci-89/river/models/SAiD/UNet1DConditionModel.xml","CPU")
        self.model = UNetModel(
            dims=1,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            model_channels=192,
            num_res_blocks=1,
            attention_resolutions=(1,),
            dropout=dropout,
            channel_mult=(1,),
            num_head_channels=32,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=self.cross_attention_dim,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        """Denoise the input sample

        Parameters
        ----------
        sample : torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Noisy inputs tensor
        timestep : torch.Tensor
            (Batch_size,), (1,), or (), Timesteps
        encoder_hidden_states : torch.Tensor
            (Batch_size, hidden_states_seq_len, cross_attention_dim), Encoder hidden states

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Predicted noise
        """
        out = sample.transpose(1, 2)
        #out_ =out.clone()
        #if self.ov_model is not None:
        #    out = self.ov_model(out, timestep, encoder_hidden_states)
        #else:
        out = self.model(out, timestep, encoder_hidden_states)

        if 0:
            dtype_mapping = {
                torch.float32: ov.Type.f32,
                torch.int64: ov.Type.i64,
                torch.float64: ov.Type.f64,
            }
    
            dummy_inputs = (out_, timestep, encoder_hidden_states)

            input_info=[]
            for input_tensor in dummy_inputs:
                shape = ov.PartialShape(input_tensor.shape)
                element_type = dtype_mapping[input_tensor.dtype]
                input_info.append((shape, element_type))
            with torch.no_grad():    
                ov_model = ov.convert_model(self.model, example_input=dummy_inputs, input=input_info)
            ov.save_model(ov_model, "UNet1DConditionModel.xml")
            del ov_model
            print("ov_model is done!")

        out = out.transpose(1, 2)

        return out
