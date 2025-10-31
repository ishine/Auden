import torch.nn as nn


class EncoderProjector(nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`): The downsample rate to use.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x, x_lens // self.downsample_rate
