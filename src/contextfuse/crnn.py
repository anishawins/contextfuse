"""
CRNN: a convolutional-recurrent network for word recognition.
Architecture follows Shi, Bai & Yao (2015), "An End-to-End Trainable Neural
Network for Image-Based Sequence Recognition".

THE PROBLEM THIS SOLVES, and why it is not just "a CNN with a classifier":

    A word image contains an unknown number of characters, in unknown
    positions, of unknown widths. A classifier needs a fixed number of
    outputs. The obvious fix - segment the image into characters first, then
    classify each - fails badly in practice, because segmenting cursive or
    touching or blurred characters is as hard as reading them.

    CRNN removes segmentation entirely. Three stages:

    1. CNN. Slide convolutions over the image. Crucially, the later pooling
       layers use stride (2,1): they halve the HEIGHT but preserve the WIDTH.
       After seven conv layers the feature map is 1 pixel tall and ~24 wide.

    2. MAP TO SEQUENCE. Read that 1 x 24 x 512 map as a SEQUENCE of 24
       vectors, left to right. Each vector is a "frame" describing a narrow
       vertical slice of the original image - roughly 4 pixels wide. This
       reinterpretation is the whole trick: a 2-D image has become a 1-D
       sequence without anyone deciding where characters start and end.

    3. BiLSTM. Each frame is ambiguous on its own - a slice through the
       middle of an 'm' looks like a slice through 'n'. Context disambiguates
       it. BIdirectional matters because the evidence can lie on either side:
       you may need the frames to the RIGHT to know that what you are looking
       at is the left half of a 'w'.

    Then CTC (see ctc.py) turns the 24 frame-predictions into a word of
    whatever length, without ever being told which frame produced which
    character.

INPUT: grayscale, height 32, width 100 (the standard MJSynth preprocessing).
OUTPUT: [T=24, batch, n_classes] logits, where class 0 is the CTC blank.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_classes: int, img_height: int = 32,
                 n_channels: int = 1, lstm_hidden: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        if img_height != 32:
            raise ValueError("this configuration assumes height 32")

        # kernel / stride / padding / out-channels per conv layer
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv(i: int, batch_norm: bool = False) -> None:
            c_in = n_channels if i == 0 else nm[i - 1]
            cnn.add_module(f"conv{i}", nn.Conv2d(c_in, nm[i], ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f"bn{i}", nn.BatchNorm2d(nm[i]))
            cnn.add_module(f"relu{i}", nn.ReLU(inplace=True))

        conv(0)
        cnn.add_module("pool0", nn.MaxPool2d(2, 2))            # 32x100 -> 16x50
        conv(1)
        cnn.add_module("pool1", nn.MaxPool2d(2, 2))            # 16x50  -> 8x25
        conv(2, batch_norm=True)
        conv(3)
        # (2,1) pooling: HEIGHT halves, WIDTH is preserved. Preserving width is
        # what keeps the output sequence long enough to hold every character -
        # CTC cannot emit more characters than it has frames.
        cnn.add_module("pool2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))   # -> 4x26
        conv(4, batch_norm=True)
        conv(5)
        cnn.add_module("pool3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))   # -> 2x27
        conv(6, batch_norm=True)                                        # -> 1x26
        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, lstm_hidden, lstm_hidden, dropout),
            BidirectionalLSTM(lstm_hidden, lstm_hidden, n_classes, 0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, 32, W]  ->  logits [T, B, n_classes]"""
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        if h != 1:
            raise RuntimeError(
                f"expected the conv stack to reduce height to 1, got {h}. "
                f"Input height must be 32.")
        conv = conv.squeeze(2)            # [B, C, W]
        conv = conv.permute(2, 0, 1)      # [W, B, C]  == [T, B, feature]
        return self.rnn(conv)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BidirectionalLSTM(nn.Module):
    """
    LSTM over the frame sequence, then a linear layer applied per timestep.

    WHY LSTM AND NOT A PLAIN RNN: the useful context can be several frames
    away, and a vanilla RNN's gradient shrinks geometrically with distance -
    it cannot learn to carry information that far. The LSTM's cell state is
    an additive path, so gradients flow along it without vanishing, and its
    gates learn what to keep and what to discard.

    WHY BIDIRECTIONAL: the frame at position t is disambiguated by evidence
    on BOTH sides. Reading left-to-right only, the first half of a 'w' is
    indistinguishable from a 'v'.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(n_hidden * 2, n_out)    # *2: forward + backward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                  # [T, B, 2*hidden]
        return self.fc(self.drop(out))        # [T, B, n_out]
