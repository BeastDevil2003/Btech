import torch
import torch.nn as nn

 
from .spatial import SpatialStream
from .frequency import FrequencyStream
from .fusion import CrossAttentionFusion
from .attention import CBAMLite
from .classifier import ClassifierHead
from .noise import NoiseInjection  


class DeepFakeDetectionModel(nn.Module):      #
    def __init__(self,
        num_classes=2,
        use_spatial=True,
        use_frequency=True,
        use_attention=True):



        super().__init__()
        
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_attention = use_attention

        self.noise = NoiseInjection(std=0.03)  

        self.spatial_stream = SpatialStream()
        self.frequency_stream = FrequencyStream()

        self.fusion = CrossAttentionFusion(channels=576)

        self.cbam = CBAMLite(channels=576)

        self.classifier = ClassifierHead(
            in_channels=576,
            num_classes=num_classes
        )

    def forward(self, x, return_attention=False):

        x = self.noise(x)

        Fs = self.spatial_stream(x)    
        Ff = self.frequency_stream(x)   

        Fused, Ws, Wf = self.fusion(Fs, Ff)


        F_att, ca, sa = self.cbam(Fused)

        logits = self.classifier(F_att)

        if return_attention:
            return {
                "logits": logits,
                "fusion_spatial_att": Ws,
                "fusion_frequency_att": Wf,
                "channel_attention": ca,
                "spatial_attention": sa
            }

        return logits

