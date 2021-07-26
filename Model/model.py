from torch import nn, mean
import os
from transformers import RobertaModel, RobertaConfig
from transformers import logging
logging.set_verbosity_error()


class MFBERT(nn.Module):
    def __init__(self, weights_dir='Model/pre-trained', return_attention=False, inference_method='mean'):
        
        super().__init__()

        self.return_attention = return_attention

        if inference_method not in ['cls','mean', 'MEAN', 'CLS']:
            raise ValueError('Please Enter a valid inference method from {"cls", "mean"}')
        else:
            self.inference = inference_method.lower()
        
        if os.path.isdir(weights_dir) and weights_dir!='':
            self.base = RobertaModel.from_pretrained('Model/pre-trained', output_attentions=return_attention)
            print('Loaded Pre-trained weights...')
        else:
            print('No Pre-trained weights found, initialising...')
            config = RobertaConfig.from_pretrained('Model/config.json')
            self.base = RobertaModel(config)

    def forward(self,inputs):

        all_output = self.base(**inputs)

        if self.return_attention:
            if self.inference=='cls':
                return {'CLS_FINGERPRINT':all_output[1], 'ATTENTION':all_output[2]}

            elif self.inference=='mean':
                return {'MEAN_FINGERPRINT':mean(all_output[0],1), 'ATTENTION':all_output[2]}
        else:
            if self.inference=='cls':
                return all_output[1]

            elif self.inference=='mean':
                return mean(all_output[0],1)          