"""Explicit backbone I/O and prediction-coordinate adapters."""
import torch
from .registry import register,build

@register('model','adapter')
class ModelAdapter(torch.nn.Module):
    def __init__(self,model,time_keyword=None,output_index=None,defaults=None,time_first=False):
        super().__init__();self.model=build('model',model);self.time_keyword=time_keyword;self.index=output_index;self.defaults=defaults or {};self.time_first=time_first
    def forward(self,state,time,**kwargs):
        kwargs={**self.defaults,**kwargs}
        if self.time_first:out=self.model(time,state,**kwargs)
        elif self.time_keyword is None:out=self.model(state,time,**kwargs)
        else:out=self.model(state,**{self.time_keyword:time},**kwargs)
        return out if self.index is None else out[self.index]

@register('prediction_adapter','tuple_item')
class TupleItem:
    def __init__(self,index=0):self.index=index
    def __call__(self,prediction,batch):return prediction[self.index]

@register('prediction_adapter','mdm_loss_coordinates')
class MDMCoordinates:
    def __call__(self,prediction,batch):
        source=batch.metadata['prediction'];target=batch.metadata['loss_target'];gammas=batch.metadata['gammas']
        if source==target:return prediction
        outputs=[]
        for state,pred,g in zip(batch.state,prediction,gammas):
            if source=='v':clean=state*g.sqrt()-pred*(1-g).sqrt()
            elif source=='epsilon':clean=(state-pred*(1-g).sqrt())/g.sqrt()
            else:raise ValueError('Unsupported nested prediction')
            # Native intentionally converts via clean rather than simplification.
            if target=='epsilon':out=(state-clean*g.sqrt())/(1-g).sqrt()
            elif target=='v':out=(g.sqrt()*state-clean)/(1-g).sqrt()
            else:raise ValueError('Unsupported nested loss target')
            outputs.append(out)
        return outputs

@register('prediction_adapter','ssd_output')
class SSDOutput:
    def __call__(self,prediction,batch):
        from .operators import resize
        rs=batch.metadata['rs'];rt=batch.metadata['rt']
        if rs!=rt:
            if prediction.shape[-1]!=batch.metadata['expected_output_size']:raise ValueError('Native SSD model output size mismatch')
            if prediction.shape[-1]!=rs:prediction=resize(prediction,size=(rs,rs),mode='bilinear',antialias=True)
        return prediction
