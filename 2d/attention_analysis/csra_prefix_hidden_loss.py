"""Training-only teacher prefix target. No inference architecture change."""
import torch


@torch.no_grad()
def teacher_prefix_hidden(teacher,batch,depth=8):
    if not 0<depth<len(teacher.blocks):raise ValueError('Invalid teacher prefix')
    if any(p.requires_grad for p in teacher.parameters()):raise ValueError('Teacher must be frozen')
    x,t,y=batch['x'],batch['t'],batch['labels']
    with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
        h=teacher.x_embedder(x)
        c=teacher.t_embedder(t)+teacher.y_embedder(y,False)
        cos,sin=teacher.get_rope(x.shape[-2]//teacher.patch_size,x.shape[-1]//teacher.patch_size,x.device)
        for block in teacher.blocks[:depth]:h=block(h,c,cos,sin)
    return h.detach()


def hidden_loss_per_image(student_hidden,teacher_hidden):
    if student_hidden.shape!=teacher_hidden.shape or student_hidden.ndim!=3:
        raise ValueError('Need matched B,N,H teacher/student hidden states')
    target=teacher_hidden.detach().float()
    error=student_hidden.float()-target
    # Normalize each image by its own teacher feature power, not student power.
    return error.square().mean((1,2))/target.square().mean((1,2)).clamp_min(1e-12)
