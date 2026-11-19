"""
Knowledge-distillation framework.

Implements
  1. Temperature-scaled softmax                    softmax(z / T)
  2. Soft-target KD (Hinton et al., 2015)          T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )
  3. Combined hard-label + soft-label objective    (1 - alpha) * CE(z_s, y) + alpha * KD
  4. Feature-based KD (FitNets-style hint loss)    beta * || norm(r(F_s)) - norm(F_t) ||^2
     where r(.) is a learnable 1x1-conv regressor mapping the student's intermediate
     feature map to the teacher's channel dimension and spatial size.

Total loss
    L = (1 - alpha) * L_CE  +  alpha * T^2 * L_KL  +  beta * L_feat

`mode` selects which terms are active:
    "logit"   -> CE + soft-target KD                         (beta ignored)
    "feature" -> CE + feature hint                           (alpha ignored)
    "both"    -> CE + soft-target KD + feature hint          (default)
"""
import torch.nn as nn
import torch.nn.functional as F


def softmax_with_temperature(logits, T):
    """Softens a distribution: higher T -> flatter probabilities that expose
    the teacher's 'dark knowledge' (relative similarity between classes)."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """KL divergence between temperature-softened teacher and student distributions.
    Multiplied by T^2 so that gradient magnitudes stay comparable to the CE term."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_with_temperature(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureAdapter(nn.Module):
    """Regressor that projects student features into the teacher's feature space.

    Only used during training - it is discarded at inference/export time.
    """

    def __init__(self, s_channels, t_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(s_channels, t_channels, 1, bias=False),
                                  nn.BatchNorm2d(t_channels))

    def forward(self, f_s, target_hw):
        if f_s.shape[-2:] != target_hw:
            f_s = F.adaptive_avg_pool2d(f_s, target_hw)
        return self.proj(f_s)


def feature_hint_loss(f_s_proj, f_t):
    """Squared distance between L2-normalised (flattened) feature maps, averaged over the batch.
    Normalisation makes the loss scale-invariant (range [0, 4])."""
    s = F.normalize(f_s_proj.flatten(1), dim=1)
    t = F.normalize(f_t.flatten(1), dim=1)
    return (s - t).pow(2).sum(1).mean()


class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.9, beta=1.0, mode="both"):
        super().__init__()
        assert mode in ("logit", "feature", "both")
        self.T, self.alpha, self.beta, self.mode = T, alpha, beta, mode
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, targets, f_s_proj=None, f_t=None):
        """Returns (total_loss, dict_of_components)."""
        hard = self.ce(s_logits, targets)
        parts = {"ce": hard.item()}
        if self.mode in ("logit", "both"):
            soft = soft_target_loss(s_logits, t_logits, self.T)
            total = (1 - self.alpha) * hard + self.alpha * soft
            parts["kd"] = soft.item()
        else:
            total = hard
        if self.mode in ("feature", "both") and f_s_proj is not None:
            feat = feature_hint_loss(f_s_proj, f_t)
            total = total + self.beta * feat
            parts["feat"] = feat.item()
        return total, parts
