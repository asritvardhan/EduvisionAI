import numpy as np

# Engagement mapping from emotions
VOCAL_ENGAGEMENT_MAP = {
    "HAP": 0.9,
    "NEU": 0.6,
    "FEA": 0.4,
    "DIS": 0.4,
    "ANG": 0.2,
    "SAD": 0.2
}

def map_vocal_to_engagement(audio_probs):
    """
    Convert emotion probabilities to engagement score
    """
    score = 0.0
    for emo, prob in audio_probs.items():
        score += VOCAL_ENGAGEMENT_MAP.get(emo, 0.5) * prob
    return float(score)


def compute_video_engagement(video_probs):
    """
    Convert engagement softmax output into scalar
    Engagement classes assumed: [Low, Medium, High, Very High]
    """
    weights = np.array([0.2, 0.5, 0.75, 1.0])
    score = np.sum(video_probs * weights)
    return float(score)


def fuse_engagement(video_score, vocal_score, alpha=0.6):
    """
    Weighted fusion of video + audio engagement
    """
    fused = alpha * video_score + (1 - alpha) * vocal_score

    if fused >= 0.75:
        state = "Highly Engaged"
    elif fused >= 0.5:
        state = "Engaged"
    elif fused >= 0.3:
        state = "Partially Engaged"
    else:
        state = "Disengaged"

    return state, float(fused)
