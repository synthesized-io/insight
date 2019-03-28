def quality_pct(old_score, new_score):
    change = (new_score - old_score) / abs(old_score)
    return max(1 + change, 0) * 100.0
