"""Terminal output; separate marginal rank, never either-reject logic."""
from .stopped_candidate_rank import stopped_candidate_rank

def attach_terminal(out, observed_terminal, calibration_terminal):
    rank=stopped_candidate_rank(observed_score=observed_terminal,
        observed_full_window_complete=out['observed_full_window_complete'],
        observed_score_valid_for_rank=out['observed_score_valid_for_rank'],
        calibration_scores=calibration_terminal,
        calibration_full_window_complete=out['calibration_full_window_complete'],
        calibration_score_valid_for_rank=out['calibration_score_valid_for_rank'],alpha=out['alpha'])
    rank['score_policy']='terminal_prefix_observed_complete'
    rank['statistic']='terminal_slope_energy'
    out['terminal']={'observed_score':observed_terminal,'calibration_scores':calibration_terminal,
        'rank':rank,'scope':'fixed terminal statistic branch; composite decision is reported by hs_grouped_composite'}
    return out
