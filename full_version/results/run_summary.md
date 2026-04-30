# BattLeDIM Stage 2 run summary

## Run context

Run folder: `/home/ali/Documents/iot_project/iot_project/battledim_stage2_runs/run_20260419_180509`

Resolved data path: `data_full_2018_2019_fixed.csv`

Resolved config path: `battledim_stage1_tuned_config.json`

## Key takeaways

- 2019 binary labels are all positive in this merged table, so 2019 cannot measure false-alarm rejection for the binary head.
- Validation actual triggers include real Stage 1 hard negatives, so the binary confirmation head is being tested on at least some genuine false-trigger cases.
- Stage 2 accepts 98.6% of actual validation triggers at the chosen threshold.
- Stage 2 accepts 0.0% of real hard-negative validation triggers.
- 2019 localisation top-3 hit rate is 71.6%.
- 2019 localisation top-1 hit rate is 58.2%.

## Packet split summary

```
       split  n_packets  n_actual_triggers  n_actual_positive  n_actual_negative  n_pseudo_hard_negative  n_easy_negative  positive_rate
stage2_train        598                431                428                  3                      17              150       0.715719
  stage2_cal        212                145                143                  2                      17               50       0.674528
  stage2_val        212                145                143                  2                      17               50       0.674528
   test_2019        730                730                730                  0                       0                0       1.000000
```

## Stage 1 metrics on late 2018

```
              metric    value
     timestep_recall 1.000000
  timestep_precision 0.991102
 false_positive_rate 0.998918
     active_fraction 0.999990
      truth_fraction 0.991093
      n_truth_events 1.000000
   n_detected_events 1.000000
        event_recall 1.000000
stage2_calls_per_day 2.001755
stage2_call_fraction 0.006950
```

## Stage 1 metrics on 2019

```
              metric    value
     timestep_recall 1.000000
  timestep_precision 1.000000
 false_positive_rate 0.000000
     active_fraction 1.000000
      truth_fraction 1.000000
      n_truth_events 1.000000
   n_detected_events 1.000000
        event_recall 1.000000
stage2_calls_per_day 2.000019
stage2_call_fraction 0.006944
```

## Binary Stage 2 metrics on actual validation triggers

```
                  metric      value
               n_packets 145.000000
           positive_rate   0.986207
               precision   1.000000
                  recall   1.000000
                      f1   1.000000
       accepted_fraction   0.986207
                   brier   0.000022
                  pr_auc   1.000000
                 roc_auc   1.000000
recall_at_precision_0.50   1.000000
recall_at_precision_0.60   1.000000
recall_at_precision_0.70   1.000000
recall_at_precision_0.80   1.000000
recall_at_precision_0.90   1.000000
```

## Binary Stage 2 metrics on 2019 actual triggers

```
                  metric      value
               n_packets 730.000000
           positive_rate   1.000000
               precision   1.000000
                  recall   0.964384
                      f1   0.981869
       accepted_fraction   0.964384
                   brier   0.014191
                  pr_auc        NaN
                 roc_auc        NaN
recall_at_precision_0.50        NaN
recall_at_precision_0.60        NaN
recall_at_precision_0.70        NaN
recall_at_precision_0.80        NaN
recall_at_precision_0.90        NaN
```

## Localisation metrics on validation positives

```
        metric    value
macro_f1_pipes 0.285714
 top1_hit_rate 1.000000
 top3_hit_rate 1.000000
     mAP_pipes 0.111888
```

## Localisation metrics on 2019 positives

```
        metric    value
macro_f1_pipes 0.227882
 top1_hit_rate 0.582192
 top3_hit_rate 0.716438
     mAP_pipes 0.360016
```

## Top binary feature groups

```
             group    importance
     raw_flow_tank  0.000000e+00
      raw_pressure -1.480297e-16
top_sensor_context -4.440892e-16
```

## Example 2019 predictions

```
          timestamp           reason  p_any_leak top_1_pipe top_2_pipe top_3_pipe     true_pipes
2019-01-01 07:35:00 periodic_recheck    0.970523     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-01 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-02 07:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-02 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-03 07:35:00 periodic_recheck    0.970508     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-03 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-04 07:35:00 periodic_recheck    0.970885     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-04 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-05 07:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-05 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-06 07:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-06 19:35:00 periodic_recheck    0.941867     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-07 07:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-07 19:35:00 periodic_recheck    0.970527     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-08 07:35:00 periodic_recheck    0.970525     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-08 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-09 07:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-09 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-10 07:35:00 periodic_recheck    0.973124     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
2019-01-10 19:35:00 periodic_recheck    0.999999     pipe_5     pipe_1    pipe_11 pipe_4, pipe_9
```